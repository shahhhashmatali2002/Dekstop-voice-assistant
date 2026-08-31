"""Push one real English sentence through the app's own playback path.

    python tools/say_english.py

Uses the same TTS, the same voice, the same resampling, the same channel
mapping and the same RenderSink the running application uses - and listens on
the capture half of the cable at the same time. So it tests the production path
end to end, with real synthesised speech rather than a tone.

The transcript proves the translation is produced; the cable meter proves
nothing arrives. Everything between those two facts is in here.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from app.audio import convert  # noqa: E402
from app.audio.devices import AudioUnavailable, DeviceManager  # noqa: E402
from app.audio.probe import driver_of  # noqa: E402
from app.audio.streams import CaptureSource, RenderSink  # noqa: E402
from app.models.config import Config  # noqa: E402
from app.providers.cascade_s2st import english_voice  # noqa: E402
from app.providers.tts import EdgeTextToSpeech  # noqa: E402

SENTENCE = "Hello, this is a test of the English voice."


def db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -99.0
    rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))
    return max(-99.0, 20.0 * math.log10(rms / 32768.0 + 1e-12))


async def run(manager, sink, listener) -> int:
    tts = EdgeTextToSpeech()
    voice = english_voice(sink.rate)
    print(f"voice        {voice.name}")
    print(f"sink         {sink.device.name!r}  {sink.rate} Hz  {sink.channels}ch")
    print(f"listening on {listener.device.name!r}\n")

    listener.ring.discard()
    chunks = 0
    written = 0
    heard: list[np.ndarray] = []

    try:
        async for chunk in tts.synthesize(SENTENCE, voice):
            if chunk.pcm.size == 0:
                continue
            chunks += 1
            pcm = chunk.pcm
            if chunk.rate != sink.rate:
                pcm = convert.resample(pcm, chunk.rate, sink.rate)
            if sink.channels > 1:
                pcm = convert.to_channels(pcm, sink.channels)
            dropped = sink.write(pcm)
            written += pcm.size
            if chunks <= 3:
                print(f"  chunk {chunks}: {chunk.pcm.size} samples at "
                      f"{chunk.rate} Hz -> wrote {pcm.size}, dropped {dropped}")
            # Let the device play it, and collect what comes back.
            for _ in range(6):
                await asyncio.sleep(0.05)
                got = listener.ring.read_available()
                if got.size:
                    heard.append(got)
    except Exception as exc:
        print(f"\nSYNTHESIS FAILED: {type(exc).__name__}: {exc}")
        return 1

    # Drain the tail.
    for _ in range(20):
        await asyncio.sleep(0.05)
        got = listener.ring.read_available()
        if got.size:
            heard.append(got)

    print(f"\nchunks from TTS      {chunks}")
    print(f"samples written      {written}")
    print(f"ring dropped         {sink.ring.dropped_samples}")
    print(f"sink active          {sink.active}")

    if not heard:
        print("\nNothing at all came back from the cable.")
        level = -99.0
    else:
        raw = np.concatenate(heard)
        channels = max(1, listener.channels)
        usable = raw.size - (raw.size % channels)
        mono = raw[:usable].reshape(-1, channels)[:, 0] if channels > 1 else raw
        level = db(mono)
        print(f"heard back           {mono.size} samples, {level:.1f} dB")

    print()
    if chunks == 0:
        print("THE VOICE PRODUCED NOTHING. The English voice name is being")
        print("rejected, or the text was emptied before synthesis.")
        return 1
    if level < -60.0:
        print("Audio WAS written to the sink and NOTHING came back off the")
        print("cable. The write path or the render stream is the fault.")
        return 1
    print("The path works: synthesised English reached the cable at "
          f"{level:.1f} dB.")
    return 0


def main() -> int:
    config = Config.load()
    cable = config.devices.virtual_mic
    if not cable:
        print("No English cable configured. Run: python run.py --call-check")
        return 1

    print("STOP the main app first - it holds this cable.\n")
    try:
        manager = DeviceManager()
    except AudioUnavailable as exc:
        print(f"No audio available: {exc}")
        return 1

    with manager:
        try:
            render = manager.find(cable, capture=False)
        except AudioUnavailable as exc:
            print(f"Could not find {cable!r}: {exc}")
            return 1

        partners = [
            d for d in manager.wasapi_devices()
            if d.can_capture and not d.is_loopback
            and driver_of(d.name) == driver_of(render.name)
        ]
        if not partners:
            print(f"No recording endpoint belongs to {cable!r}.")
            return 1

        sink = RenderSink(manager, render)
        listener = CaptureSource(manager, partners[0])
        sink.start()
        listener.start()
        try:
            return asyncio.run(run(manager, sink, listener))
        finally:
            with contextlib.suppress(Exception):
                sink.stop()
            with contextlib.suppress(Exception):
                listener.stop()


if __name__ == "__main__":
    raise SystemExit(main())
