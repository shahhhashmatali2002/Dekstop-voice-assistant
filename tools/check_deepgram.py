"""Ask the recogniser directly, and print exactly what it says back.

    python tools/check_deepgram.py            # listen to the client, via the
                                              # same cable the app uses
    python tools/check_deepgram.py --mic      # listen to your own microphone

Streams live audio to Deepgram exactly the way the application does - same URL,
same parameters, same 100 ms chunks - and prints EVERY message that comes back,
raw, plus the reason if the socket is dropped.

This exists because inside the application every failure here arrives as the
same sentence: "speech is reaching the recogniser but nothing is coming back,
check the network and DEEPGRAM_API_KEY". That sentence is a guess. It covers a
rejected key, an exhausted account, a refused parameter, a language mismatch
and a dropped websocket, which have five different fixes. Deepgram says which
one it is; the application throws that away. This does not.

Nothing in the application is changed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from app.audio import convert  # noqa: E402
from app.audio.devices import AudioUnavailable, DeviceManager  # noqa: E402
from app.audio.streams import CaptureSource  # noqa: E402
from app.models.config import Config, get_secret  # noqa: E402
from app.providers.base import AsrConfig  # noqa: E402
from app.providers.deepgram_asr import DeepgramSpeechToText  # noqa: E402

RATE = 16_000
SECONDS = 20.0


def pick_device(manager, use_mic: bool):
    """The same endpoint the application would be listening to."""
    config = Config.load()
    if use_mic:
        name = config.devices.microphone
        if not name:
            return manager.default_input(), "your microphone (system default)"
        return manager.find(name, capture=True), f"your microphone {name!r}"

    name = config.devices.teams_speaker
    if not name:
        raise AudioUnavailable(
            "no listening cable is configured - run: python run.py --call-check"
        )
    render = manager.find(name, capture=False)
    return manager.loopback_for_output(render), f"the client, via {name!r}"


async def run(source: CaptureSource, label: str) -> int:
    key = get_secret("DEEPGRAM_API_KEY")
    if not key:
        print("DEEPGRAM_API_KEY is not set anywhere the application can see it.")
        return 1
    print(f"Key: {key[:4]}...{key[-4:]} ({len(key)} characters)")

    import websockets

    provider = DeepgramSpeechToText(api_key=key)
    url = provider._url(AsrConfig(language="en", rate=RATE, interim_results=True))
    print(f"Endpoint: {url.split('?')[0]}")
    print(f"Parameters: {url.split('?', 1)[1][:200]}")

    try:
        socket = await websockets.connect(
            url, additional_headers={"Authorization": f"Token {key}"}
        )
    except Exception as exc:
        print(f"\nREFUSED AT CONNECT: {type(exc).__name__}: {exc}")
        print("  401 -> the key is wrong or revoked")
        print("  402 / 403 -> the account has no credit left")
        print("  anything else -> network, VPN or firewall")
        return 1

    print(f"\nConnected. Listening to {label} for {SECONDS:.0f}s - TALK NOW.\n")

    replies = 0
    transcripts = 0
    closed = ""

    async def reader():
        nonlocal replies, transcripts, closed
        try:
            async for raw in socket:
                replies += 1
                try:
                    message = json.loads(raw)
                except Exception:
                    print(f"  <- (not json) {raw!r}")
                    continue

                kind = message.get("type", "?")
                if message.get("type") == "Error" or message.get("error"):
                    print(f"  <- ERROR: {json.dumps(message)[:400]}")
                    continue

                alternatives = (message.get("channel") or {}).get("alternatives") or [{}]
                text = (alternatives[0].get("transcript") or "").strip()
                if text:
                    transcripts += 1
                    final = message.get("is_final") or message.get("speech_final")
                    print(f"  <- {kind}: {text!r}   (final={bool(final)})")
                else:
                    print(f"  <- {kind}: {json.dumps(message)[:150]}")
        except Exception as exc:
            closed = f"{type(exc).__name__}: {exc}"

    task = asyncio.create_task(reader())

    sent_ms = 0.0
    loudest = -99.0
    deadline = time.monotonic() + SECONDS
    source.ring.discard()
    try:
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            raw = source.ring.read_available()
            if raw.size == 0:
                continue
            pcm = convert.endpoint_to_pipeline(raw, source.rate, source.channels, RATE)
            if pcm.size == 0:
                continue
            rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
            if rms > 0:
                loudest = max(loudest, 20.0 * np.log10(rms / 32768.0))
            sent_ms += 1000.0 * pcm.size / RATE
            await socket.send(pcm.astype(np.int16).tobytes())
    finally:
        with __import__("contextlib").suppress(Exception):
            await socket.send(json.dumps({"type": "CloseStream"}))
        await asyncio.sleep(2.0)
        task.cancel()

    print("\n" + "-" * 62)
    print(f"sent            {sent_ms / 1000:.1f}s of audio")
    print(f"loudest         {loudest:.1f} dB   "
          f"({'silence - nothing was playing' if loudest < -70 else 'real audio'})")
    print(f"replies         {replies}")
    print(f"with words      {transcripts}")
    if closed:
        print(f"socket closed   {closed}")
    print("-" * 62)

    if replies == 0:
        print("Deepgram accepted the connection and then said NOTHING - not even")
        print("the empty keepalive frames it normally sends. That is the network:")
        print("something between you and api.deepgram.com is holding the websocket")
        print("open but dropping its traffic. A VPN or corporate firewall does this.")
        return 1
    if transcripts == 0 and loudest < -70:
        print("Deepgram was talking to us, but we sent it silence. Nothing was")
        print("playing into that cable - this is not a recogniser problem.")
        return 1
    if transcripts == 0:
        print("Deepgram replied but recognised no words in audio that was NOT")
        print("silent. Check the language: this asks for English. If the audio")
        print("was Urdu, no English words is the correct answer.")
        return 1
    print("The recogniser works. Words came back.")
    return 0


def main() -> int:
    use_mic = "--mic" in sys.argv
    try:
        manager = DeviceManager()
    except AudioUnavailable as exc:
        print(f"No audio available: {exc}")
        return 1

    with manager:
        try:
            device, label = pick_device(manager, use_mic)
        except Exception as exc:
            print(f"Could not choose a device: {exc}")
            return 1

        source = CaptureSource(manager, device)
        source.start()
        try:
            return asyncio.run(run(source, label))
        finally:
            with __import__("contextlib").suppress(Exception):
                source.stop()


if __name__ == "__main__":
    raise SystemExit(main())
