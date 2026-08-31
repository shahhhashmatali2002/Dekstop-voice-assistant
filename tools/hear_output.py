"""Is the translated English actually on Teams' microphone?

Run it WHILE the app is running, then speak Urdu:

    python tools/hear_output.py

It listens on the capture half of the cable the app writes English into -
which is the exact device Teams is set to use as its microphone - and prints a
live level. Nothing is played and nothing is changed.

It answers the one question that separates two completely different faults:

  * levels move  -> the app IS delivering English to Teams' microphone, so the
                    silence is Teams' own doing: muted, or noise suppression
                    gating synthesised speech.
  * levels flat  -> the app is not writing to that cable at all, and the fault
                    is in this codebase.
"""

from __future__ import annotations

import contextlib
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")

from app.audio.devices import AudioUnavailable, DeviceManager  # noqa: E402
from app.audio.probe import driver_of  # noqa: E402
from app.audio.streams import CaptureSource  # noqa: E402
from app.models.config import Config  # noqa: E402

SECONDS = 20.0
TICK = 0.25


def db(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -99.0
    rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))
    return max(-99.0, 20.0 * math.log10(rms / 32768.0 + 1e-12))


def bar(value: float) -> str:
    filled = int(max(0.0, min(1.0, (value + 60.0) / 60.0)) * 30)
    return "#" * filled + "." * (30 - filled)


def main() -> int:
    config = Config.load()
    cable = config.devices.virtual_mic
    if not cable:
        print("No English cable configured. Run: python run.py --call-check")
        return 1

    try:
        manager = DeviceManager()
    except AudioUnavailable as exc:
        print(f"No audio available: {exc}")
        return 1

    with manager:
        driver = driver_of(cable)
        listeners = [
            d for d in manager.wasapi_devices()
            if d.can_capture and not d.is_loopback and driver_of(d.name) == driver
        ]
        if not listeners:
            print(f"No recording endpoint belongs to {cable!r}.")
            return 1

        print(f"\nThe app writes English into: {cable}")
        print("Teams should be set to the endpoint below that MOVES.\n")

        sources = []
        for device in listeners:
            try:
                source = CaptureSource(manager, device)
                source.start()
                sources.append((device.name, source))
            except Exception as exc:
                print(f"  could not open {device.name}: {exc}")

        if not sources:
            return 1

        width = max(len(name) for name, _ in sources)
        peaks = {name: -99.0 for name, _ in sources}
        print(f"SPEAK URDU INTO THE APP NOW - watching for {SECONDS:.0f}s\n")

        deadline = time.monotonic() + SECONDS
        try:
            while time.monotonic() < deadline:
                time.sleep(TICK)
                for name, source in sources:
                    heard = source.ring.read_available()
                    level = db(heard)
                    peaks[name] = max(peaks[name], level)
                    print(f"{name:<{width}}  {level:6.1f}  {bar(level)}")
                print("-" * (width + 42))
        except KeyboardInterrupt:
            pass
        finally:
            for _, source in sources:
                with contextlib.suppress(Exception):
                    source.stop()

    print("\nLOUDEST EACH ENDPOINT GOT:\n")
    for name, value in sorted(peaks.items(), key=lambda kv: -kv[1]):
        state = "CARRIED THE ENGLISH" if value > -60.0 else "silent"
        print(f"  {value:6.1f} dB   {name:<{width}}  {state}")

    best = max(peaks.values())
    print()
    if best > -60.0:
        print("The app IS delivering English to that cable. So Teams is")
        print("receiving it and not sending it on. Two causes, both in Teams:")
        print("  1. you are muted in the call")
        print("  2. Settings > Devices > Noise suppression is not Off -")
        print("     Teams classifies synthesised speech as noise and gates it")
        return 0

    print("The app wrote NOTHING to that cable while you were speaking, even")
    print("though the transcript showed a translation. The fault is in the")
    print("app's playback path, not in Teams. Send me this output.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
