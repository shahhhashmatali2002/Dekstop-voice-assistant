#!/usr/bin/env python
"""Start the translator.

    python run.py                 open the application
    python run.py --ui-preview    the overlay only, no audio, no providers
    python run.py --devices       list audio endpoints and exit
    python run.py --check         verify the install and exit

The application opens its overlay whether or not audio is configured. If
devices are missing it comes up in red and opens Settings, where they are
chosen from dropdowns - there is no file to edit by hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def check() -> int:
    ok = True
    print()
    for label, module, hint in (
        ("numpy", "numpy", "pip install numpy"),
        ("soxr (resampler)", "soxr", "pip install soxr"),
        ("PySide6 (interface)", "PySide6", "pip install PySide6"),
        ("PyAudioWPatch (audio)", "pyaudiowpatch", "pip install PyAudioWPatch"),
    ):
        try:
            __import__(module)
            print(f"  ok       {label}")
        except ImportError:
            ok = False
            print(f"  MISSING  {label:<24} {hint}")

    if sys.platform != "win32":
        print("\n  note: audio and the overlay need Windows; "
              f"this is {sys.platform}.")

    from app.models.config import Config, missing_secrets

    cfg = Config.load()
    d = cfg.devices
    print()
    for label, value in (
        ("Microphone", d.microphone),
        ("Headphones", d.headphones),
        ("Translator microphone", d.virtual_mic),
        ("Teams speaker", d.teams_speaker),
    ):
        print(f"  {'ok      ' if value else 'not set '} {label:<24} {value}")

    missing = missing_secrets(cfg)
    print()
    if missing:
        print(f"  API keys still needed: {', '.join(missing)}")
    else:
        print("  API keys: all present")
    print("\n  Set devices and keys from Settings inside the application.\n")
    return 0 if ok else 1


def devices() -> int:
    from app.audio.devices import AudioUnavailable, DeviceManager, format_device_table

    try:
        with DeviceManager() as dm:
            print("\n  Playback and capture endpoints\n")
            print(format_device_table(dm.wasapi_devices()))
            print("\n  Loopback endpoints (how we hear Teams)\n")
            print(format_device_table(dm.loopback_devices()))
            print()
    except AudioUnavailable as exc:
        print(f"\n  {exc}\n")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Real-time Urdu <-> English meeting translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ui-preview", action="store_true",
                    help="overlay only, scripted data, no audio")
    ap.add_argument("--devices", action="store_true",
                    help="list audio endpoints and exit")
    ap.add_argument("--check", action="store_true",
                    help="verify the install and exit")
    ap.add_argument("--mode", choices=("video", "meeting"),
                    help="'video': something is playing and you want to "
                         "understand it - one cable, microphone off. "
                         "'meeting': a two-way Teams call. Asked once and "
                         "remembered if not given.")
    ap.add_argument("--allow-capture", action="store_true",
                    help="let every window appear in screenshots and screen "
                         "shares (they are excluded by default, §2.4)")
    args = ap.parse_args()

    if args.check:
        return check()
    if args.devices:
        return devices()

    try:
        import PySide6  # noqa: F401
    except ImportError:
        print(
            "\n  PySide6 is not installed, so there is no interface to show.\n"
            "      pip install PySide6\n"
        )
        return 2

    if args.ui_preview:
        from app.views.demo import main as demo_main

        print("\n  Overlay preview. Click the bubble. States cycle every 7s.\n")
        return demo_main()

    from app.controllers.main import main as app_main

    return app_main(allow_capture=args.allow_capture, mode=args.mode or "")


if __name__ == "__main__":
    raise SystemExit(main())
