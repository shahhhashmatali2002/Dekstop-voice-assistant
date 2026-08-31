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


def call_check() -> int:
    """Meeting-mode readiness. The expensive failures happen mid-call."""
    from app.models.config import Config
    from app.controllers.preflight import meeting_preflight

    config = Config.load()
    manager = None
    try:
        from app.audio.devices import DeviceManager

        manager = DeviceManager()
    except Exception as exc:
        print(f"\n  note: cannot open the audio stack ({exc}); "
              "checking the configuration only.")

    # Auto-configuration first, exactly as the application does on startup.
    # Reading the saved settings alone reports blanks the app would have filled
    # in the moment it opened - which is how this said "not set" about a cable
    # installed ten minutes earlier.
    if manager is not None:
        from app.controllers.autoconfig import autoconfigure

        try:
            notes, _problems = autoconfigure(config, manager, save=True)
            for note in notes:
                print(f"  set      {note}")
            if notes:
                print()
        except Exception as exc:
            print(f"  note: could not auto-configure ({exc})")

    report = meeting_preflight(config, manager)
    print("\n  Teams call readiness\n")
    print(report.render())
    print()
    if report.ok:
        print("  Ready. Start a call, then press Start in the app.\n")
    else:
        print("  Not ready yet - the BLOCKED lines above have to be fixed.\n")
    if manager is not None:
        try:
            manager.close()
        except Exception:
            pass
    return 0 if report.ok else 1


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
    ap.add_argument("--call-check", action="store_true",
                    help="is this machine ready to translate a Teams call? "
                         "Checks the two cables, the microphone, the keys and "
                         "the echo risk, and prints what to set in Teams. "
                         "Opens nothing and calls nobody.")
    ap.add_argument("--no-clone-voice", action="store_true",
                    help="do not try to match the speaker's voice at all; "
                         "read everything in the stock Urdu voice")
    ap.add_argument("--clone-voice", action="store_true",
                    help="on by default. Speak the Urdu in the voice of whoever is speaking "
                         "on the source. The voice is learned from the audio "
                         "as it plays, so the first half-minute is in the "
                         "normal voice. The full clone needs a GPU; without "
                         "one it measures itself, says so, and drops to a "
                         "light pitch-and-tone match that any CPU can run.")
    ap.add_argument("--allow-capture", action="store_true",
                    help="let every window appear in screenshots and screen "
                         "shares (they are excluded by default, §2.4)")
    args = ap.parse_args()

    if args.check:
        return check()
    if args.call_check:
        return call_check()
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

    return app_main(
        allow_capture=args.allow_capture,
        mode=args.mode or "",
        clone_voice=None if not (args.clone_voice or args.no_clone_voice)
        else not args.no_clone_voice,
    )


if __name__ == "__main__":
    raise SystemExit(main())
