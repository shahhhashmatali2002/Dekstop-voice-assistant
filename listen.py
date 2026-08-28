#!/usr/bin/env python
"""Listen to anything in Urdu.

A one-way tool: whatever an application is playing in English comes out of
your headphones in Urdu, and the English does not. YouTube, a Teams call, a
recording - it is a device, not an integration, so anything Windows can point
at the cable works.

    python listen.py                      the window
    python listen.py --console            no interface; starts straight away
    python listen.py --devices            list playback endpoints
    python listen.py --check              what is installed and configured

    python listen.py --source "CABLE Input" --out "Headphones" --save

ONE-TIME SETUP
--------------
1. Install VB-CABLE (vb-audio.com/Cable) and reboot.
2. Tell this tool the cable is the source:
       python listen.py --source "CABLE Input" --save
3. Route the application you want translated into the cable:
       Windows Settings > System > Sound > Volume mixer
       > (Chrome, or Teams) > Output device > CABLE Input
   Only that application moves. Everything else on the machine is untouched,
   and your headphones stay your default device.

Then start it, press Start, and play the video. Before Start you hear the
original English straight through; after Start you hear Urdu only.

WHY A CABLE AT ALL
------------------
What we capture is the render stream of the endpoint we capture from, so
muting the endpoint mutes our own input too. Hearing only the Urdu therefore
means the English has to be rendered somewhere you are not listening. See the
header of app/controllers/listen.py for the alternative and why it is not
built yet.
"""

from __future__ import annotations

import argparse
import contextlib
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.models.config import Config, missing_secrets  # noqa: E402


def _utf8_console() -> None:
    """Urdu into a Windows console that still defaults to a 1252 code page."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")


# ----------------------------------------------------------------------
# information commands
# ----------------------------------------------------------------------


def cmd_devices() -> int:
    from app.audio.devices import AudioUnavailable, DeviceManager, format_device_table
    from app.controllers.listen import cable_scan

    try:
        with DeviceManager() as dm:
            print("\n  Playback endpoints - route the application to one of these\n")
            print(format_device_table([d for d in dm.wasapi_devices() if d.can_render]))
            print("\n  Loopback endpoints - how we hear it (resolved for you)\n")
            print(format_device_table(dm.loopback_devices()))
            print("\n  Virtual cable check\n")
            print(cable_scan(dm.all_devices()))
            print()
    except AudioUnavailable as exc:
        print(f"\n  {exc}\n")
        return 2
    return 0



def _check_devices(cfg: Config) -> bool:
    """Does the configured source actually resolve to something we can capture?

    --check is the command people run when it will not start, so it has to
    answer the real question rather than echo the setting back at them.
    """
    from app.audio.devices import AudioUnavailable, DeviceManager
    from app.controllers.listen import _playback_hint

    if not cfg.listen.source:
        return False
    try:
        dm = DeviceManager()
    except AudioUnavailable as exc:
        print(f"\n  Cannot enumerate devices: {exc}")
        return False

    try:
        with dm:
            try:
                source = dm.find(cfg.listen.source, capture=False)
            except AudioUnavailable as exc:
                print(f"\n  Source does not resolve: {exc}")
                hint = _playback_hint(dm)
                if hint:
                    print(hint)
                return False

            print(f"\n  ok       Source resolves to      {source.name}")
            try:
                loopback = dm.loopback_for_output(source)
            except AudioUnavailable as exc:
                print(f"  MISSING  Loopback              {exc}")
                return False
            print(f"  ok       Captured through       {loopback.name}")
            return True
    except AudioUnavailable as exc:  # pragma: no cover - platform dependent
        print(f"\n  {exc}")
        return False


def cmd_check(cfg: Config) -> int:
    ok = True
    print()
    for label, module, hint in (
        ("numpy", "numpy", "pip install numpy"),
        ("soxr (resampler)", "soxr", "pip install soxr"),
        ("PySide6 (window)", "PySide6", "pip install PySide6"),
        ("PyAudioWPatch (audio)", "pyaudiowpatch", "pip install PyAudioWPatch"),
    ):
        try:
            __import__(module)
            print(f"  ok       {label}")
        except ImportError:
            # PySide6 is not fatal here: --console needs no interface at all.
            if module != "PySide6":
                ok = False
            print(f"  MISSING  {label:<24} {hint}")

    if sys.platform != "win32":
        print(f"\n  note: audio needs Windows; this is {sys.platform}.")

    print()
    src = cfg.listen.source
    out = cfg.listen.output
    print(f"  {'ok      ' if src else 'not set '} Source (the cable)      "
          f"{src or 'required - e.g. CABLE Input'}")
    print(f"  {'ok      ' if out else 'default '} Output (your ears)      "
          f"{out or 'whatever Windows is using'}")
    print(f"           Urdu voice              {cfg.providers.urdu_voice}")

    ok = _check_devices(cfg) and ok

    missing = missing_secrets(cfg)
    print()
    if missing:
        print(f"  API keys still needed: {', '.join(missing)}")
        print("  Set them in the main application's Settings, or as environment "
              "variables.")
    else:
        print("  API keys: all present")
    print()
    return 0 if ok else 1


# ----------------------------------------------------------------------
# shared startup
# ----------------------------------------------------------------------


def open_audio(cfg: Config):
    """Returns (manager, ports, problems, notes). ports is None on any problem."""
    from app.audio.devices import AudioUnavailable, DeviceManager
    from app.controllers.listen import build_listen_ports

    try:
        dm = DeviceManager()
    except AudioUnavailable as exc:
        return None, None, [str(exc)], []
    ports, problems, notes = build_listen_ports(cfg, dm)
    return dm, ports, problems, notes


def warn_about_keys(cfg: Config) -> list[str]:
    missing = missing_secrets(cfg)
    if missing:
        return [
            "No " + ", ".join(missing) + " configured. Recognition or "
            "synthesis will fail as soon as you press Start, and you will "
            "fall back to the original audio."
        ]
    return []


# ----------------------------------------------------------------------
# console mode
# ----------------------------------------------------------------------


def format_health(snapshot: dict) -> str:
    """One line that says which of the three fluency problems you have.

    They sound similar and have different fixes, so the numbers matter:

      lag      how far behind the speaker you are, in seconds
      stretch  1.00 means the voice is untouched; above it the audio is being
               time-compressed to catch up, heard as a metallic edge
      dropped  whole sentences thrown away because the lag got out of hand -
               heard as missing words
      gaps     ring underruns: the sink ran dry mid-utterance and played
               silence, heard as a stutter inside a word
    """
    inc = snapshot.get("metrics", {}).get("incoming", {})
    audio = snapshot.get("metrics", {}).get("audio", {})
    ttfta = (inc.get("ttfta") or {}).get("p50")
    out = audio.get("output") or {}
    return (
        f"  [{snapshot.get('status', '?')}] "
        f"lag {inc.get('queue_depth_ms', 0) / 1000:.1f}s  "
        f"stretch {inc.get('drift_rate', 1.0):.2f}x  "
        f"units {inc.get('units', 0)}  "
        f"dropped {inc.get('dropped', 0)}  "
        f"gaps {out.get('underruns', 0)}  "
        + (f"behind {ttfta / 1000:.1f}s" if ttfta else "")
    )


def run_console(cfg: Config) -> int:
    from app.controllers.listen import ListenEngine
    from app.providers.factory import build_providers

    dm, ports, problems, notes = open_audio(cfg)
    for note in notes:
        print(f"  {note}")
    if ports is None:
        print("\n  Cannot start:\n")
        for p in problems:
            print(f"    - {p}")
        print()
        return 2
    for warning in warn_about_keys(cfg):
        print(f"  warning: {warning}")

    def on_transcript(payload: dict) -> None:
        src = (payload.get("source") or "").strip()
        ur = (payload.get("translated") or "").strip()
        if payload.get("dropped"):
            print(f"  EN  {src}\n  --  dropped, not translated")
            return
        if src:
            print(f"  EN  {src}")
        if ur:
            print(f"  UR  {ur}")

    def on_notice(payload: dict) -> None:
        print(f"  [{payload.get('severity', 'info')}] {payload.get('message', '')}")

    def on_status(payload: dict) -> None:
        detail = payload.get("detail") or ""
        print(f"  -- {payload.get('status')}{' - ' + detail if detail else ''}")

    engine = ListenEngine(
        cfg,
        ports,
        build_providers(cfg, include_s2st=False),
        on_transcript=on_transcript,
        on_notice=on_notice,
        on_status=on_status,
    )
    print(f"\n  {ports.describe()}")
    print("  Ctrl+C to stop.\n")
    engine.start()
    engine.start_listening()
    try:
        # A health line now and then, because "it does not sound right" has
        # three different causes and they are told apart by these numbers.
        next_report = time.monotonic() + 15.0
        while True:
            time.sleep(0.5)
            if time.monotonic() >= next_report:
                next_report = time.monotonic() + 15.0
                print(format_health(engine.snapshot()))
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        engine.shutdown()
        if dm is not None:
            with contextlib.suppress(Exception):
                dm.close()
    return 0


# ----------------------------------------------------------------------
# window mode
# ----------------------------------------------------------------------


def run_window(cfg: Config) -> int:
    try:
        from PySide6.QtCore import QObject, QTimer, Signal
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("\n  PySide6 is not installed, so there is no window to show.\n"
              "      pip install PySide6\n"
              "  Or run without one:  python listen.py --console\n")
        return 2

    from app.controllers.listen import ListenEngine
    from app.providers.factory import build_providers
    from app.views import capture_guard
    from app.views.errors import install as install_errors
    from app.views.listen_window import ListenWindow
    from app.views.transcript import TranscriptWindow

    class Bridge(QObject):
        """Engine thread to Qt thread. Queued connections only (§11.5)."""

        status = Signal(dict)
        transcript = Signal(dict)
        notice = Signal(dict)

    qt = QApplication.instance() or QApplication(sys.argv)
    with contextlib.suppress(Exception):
        install_errors(qt)

    # Ctrl+C in the console has to work. Qt's event loop blocks inside C++, so
    # the interpreter never gets a chance to run its signal handlers and SIGINT
    # is swallowed: the window stays up and the terminal looks frozen with no
    # way out but Task Manager. A timer that fires every 200 ms gives Python
    # the opening it needs. Same fix as Application._install_signal_handlers.
    signal.signal(signal.SIGINT, lambda *_: qt.quit())
    sigint_timer = QTimer()
    sigint_timer.setInterval(200)
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start()
    print("\n  Ctrl+C here, or close the window, quits.")

    bridge = Bridge()
    window = ListenWindow()
    subtitles = TranscriptWindow()

    dm, ports, problems, notes = open_audio(cfg)
    if ports is None:
        # Also to the console. Whoever ran this is looking at a terminal, and
        # a window that may be behind it is not where they will look first.
        print("\n  Cannot start:\n")
        for problem in problems:
            print(problem if problem.startswith("    ") else f"    - {problem}")
        print()
        window.set_blocked(problems)
        window.show()
        if not cfg.ui.exclude_from_capture:
            capture_guard.allow_capture(window)
        return qt.exec()

    window.set_devices(ports.describe())
    engine = ListenEngine(
        cfg,
        ports,
        build_providers(cfg, include_s2st=False),
        on_transcript=bridge.transcript.emit,
        on_notice=bridge.notice.emit,
        on_status=bridge.status.emit,
    )

    bridge.status.connect(
        lambda p: window.set_status(p.get("status", "idle"), p.get("detail", ""))
    )
    bridge.transcript.connect(subtitles.add_entry)
    bridge.notice.connect(
        lambda p: subtitles.add_notice(p.get("severity", "info"), p.get("message", ""))
    )

    window.start_clicked.connect(engine.start_listening)
    window.stop_clicked.connect(engine.stop_listening)
    window.pause_clicked.connect(engine.toggle)
    def show_captured(widget) -> None:
        """Show a window, then undo the capture exclusion if it was waived.

        Order matters: OverlayWindow.showEvent is what applies the exclusion,
        and it cannot run before the window has a native handle. Undoing it
        first would simply be overwritten.
        """
        widget.show()
        if not cfg.ui.exclude_from_capture:
            capture_guard.allow_capture(widget)

    def toggle_subtitles() -> None:
        if subtitles.isVisible():
            subtitles.hide()
            return
        show_captured(subtitles)

    window.subtitles_clicked.connect(toggle_subtitles)
    qt.aboutToQuit.connect(engine.shutdown)

    for note in notes:
        subtitles.add_notice("info", note)
    for warning in warn_about_keys(cfg):
        subtitles.add_notice("warning", warning)

    engine.start()
    show_captured(window)
    if cfg.listen.autostart:
        engine.start_listening()
    return qt.exec()


# ----------------------------------------------------------------------


def main() -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(
        description="Hear any application in Urdu",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--console", action="store_true",
                    help="no window; start translating immediately")
    ap.add_argument("--devices", action="store_true",
                    help="list playback endpoints and exit")
    ap.add_argument("--check", action="store_true",
                    help="report the install and the configuration, then exit")
    ap.add_argument("--source", metavar="NAME",
                    help="playback device the application is routed to, "
                         "e.g. 'CABLE Input'")
    ap.add_argument("--out", metavar="NAME",
                    help="where the Urdu is played; default is the Windows default")
    ap.add_argument("--voice", metavar="NAME", help="Urdu TTS voice")
    ap.add_argument("--preset", choices=("fluent", "balanced", "fast"),
                    help="fluency against lag: 'fluent' speaks whole thoughts "
                         "and runs several seconds behind (video), 'fast' "
                         "stays close to real time in short phrases "
                         "(conversation), 'balanced' sits between them")
    ap.add_argument("--autostart", action="store_true",
                    help="begin translating as soon as the window opens")
    ap.add_argument("--allow-capture", action="store_true",
                    help="let the windows appear in screenshots and screen "
                         "shares (they are excluded by default, §2.4)")
    ap.add_argument("--save", action="store_true",
                    help="remember the options given above")
    args = ap.parse_args()

    cfg = Config.load()
    from app.controllers.listen import apply_preset

    if args.preset:
        cfg.listen.preset = args.preset
    try:
        print(f"\n  preset: {apply_preset(cfg, cfg.listen.preset)}")
    except KeyError:
        cfg.listen.preset = "balanced"
        print(f"\n  preset: {apply_preset(cfg, 'balanced')}")
    if args.source:
        cfg.listen.source = args.source
    if args.out:
        cfg.listen.output = args.out
    if args.voice:
        cfg.providers.urdu_voice = args.voice
    if args.autostart:
        cfg.listen.autostart = True
    if args.allow_capture:
        cfg.ui.exclude_from_capture = False
        print("\n  Capture exclusion waived: the window and the subtitles will "
              "now\n  show up in screenshots - and in a Teams screen share. "
              "Do not leave\n  this on for a real call.")
    if args.devices:
        return cmd_devices()
    if args.check:
        return cmd_check(cfg)

    if args.save:
        # The capture waiver is never persisted. It is a debugging convenience
        # and §2.4 is not a preference: a saved "off" would sit there until the
        # call where it matters.
        if args.allow_capture:
            cfg.ui.exclude_from_capture = True
            print("  (not saving --allow-capture; it applies to one run only)")
        # Saving finishes here rather than falling through into the window.
        # `--source X --save` reads as "remember this", and the terminal
        # freezing on a Qt event loop nobody asked for is not what it means.
        cfg.save()
        print(f"\n  saved to {Config.path()}")
        print(f"    source  {cfg.listen.source or '(not set)'}")
        print(f"    preset  {cfg.listen.preset}")
        print(f"    output  {cfg.listen.output or 'system default'}")
        print("\n  Check it resolves:   python listen.py --check")
        print("  Then start it:       python listen.py\n")
        return 0

    if args.console:
        return run_console(cfg)
    return run_window(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
