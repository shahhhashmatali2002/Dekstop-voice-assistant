"""Test both cables before a call, and say which device Teams needs.

    python tools/wire_check.py

Takes about five seconds. It plays a short tone into each cable the
application uses and listens on every endpoint that could carry it, then tells
you - measured, not guessed - which recording device Teams should be pointed
at, or which wire is dead.

Nothing in the application is changed. It opens its own streams, plays its own
tone, and closes them.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from app.audio.devices import AudioUnavailable, DeviceManager  # noqa: E402
from app.audio.probe import probe_wire  # noqa: E402
from app.models.config import Config  # noqa: E402


def show(report) -> None:
    print(f"\n{report.role}")
    print(f"  played into: {report.render!r}")
    if not report.listeners and report.error:
        print(f"  {report.verdict()}")
        return

    for listener in report.listeners:
        if listener.error:
            print(f"    ----     {listener.name}   ({listener.error})")
            continue
        mark = "QUIET" if listener.quiet else ("HEARD" if listener.heard else "  -  ")
        print(
            f"    {mark}  {listener.level_db:7.1f} dB  tone {listener.purity:4.2f}"
            f"   {listener.name}   [{listener.kind}]"
        )
    print(f"  => {report.verdict()}")


def main() -> int:
    config = Config.load()
    devices = config.devices

    if not devices.virtual_mic and not devices.teams_speaker:
        print("No cables are configured yet. Run: python run.py --call-check")
        return 1

    try:
        manager = DeviceManager()
    except AudioUnavailable as exc:
        print(f"No audio available: {exc}")
        return 1

    print("WIRE CHECK - playing a 997 Hz tone into each cable.")
    reports = []
    with manager:
        if devices.virtual_mic:
            reports.append(
                probe_wire(
                    manager,
                    devices.virtual_mic,
                    role="YOUR ENGLISH -> TEAMS (Teams listens to this cable)",
                )
            )
        if devices.teams_speaker:
            reports.append(
                probe_wire(
                    manager,
                    devices.teams_speaker,
                    role="TEAMS -> US (Teams plays into this cable)",
                )
            )

    for report in reports:
        show(report)

    print("\n" + "-" * 62)
    send = next((r for r in reports if "ENGLISH" in r.role), None)
    listen = next((r for r in reports if "TEAMS ->" in r.role), None)

    if send is not None:
        best = max(send.recording, key=lambda l: l.level_db) if send.recording else None
        if best is not None:
            print(f"Teams > Settings > Devices > Microphone:  {best.name}")
        else:
            print("Teams' Microphone: NO device carries our English. This cable")
            print("cannot reach Teams - the client will hear nothing from you.")

    if listen is not None:
        if listen.loopback or listen.recording:
            source = (listen.loopback or listen.recording)[0]
            print(f"Teams > Settings > Devices > Speaker:     {listen.render}")
            print(f"  (we capture it from {source.name!r})")
        else:
            print("Teams' Speaker: this cable carries nothing back to us. You")
            print("will not hear the client translated.")

    both = all(r.ok for r in reports) if reports else False
    print("-" * 62)
    print("Both cables carry sound." if both else "At least one cable is dead - see above.")
    return 0 if both else 1


if __name__ == "__main__":
    raise SystemExit(main())
