"""Phase 0 - the audio spine.

Spec reference: §17.2 Phase 0, §17.3 steps 1-7.

Run these in order on the Windows machine. Each step proves exactly one thing
and refuses to continue if it cannot. No AI, no network, no UI - if this
sequence does not pass, nothing downstream matters.

    python tools/phase0.py devices          1. enumerate endpoints
    python tools/phase0.py mic              2. mic level meter
    python tools/phase0.py passthrough      3. mic -> headphones
    python tools/phase0.py loopback         4. capture what a device plays
    python tools/phase0.py tone             5. tone -> virtual mic (Teams hears it)
    python tools/phase0.py wire             6. full two-way passthrough
    python tools/phase0.py soak --minutes 30   7. stability run

Every step prints the health counters at the end. The two numbers that matter
are `overflows` and `underruns`: both must be zero. A non-zero underrun count
means the GIL discipline in §4.3 is not holding, and that is the finding Phase
0 exists to surface.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.audio import convert  # noqa: E402
from app.audio.devices import (  # noqa: E402
    AudioUnavailable,
    DeviceManager,
    format_device_table,
)
from app.audio.formats import DTYPE  # noqa: E402
from app.audio.passthrough import PassthroughPair, PassthroughWire  # noqa: E402
from app.audio.streams import CaptureSource, RenderSink  # noqa: E402

BAR = "=" * 72


def banner(step: str, title: str) -> None:
    print(f"\n{BAR}\n  PHASE 0 - {step}: {title}\n{BAR}")


def report(label: str, health: dict) -> None:
    print(f"\n  {label}")
    for k, v in health.items():
        flag = ""
        if k in ("overflows", "underruns", "late") and v:
            flag = "   <-- INVESTIGATE"
        print(f"    {k:<18} {v}{flag}")


def verdict(*healths: dict) -> int:
    """Return a process exit code: 0 only if every counter that matters is zero."""
    bad = 0
    for h in healths:
        bad += h.get("overflows", 0) + h.get("underruns", 0) + h.get("late", 0)
    print()
    if bad:
        print("  RESULT: FAIL - buffer or timing errors occurred (see above).")
        print("  This is the Phase 0 finding. Do not move on until it is zero.")
        return 1
    print("  RESULT: PASS - no overflows, no underruns, no late callbacks.")
    return 0


# ----------------------------------------------------------------------
# 1. devices
# ----------------------------------------------------------------------


def cmd_devices(args) -> int:
    banner("step 1", "endpoint enumeration")
    with DeviceManager() as dm:
        print("\n  WASAPI endpoints\n")
        print(format_device_table(dm.wasapi_devices()))
        print("\n  Loopback capture endpoints (this is how we hear Teams)\n")
        try:
            print(format_device_table(dm.loopback_devices()))
        except AudioUnavailable as exc:
            print(f"  ERROR: {exc}")
            return 1
        print(f"\n  Default output: {dm.default_output().name}")
        print(f"  Default input:  {dm.default_input().name}")
    print(
        "\n  Note the exact names of: your microphone, your headphones,\n"
        "  and the virtual audio cable's input and output halves.\n"
        "  Every later step takes them as --mic / --out / --virtual-mic."
    )
    return 0


# ----------------------------------------------------------------------
# 2. mic level meter
# ----------------------------------------------------------------------


def cmd_mic(args) -> int:
    banner("step 2", "microphone capture")
    with DeviceManager() as dm:
        dev = dm.find(args.mic, capture=True) if args.mic else dm.default_input()
        print(f"\n  Capturing from: {dev.name}")
        print("  Speak. Ctrl-C to stop.\n")
        with CaptureSource(dm, dev) as src:
            try:
                while True:
                    time.sleep(0.1)
                    chunk = src.ring.read_available()
                    if chunk.size:
                        db = convert.rms_dbfs(chunk)
                        n = 0 if db == float("-inf") else max(0, int((db + 60) / 2))
                        print(f"\r  [{'#' * n:<30}] {db:6.1f} dBFS", end="", flush=True)
            except KeyboardInterrupt:
                print()
            report("capture", src.health())
            return verdict(src.health())


# ----------------------------------------------------------------------
# 3. mic -> headphones
# ----------------------------------------------------------------------


def cmd_passthrough(args) -> int:
    banner("step 3", "microphone -> headphones")
    print(
        "\n  WEAR HEADPHONES. Loudspeakers will feed back (§5.6 F1)."
        "\n  You should hear yourself, live, with no perceptible delay.\n"
    )
    with DeviceManager() as dm:
        mic = dm.find(args.mic, capture=True) if args.mic else dm.default_input()
        out = dm.find(args.out, capture=False) if args.out else dm.default_output()
        print(f"  {mic.name}\n    -> {out.name}\n")

        with CaptureSource(dm, mic) as src, RenderSink(dm, out) as sink:
            wire = PassthroughWire(src, sink, name="mic->phones")
            with wire:
                _run_until_interrupt(args.seconds)
            report("wire", wire.stats.as_dict())
            report("capture", src.health())
            report("render", sink.health())
            return verdict(src.health(), sink.health())


# ----------------------------------------------------------------------
# 4. loopback capture
# ----------------------------------------------------------------------


def cmd_loopback(args) -> int:
    banner("step 4", "loopback capture - hearing what a device plays")
    print(
        "\n  This is Option A from §5.2 and the basis of the incoming pipeline."
        "\n  Play something on the target device (or start a Teams call).\n"
    )
    with DeviceManager() as dm:
        target = dm.find(args.target, capture=False) if args.target else dm.default_output()
        lb = dm.loopback_for_output(target)
        print(f"  Target render device: {target.name}")
        print(f"  Loopback endpoint:    {lb.name}\n")

        with CaptureSource(dm, lb) as src:
            print("  Level of whatever that device is playing. Ctrl-C to stop.\n")
            try:
                silent_for = 0.0
                while True:
                    time.sleep(0.1)
                    chunk = src.ring.read_available()
                    if chunk.size:
                        db = convert.rms_dbfs(chunk)
                        silent_for = silent_for + 0.1 if db < -70 else 0.0
                        n = 0 if db == float("-inf") else max(0, int((db + 60) / 2))
                        tag = "  (silence)" if silent_for > 3 else ""
                        print(
                            f"\r  [{'#' * n:<30}] {db:6.1f} dBFS{tag:<12}",
                            end="",
                            flush=True,
                        )
            except KeyboardInterrupt:
                print()
            report("loopback capture", src.health())
            return verdict(src.health())


# ----------------------------------------------------------------------
# 5. tone -> virtual mic
# ----------------------------------------------------------------------


def cmd_tone(args) -> int:
    banner("step 5", "tone -> virtual microphone (Teams must hear it)")
    print(
        "\n  Set the Teams microphone to the virtual cable, join a test meeting,"
        "\n  and have someone confirm they hear a steady tone."
        "\n  This is the step that proves the outgoing path exists at all.\n"
    )
    with DeviceManager() as dm:
        sink_dev = dm.find(args.virtual_mic, capture=False)
        print(f"  Rendering into: {sink_dev.name}\n")

        with RenderSink(dm, sink_dev) as sink:
            rate, ch = sink.rate, sink.channels
            phase = 0.0
            step = 2 * np.pi * args.freq / rate
            block = 1024
            deadline = time.monotonic() + (args.seconds or 1e9)
            try:
                while time.monotonic() < deadline:
                    if sink.ring.available < rate * ch // 5:  # keep ~200 ms queued
                        t = phase + step * np.arange(block)
                        phase = float(t[-1] + step)
                        tone = (np.sin(t) * 0.25 * 32767).astype(DTYPE)
                        sink.write(convert.to_channels(tone, ch))
                    time.sleep(0.005)
            except KeyboardInterrupt:
                pass
            print()
            report("render", sink.health())
            return verdict(sink.health())


# ----------------------------------------------------------------------
# 6. full two-way wire
# ----------------------------------------------------------------------


def _build_pair(dm, args):
    mic = dm.find(args.mic, capture=True)
    phones = dm.find(args.out, capture=False)
    vmic = dm.find(args.virtual_mic, capture=False)
    teams_out = dm.find(args.teams_speaker, capture=False)
    teams_lb = dm.loopback_for_output(teams_out)

    print(f"  OUTGOING  {mic.name}\n              -> {vmic.name}")
    print(f"  INCOMING  {teams_lb.name}\n              -> {phones.name}\n")

    src_mic = CaptureSource(dm, mic)
    sink_vmic = RenderSink(dm, vmic)
    src_teams = CaptureSource(dm, teams_lb)
    sink_phones = RenderSink(dm, phones)

    for s in (src_mic, sink_vmic, src_teams, sink_phones):
        s.start()

    pair = PassthroughPair(
        outgoing=PassthroughWire(src_mic, sink_vmic, name="mic->vmic"),
        incoming=PassthroughWire(src_teams, sink_phones, name="teams->phones"),
    )
    return pair, (src_mic, sink_vmic, src_teams, sink_phones)


def cmd_wire(args) -> int:
    banner("step 6", "full two-way passthrough")
    print(
        "\n  Both directions live, untranslated. This is the resting state the"
        "\n  application will always be in when it is not translating (§2.5)."
        "\n  In a Teams call the far side should hear you normally, and you"
        "\n  should hear them normally, through our process.\n"
    )
    with DeviceManager() as dm:
        pair, streams = _build_pair(dm, args)
        try:
            with pair:
                _run_until_interrupt(args.seconds)
        finally:
            for s in streams:
                s.stop()
        h = pair.health()
        report("outgoing wire", h["outgoing"]["wire"])
        report("outgoing capture", h["outgoing"]["source"])
        report("outgoing render", h["outgoing"]["sink"])
        report("incoming capture", h["incoming"]["source"])
        report("incoming render", h["incoming"]["sink"])
        return verdict(
            h["outgoing"]["source"],
            h["outgoing"]["sink"],
            h["incoming"]["source"],
            h["incoming"]["sink"],
        )


# ----------------------------------------------------------------------
# 7. soak
# ----------------------------------------------------------------------


def cmd_soak(args) -> int:
    banner("step 7", f"{args.minutes}-minute stability run")
    print(
        "\n  The Phase 0 acceptance criterion: 30 minutes, zero underruns,"
        "\n  zero overflows, flat memory. Leave it running and walk away.\n"
    )
    try:
        import psutil

        proc = psutil.Process()
    except ImportError:
        proc = None
        print("  (pip install psutil for memory tracking)\n")

    with DeviceManager() as dm:
        pair, streams = _build_pair(dm, args)
        deadline = time.monotonic() + args.minutes * 60
        rss0 = proc.memory_info().rss if proc else 0
        try:
            with pair:
                while time.monotonic() < deadline:
                    time.sleep(15)
                    h = pair.health()
                    bad = sum(
                        h[d][s].get(k, 0)
                        for d in ("outgoing", "incoming")
                        for s in ("source", "sink")
                        for k in ("overflows", "underruns", "late")
                    )
                    left = int(deadline - time.monotonic())
                    mem = ""
                    if proc:
                        rss = proc.memory_info().rss
                        mem = f"  rss {rss / 1e6:6.1f} MB  ({rss - rss0:+,} B)"
                    print(
                        f"\r  {left // 60:>3}m{left % 60:02d}s left   "
                        f"errors {bad}{mem}      ",
                        end="",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\n  interrupted")
        finally:
            for s in streams:
                s.stop()
        print()
        h = pair.health()
        for d in ("outgoing", "incoming"):
            report(f"{d} capture", h[d]["source"])
            report(f"{d} render", h[d]["sink"])
        if proc:
            rss = proc.memory_info().rss
            print(f"\n  memory growth: {(rss - rss0) / 1e6:+.1f} MB")
        return verdict(
            h["outgoing"]["source"],
            h["outgoing"]["sink"],
            h["incoming"]["source"],
            h["incoming"]["sink"],
        )


# ----------------------------------------------------------------------


def _run_until_interrupt(seconds: int | None) -> None:
    deadline = time.monotonic() + (seconds or 1e9)
    print("  Running. Ctrl-C to stop.")
    try:
        while time.monotonic() < deadline:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("  interrupted")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Phase 0 audio spine checks (Windows only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, *, mic=False, out=False, vmic=False, teams=False, secs=False):
        if mic:
            p.add_argument("--mic", help="substring of the microphone name")
        if out:
            p.add_argument("--out", help="substring of the headphone name")
        if vmic:
            p.add_argument(
                "--virtual-mic",
                required=True,
                help="substring of the virtual cable's RENDER half - what we write into",
            )
        if teams:
            p.add_argument(
                "--teams-speaker",
                required=True,
                help="substring of the render device Teams is set to output to",
            )
        if secs:
            p.add_argument("--seconds", type=int, default=None, help="auto-stop after N s")

    p = sub.add_parser("devices", help="1. list endpoints")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("mic", help="2. microphone level meter")
    common(p, mic=True)
    p.set_defaults(func=cmd_mic)

    p = sub.add_parser("passthrough", help="3. mic -> headphones")
    common(p, mic=True, out=True, secs=True)
    p.set_defaults(func=cmd_passthrough)

    p = sub.add_parser("loopback", help="4. capture what a device plays")
    p.add_argument("--target", help="substring of the render device to listen to")
    p.set_defaults(func=cmd_loopback)

    p = sub.add_parser("tone", help="5. tone -> virtual mic")
    common(p, vmic=True, secs=True)
    p.add_argument("--freq", type=float, default=440.0)
    p.set_defaults(func=cmd_tone)

    p = sub.add_parser("wire", help="6. full two-way passthrough")
    common(p, mic=True, out=True, vmic=True, teams=True, secs=True)
    p.set_defaults(func=cmd_wire)

    p = sub.add_parser("soak", help="7. stability run")
    common(p, mic=True, out=True, vmic=True, teams=True)
    p.add_argument("--minutes", type=int, default=30)
    p.set_defaults(func=cmd_soak)

    args = ap.parse_args()
    try:
        return args.func(args)
    except AudioUnavailable as exc:
        print(f"\n  AUDIO ERROR: {exc}\n")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
