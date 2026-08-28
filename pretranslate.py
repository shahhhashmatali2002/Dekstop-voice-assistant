#!/usr/bin/env python
"""Watch a recorded video with Urdu in sync.

    python pretranslate.py "https://www.youtube.com/watch?v=..."

Live translation is always behind the picture, because a sentence has to be
spoken before it can be translated. A RECORDING does not have that problem:
its words already exist. So this reads the whole transcript, translates and
synthesises everything up front, and then speaks each line at the timestamp it
belongs to. Nothing is being waited for during playback, so nothing is behind.

    1. it prepares       - a wait, once, proportional to the video's length
    2. you press play    - and press Enter here at the same moment
    3. it speaks         - each sentence on the video's own clock

The original English is silenced the same way as in listen mode: point the
browser at the cable (Settings > System > Sound > Volume mixer) so the only
thing your speakers carry is the Urdu.

    python pretranslate.py URL --start 1:30    # you are resuming mid-video
    python pretranslate.py URL --offset -0.4   # it is consistently late
    python pretranslate.py URL --out "Headset" # where the Urdu is played

This cannot work on a Teams call or a live stream: there is no future to read.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.models.config import Config, missing_secrets  # noqa: E402


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_position(text: str) -> float:
    """'90', '1:30' and '01:02:03' all mean a number of seconds."""
    parts = (text or "0").strip().split(":")
    try:
        numbers = [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not a time") from exc
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def format_position(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


async def build_script(cfg: Config, url: str, languages):
    from app.audio.formats import RATE_S2ST
    from app.controllers.pretranslate import (
        fetch_cues,
        merge_sentences,
        overruns,
        prepare,
        video_id,
    )
    from app.providers.base import VoiceId
    from app.providers.factory import build_providers

    print(f"\n  Reading captions for {url}")
    cues = merge_sentences(fetch_cues(video_id(url), languages))
    span = cues[-1].end if cues else 0.0
    print(f"  {len(cues)} sentences over {format_position(span)}")

    providers = build_providers(cfg, include_s2st=False)
    voice = VoiceId(
        vendor=cfg.providers.tts, name=cfg.providers.urdu_voice, rate=RATE_S2ST
    )

    started = time.monotonic()

    def progress(index: int, total: int, text: str) -> None:
        done = index / total
        bar = "#" * int(done * 28)
        eta = (time.monotonic() - started) / max(1, index - 1) * (total - index)
        sys.stdout.write(
            f"\r  [{bar:<28}] {index}/{total}  ~{int(eta)}s left   "
        )
        sys.stdout.flush()

    lines = await prepare(
        cues,
        translator=providers["translator"],
        tts=providers["tts"],
        voice=voice,
        on_progress=progress,
    )
    print(f"\r  [{'#' * 28}] {len(lines)} lines ready" + " " * 20)

    late = overruns(lines)
    if late:
        worst = max(extra for _line, extra in late)
        print(
            f"  note: {len(late)} lines are longer in Urdu than the gap the "
            f"English left (worst {worst:.1f}s).\n"
            "        They will overlap slightly rather than be cut short."
        )
    return lines


def main() -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(
        description="Watch a recorded video with Urdu in sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("url", help="YouTube URL, or just the video id")
    ap.add_argument("--out", metavar="NAME",
                    help="where the Urdu is played; default is the Windows default")
    ap.add_argument("--start", type=parse_position, default=0.0, metavar="TIME",
                    help="where the video is when you press play, e.g. 1:30")
    ap.add_argument("--offset", type=float, default=0.0, metavar="SECONDS",
                    help="shift every line; negative speaks earlier")
    ap.add_argument("--lang", default="en", metavar="CODE",
                    help="caption language to read (default en)")
    args = ap.parse_args()

    cfg = Config.load()
    if args.out:
        cfg.listen.output = args.out

    missing = missing_secrets(cfg)
    if missing:
        print(f"\n  API keys still needed: {', '.join(missing)}\n")
        return 2

    from app.controllers.pretranslate import PrepareError, Player

    try:
        lines = asyncio.run(build_script(cfg, args.url, [args.lang]))
    except PrepareError as exc:
        print(f"\n  {exc}\n")
        return 2

    # Audio comes up only now: preparing takes minutes and holding an output
    # device open through it would block anything else that wants one.
    from app.audio.devices import AudioUnavailable, DeviceManager
    from app.audio.streams import RenderSink

    try:
        dm = DeviceManager()
        device = (
            dm.find(cfg.listen.output, capture=False)
            if cfg.listen.output
            else dm.default_output()
        )
        sink = RenderSink(dm, device)
        sink.start()
    except AudioUnavailable as exc:
        print(f"\n  {exc}\n")
        return 2

    print(f"\n  Playing into {device.name}")
    print(f"  Start the video at {format_position(args.start)}, then press "
          "Enter here at the same moment.")
    with contextlib.suppress(EOFError, KeyboardInterrupt):
        input("  > ")

    player = Player(sink, lines, offset=args.offset)

    def announce(line) -> None:
        print(f"  {format_position(line.start)}  {line.english}")
        print(f"           {line.urdu}")

    try:
        asyncio.run(player.run(position=args.start, on_line=announce))
        print(f"\n  Done. {player.spoken} lines spoken"
              + (f", {player.skipped} skipped as already past." if player.skipped
                 else "."))
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        with contextlib.suppress(Exception):
            sink.stop()
            dm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
