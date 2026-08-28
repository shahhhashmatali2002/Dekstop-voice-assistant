#!/usr/bin/env python
"""Watch a recorded video with Urdu in sync.

    python pretranslate.py "https://www.youtube.com/watch?v=..."

Live translation is always behind the picture, because a sentence has to be
spoken before it can be translated. A RECORDING does not have that problem:
its words already exist. So this reads the whole transcript, translates and
synthesises everything up front, and then speaks each line at the timestamp it
belongs to. Nothing is being waited for during playback, so nothing is behind.

    1. it prepares the first lines   - seconds, not the whole video
    2. you press play                - it notices, and starts
    3. it speaks                     - each sentence on the video's clock
    4. you pause                     - it notices, and holds

Preparation continues while you watch, always a few lines ahead. It follows
your play and pause by watching whether the video's audio is still arriving on
the cable, so there is nothing to press here and nothing to keep in step.

The original English is silenced the same way as in listen mode: point the
browser at the cable (Settings > System > Sound > Volume mixer) so the only
thing your speakers carry is the Urdu. That routing is also what lets it
follow the video.

    python pretranslate.py URL --start 1:30    # you are resuming mid-video
    python pretranslate.py URL --no-follow     # start on Enter, ignore the cable
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


def read_transcript(source: str, languages, *, transcribe_audio: bool):
    """Timed sentences for `source`, from captions or from the audio itself."""
    import tempfile

    from app.controllers.pretranslate import (
        PrepareError,
        download_audio,
        fetch_cues,
        merge_sentences,
        transcribe,
        video_id,
    )
    from app.models.config import get_secret

    local = Path(source)
    if local.exists():
        print(f"\n  Transcribing {local.name}")
        return merge_sentences(transcribe(local, get_secret("DEEPGRAM_API_KEY")))

    video = video_id(source)

    if not transcribe_audio:
        print(f"\n  Reading captions for {video}")
        try:
            return merge_sentences(fetch_cues(video, languages))
        except PrepareError as exc:
            print(f"  {exc}\n")
            raise

    with tempfile.TemporaryDirectory(prefix="pretranslate-") as scratch:
        print(f"\n  Downloading the audio of {video}")

        def show(status: dict) -> None:
            if status.get("status") != "downloading":
                return
            done = status.get("downloaded_bytes") or 0
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if total:
                bar = "#" * int(done / total * 28)
                sys.stdout.write(f"\r  [{bar:<28}] {done / 1_000_000:.1f} MB   ")
            else:
                sys.stdout.write(f"\r  {done / 1_000_000:.1f} MB   ")
            sys.stdout.flush()

        audio = download_audio(source, Path(scratch), on_progress=show)
        sys.stdout.write("\r" + " " * 50 + "\r")
        size = audio.stat().st_size / 1_000_000
        print(f"  {size:.1f} MB, transcribing with Deepgram")
        return merge_sentences(transcribe(audio, get_secret("DEEPGRAM_API_KEY")))


def open_audio(cfg: Config, *, follow: bool):
    """The output sink, and the source we watch to follow the video."""
    from app.audio.devices import AudioUnavailable, DeviceManager
    from app.audio.streams import CaptureSource, RenderSink

    dm = DeviceManager()
    device = (
        dm.find(cfg.listen.output, capture=False)
        if cfg.listen.output
        else dm.default_output()
    )
    sink = RenderSink(dm, device)
    sink.start()

    source = None
    if follow and cfg.listen.source:
        try:
            loopback = dm.loopback_for_output(
                dm.find(cfg.listen.source, capture=False)
            )
            source = CaptureSource(dm, loopback)
            source.start()
        except AudioUnavailable as exc:
            print(f"  note: cannot follow the video ({exc}).")
            print("        Falling back to a clock you start with Enter.")
    return dm, sink, source, device


async def play(cfg: Config, cues, sink, source, *, start_at: float, offset: float):
    """Prepare and play at the same time, following the video if we can."""
    from app.audio.formats import RATE_S2ST
    from app.controllers.pretranslate import (
        PausableClock, Player, Script, SourceMonitor, prepare_into,
    )
    from app.providers.base import VoiceId
    from app.providers.factory import build_providers

    providers = build_providers(cfg, include_s2st=False)
    voice = VoiceId(
        vendor=cfg.providers.tts, name=cfg.providers.urdu_voice, rate=RATE_S2ST
    )

    script = Script()
    total = len(cues)
    prepared = time.monotonic()

    def progress(index: int, _total: int, _text: str) -> None:
        if index % 5 == 0 or index == total:
            sys.stdout.write(f"\r  prepared {index}/{total}   ")
            sys.stdout.flush()

    preparing = asyncio.ensure_future(prepare_into(
        script, cues, translator=providers["translator"], tts=providers["tts"],
        voice=voice, on_progress=progress,
    ))

    # A couple of lines in hand before the video starts, so the first sentence
    # is never the one we are still synthesising.
    while script.ready < 2 and not script.finished:
        await asyncio.sleep(0.1)
    if script.error:
        raise script.error
    print(f"\r  ready in {time.monotonic() - prepared:.0f}s"
          f" - {script.ready} lines prepared, the rest continues while you watch")

    clock = PausableClock(start_at)
    monitor = None

    if source is not None:
        def announce(playing: bool) -> None:
            print("  ▶ playing" if playing else "  ⏸ paused - holding the Urdu")

        monitor = SourceMonitor(source, clock, on_change=announce)
        asyncio.ensure_future(monitor.run())
        print("\n  Press play on the video. It follows your play and pause.\n")
    else:
        print(f"\n  Start the video at {format_position(start_at)}, then press "
              "Enter here.")
        await asyncio.get_event_loop().run_in_executor(None, input, "  > ")
        clock.start()

    player = Player(sink, [], offset=offset)

    def announce_line(line) -> None:
        print(f"  {format_position(line.start)}  {line.english}")
        print(f"           {line.urdu}")

    try:
        await player.follow(clock, script, on_line=announce_line)
    finally:
        if monitor is not None:
            monitor.stop()
        preparing.cancel()
        with contextlib.suppress(Exception):
            await preparing
    return player


def main() -> int:
    _utf8_console()
    ap = argparse.ArgumentParser(
        description="Watch a recorded video with Urdu in sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("url", metavar="VIDEO",
                    help="YouTube URL, a video id, or the path to a local "
                         "audio or video file")
    ap.add_argument("--out", metavar="NAME",
                    help="where the Urdu is played; default is the Windows default")
    ap.add_argument("--start", type=parse_position, default=0.0, metavar="TIME",
                    help="where the video is when you press play, e.g. 1:30")
    ap.add_argument("--offset", type=float, default=0.0, metavar="SECONDS",
                    help="shift every line; negative speaks earlier")
    ap.add_argument("--lang", default="en", metavar="CODE",
                    help="caption language to read (default en)")
    ap.add_argument("--no-follow", action="store_true",
                    help="do not watch the cable to follow the video's play "
                         "and pause; start the clock on Enter instead")
    ap.add_argument("--transcribe", action="store_true",
                    help="skip YouTube's captions: download the audio with "
                         "yt-dlp and transcribe it with Deepgram. Slower and "
                         "costs credit, but works on videos with captions "
                         "disabled and on ones YouTube will not serve to a "
                         "plain HTTP client.")
    args = ap.parse_args()

    cfg = Config.load()
    if args.out:
        cfg.listen.output = args.out

    missing = missing_secrets(cfg)
    if missing:
        print(f"\n  API keys still needed: {', '.join(missing)}\n")
        return 2

    from app.controllers.pretranslate import PrepareError

    try:
        cues = read_transcript(
            args.url, [args.lang], transcribe_audio=args.transcribe
        )
    except PrepareError as exc:
        print(f"\n  {exc}\n")
        return 2
    span = cues[-1].end if cues else 0.0
    print(f"  {len(cues)} sentences over {format_position(span)}")

    from app.audio.devices import AudioUnavailable

    try:
        dm, sink, source, device = open_audio(cfg, follow=not args.no_follow)
    except AudioUnavailable as exc:
        print(f"\n  {exc}\n")
        return 2

    print(f"\n  Playing into {device.name}")
    try:
        player = asyncio.run(play(
            cfg, cues, sink, source,
            start_at=args.start, offset=args.offset,
        ))
        print(f"\n  Done. {player.spoken} lines spoken"
              + (f", {player.skipped} skipped as already past." if player.skipped
                 else "."))
    except PrepareError as exc:
        print(f"\n  {exc}\n")
        return 2
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        with contextlib.suppress(Exception):
            sink.stop()
            if source is not None:
                source.stop()
            dm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
