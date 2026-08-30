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


def build_clone(source: str, *, seconds: float = 0.0, backend: str = "knn"):
    """A converter, measured on this machine, and the voice to imitate.

    The reference is either a recording the user pointed at or a clip of the
    video's own audio - which is the interesting case, because it means the
    voice you hear the Urdu in is the voice that was actually speaking.

    The measurement is printed rather than hidden. On a CPU this is several
    times slower than real time, and a user who is about to wait twenty minutes
    for a forty-minute video deserves to be told before the wait, not during
    it.
    """
    from pathlib import Path
    import tempfile

    from app.controllers.pretranslate import PrepareError, download_audio
    from app.providers.voice_clone import build_converter, pick_reference

    converter = build_converter(backend)
    # The backend decides how much reference it wants: kNN-VC matches frames
    # against the target's own audio, so minutes help it, while Seed-VC needs
    # only a few seconds. For a video neither costs anything - the audio is
    # already downloaded.
    want = seconds or converter.preferred_reference_s

    from app.audio.decode import DecodeUnavailable, read_audio_file

    def decode(path: Path):
        # A downloaded YouTube audio track is usually Opus in WebM, which
        # miniaudio cannot read - read_audio_file falls back to ffmpeg.
        try:
            return read_audio_file(path)
        except DecodeUnavailable as exc:
            raise PrepareError(str(exc)) from exc

    local = Path(source)
    if local.exists():
        pcm, rate = decode(local)
    else:
        print(f"\n  Taking the reference voice from {source}")
        with tempfile.TemporaryDirectory(prefix="clone-ref-") as scratch:
            pcm, rate = decode(download_audio(source, Path(scratch)))

    reference = pick_reference(pcm, rate, seconds=max(1.0, min(300.0, want)))
    if reference.pcm.size == 0:
        raise PrepareError("Could not find any audio to use as a reference.")

    print(f"  reference clip: {reference.seconds:.0f}s   "
          f"backend: {converter.name}   device: {converter.device}")
    factor = converter.measure(reference, seconds=2.0)
    print(f"  voice conversion runs at {factor:.2f}x compute per second of audio")
    if factor >= 1.0:
        print("  slower than real time, so everything is prepared before the "
              "video starts.")
    return converter, reference


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


async def play(cfg: Config, cues, sink, source, *, start_at: float, offset: float,
               converter=None, reference=None):
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
        converter=converter, reference=reference,
    ))

    # How many lines to have in hand before the video starts. Normally two -
    # enough that the first sentence is never the one still being synthesised.
    # With voice conversion slower than real time, preparation can never catch
    # up with playback, so the whole thing is prepared first: a wait before the
    # video, instead of the Urdu falling further behind for forty minutes.
    factor = getattr(converter, "realtime_factor", 0.0) or 0.0
    head_start = len(cues) if factor >= 1.0 else 2
    if head_start > 2:
        print(f"  preparing all {len(cues)} lines before starting"
              f" - about {len(cues) * factor / 60:.0f} min of conversion")

    while script.ready < head_start and not script.finished:
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
    ap.add_argument("--clone-voice", action="store_true",
                    help="speak the Urdu in the voice of the video's own "
                         "speaker, by converting the synthesised audio onto a "
                         "reference clip taken from the video. Needs torch "
                         "installed, and is several times slower than real "
                         "time on a CPU - run tools/clone_bench.py first.")
    ap.add_argument("--clone-ref", metavar="FILE",
                    help="use this recording as the voice to imitate instead "
                         "of a clip of the video")
    ap.add_argument("--clone-seconds", type=float, default=0.0, metavar="N",
                    help="how much reference audio to use; the default is what "
                         "the backend wants - about two and a half minutes for "
                         "knn-vc, ten seconds for seed-vc")
    ap.add_argument("--clone-backend", default="knn",
                    choices=("knn", "light", "seed"),
                    help="knn (default) is the real thing and needs torch; "
                         "light is a pitch-and-tone resemblance that runs on "
                         "any CPU in real time; seed pins numpy below 2.0 and "
                         "will not install on Python 3.13 without a compiler")
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

    converter = reference = None
    if args.clone_voice:
        from app.providers.base import ProviderError
        try:
            converter, reference = build_clone(
                args.clone_ref or args.url,
                seconds=args.clone_seconds,
                backend=args.clone_backend,
            )
        except (PrepareError, ProviderError) as exc:
            print(f"\n  {exc}\n")
            return 2

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
            converter=converter, reference=reference,
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
