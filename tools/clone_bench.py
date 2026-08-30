"""Is voice conversion fast enough on THIS machine? Answer with a number.

    python tools/clone_bench.py --from-audio clip.mp3
    python tools/clone_bench.py --from-video "https://youtu.be/..."
    python tools/clone_bench.py --from-audio clip.mp3 --line "کیا حال ہے"

It takes a reference clip of the voice to imitate, synthesises one Urdu line
with the normal voice, converts that line onto the reference, and writes both
so they can be compared by ear. Then it prints the only number that decides
whether this feature is usable here: seconds of compute per second of audio.

    below 1.0   fast enough to convert while a video plays
    1.0 to 4.0  a recording can be prepared before playback; live is out
    above 4.0   preparing a 40-minute video takes longer than watching it

The reason this exists as a separate tool is that the honest answer depends
entirely on the hardware, and the difference between a GPU and a laptop CPU
here is more than an order of magnitude. Guessing it wrong means either a
feature that stutters or a feature nobody dares turn on.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.providers.base import VoiceId, ProviderError          # noqa: E402
from app.providers.tts import EdgeTextToSpeech                 # noqa: E402
from app.providers.voice_clone import (                        # noqa: E402
    BACKENDS,
    Reference,
    build_converter,
    pick_reference,
)

SAMPLE_LINE = (
    "خوش آمدید۔ آج ہم انگریزی سننے کی مشق کریں گے اور قدرتی گفتگو سنیں گے۔"
)


def read_audio(path: Path, rate: int = 24_000) -> tuple[np.ndarray, int]:
    """Any audio file, as mono int16 at `rate`. ffmpeg where miniaudio cannot."""
    from app.audio.decode import DecodeUnavailable, read_audio_file

    try:
        return read_audio_file(path, rate)
    except DecodeUnavailable as exc:
        raise SystemExit(f"\n  {exc}\n") from exc


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())


async def synthesise(line: str, voice: str, rate: int) -> np.ndarray:
    tts = EdgeTextToSpeech(rate=rate)
    chunks = [c.pcm async for c in tts.synthesize(line, VoiceId("edge", voice))
              if c.pcm.size]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


def reference_from_video(url: str, rate: int) -> tuple[np.ndarray, int]:
    from app.controllers.pretranslate import download_audio

    with tempfile.TemporaryDirectory() as tmp:
        print("  downloading the video's audio...")
        audio = download_audio(url, Path(tmp))
        return read_audio(audio, rate)


def verdict(factor: float) -> str:
    if factor < 1.0:
        return ("fast enough to convert while the video plays, so it can go on "
                "the pre-translate path with no waiting, and is worth trying "
                "live too.")
    if factor < 4.0:
        return ("too slow for live, fine for a recording: a video has to be "
                "prepared before playback starts.")
    return ("slower than watching the video. Usable only for something short, "
            "or on a machine with a GPU.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure voice conversion here.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-audio", metavar="FILE",
                     help="a recording of the voice to imitate")
    src.add_argument("--from-video", metavar="URL",
                     help="take the reference clip from a video's own audio")
    ap.add_argument("--line", default=SAMPLE_LINE, help="the Urdu to speak")
    ap.add_argument("--voice", default="ur-PK-UzmaNeural")
    ap.add_argument("--backend", default="knn", choices=sorted(BACKENDS),
                    help="knn (default; needs only torch and torchaudio) or "
                         "seed (will not install on Python 3.13)")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="how much reference audio to use; default is what the "
                         "backend actually wants - minutes for knn, seconds "
                         "for seed")
    ap.add_argument("--rate", type=int, default=24_000)
    ap.add_argument("--device", default="", help="cpu or cuda; default is auto")
    args = ap.parse_args()

    converter = build_converter(args.backend, device=args.device)
    want = args.seconds or converter.preferred_reference_s

    if args.from_video:
        pcm, rate = reference_from_video(args.from_video, args.rate)
    else:
        pcm, rate = read_audio(Path(args.from_audio), args.rate)
    if pcm.size == 0:
        print("The reference audio is empty.", file=sys.stderr)
        return 1

    reference = pick_reference(pcm, rate, seconds=want)
    write_wav("clone_reference.wav", reference.pcm, reference.rate)
    print(f"  reference: {reference.seconds:.1f}s  -> clone_reference.wav")

    print("  synthesising the Urdu line with the normal voice...")
    try:
        urdu = asyncio.run(synthesise(args.line, args.voice, args.rate))
    except ProviderError as exc:
        print(f"Synthesis failed: {exc}", file=sys.stderr)
        return 1
    if urdu.size == 0:
        print("No audio came back from the voice.", file=sys.stderr)
        return 1
    write_wav("clone_before.wav", urdu, args.rate)
    print(f"  before: {urdu.size / args.rate:.1f}s -> clone_before.wav")

    print(f"  backend: {converter.name}   device: {converter.device}")
    print("  converting (first run downloads the model, which takes a while)...")
    try:
        # Timed on the real synthesised line, not on a generated tone: the
        # converter trims non-speech before matching, so a tone is trimmed to
        # nothing and the measurement crashes instead of returning a number.
        factor = converter.measure(urdu, args.rate, reference)
        converted, out_rate = converter.convert(urdu, args.rate, reference)
    except ProviderError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    write_wav("clone_after.wav", converted, out_rate)
    minutes = 40 * factor
    print(f"  after: -> clone_after.wav")
    print()
    print(f"  {factor:.2f} seconds of compute per second of audio")
    print(f"  a 40-minute video would take about {minutes:.0f} minutes to prepare")
    print(f"  {verdict(factor)}")
    print()
    print("  Listen to clone_before.wav and clone_after.wav. The question is not")
    print("  whether it is identical - it will not be - but whether it sounds")
    print("  like a person rather than an artefact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
