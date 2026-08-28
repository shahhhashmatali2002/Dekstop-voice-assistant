"""Say one line out loud, or write it to a WAV - for judging the voice by ear.

    python tools/say.py "ہم نے API کا test کیا"
    python tools/say.py --english "We tested the API in production."
    python tools/say.py --voice ur-PK-AsadNeural --speed 1.1 "کیا حال ہے"
    python tools/say.py --show "YouTube پر 50% servers"     # no audio, just text

The point of this existing separately from the application is that the voice is
the one thing that has to be judged by listening, and starting the whole
pipeline to hear one sentence means routing a browser, waiting for
recognition, and hoping the line you wanted to test is the line that comes out.

`--show` prints what the voice will actually be given, after the pronunciation
pass in app/providers/urdu_speech - which is the fastest way to find a word
missing from the dictionary.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.providers.base import VoiceId                       # noqa: E402
from app.providers.tts import EdgeTextToSpeech               # noqa: E402
from app.providers.urdu_speech import for_speech             # noqa: E402


async def synthesise(text: str, voice: str, speed: float, rate: int):
    tts = EdgeTextToSpeech(rate=rate)
    tts.set_speed(speed)
    chunks: list[np.ndarray] = []
    async for chunk in tts.synthesize(text, VoiceId("edge", voice)):
        if chunk.pcm.size:
            chunks.append(chunk.pcm)
    print(f"  rate sent to the voice: {tts.request_rate}")
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int16)


async def translate(text: str) -> str:
    from app.providers.base import TranslationUnit, RollingContext
    from app.providers.llm_translator import LlmTranslator

    translator = LlmTranslator()
    unit = TranslationUnit(text=text, source_language="en", target_language="ur")
    out = await translator.translate(unit, RollingContext(), None)
    return out.text


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Say one line with the Urdu voice.")
    ap.add_argument("text", help="Urdu to speak, or English with --english")
    ap.add_argument("--english", action="store_true",
                    help="translate the text first (needs OPENAI_API_KEY)")
    ap.add_argument("--voice", default="ur-PK-UzmaNeural")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1.0 to 1.2; what the drift controller asks for when "
                         "playback is behind")
    ap.add_argument("--rate", type=int, default=24_000)
    ap.add_argument("--out", default="say.wav")
    ap.add_argument("--fragment", action="store_true",
                    help="close it as a mid-sentence clause, with a comma "
                         "rather than a full stop")
    ap.add_argument("--show", action="store_true",
                    help="print the prepared line and exit; no synthesis")
    args = ap.parse_args()

    text = args.text
    if args.english:
        print("Translating...")
        text = asyncio.run(translate(args.text))
        print(f"  Urdu: {text}")

    prepared = for_speech(text, final=not args.fragment)
    print(f"  what the voice is given: {prepared}")
    if args.show:
        return 0

    # The prepared line, not the raw one, so what is heard is what was
    # printed - including the fragment ending.
    pcm = asyncio.run(synthesise(prepared, args.voice, args.speed, args.rate))
    if pcm.size == 0:
        print("No audio came back.", file=sys.stderr)
        return 1

    write_wav(args.out, pcm, args.rate)
    seconds = pcm.size / args.rate
    print(f"  wrote {args.out} - {seconds:.1f}s. Play it and judge it by ear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
