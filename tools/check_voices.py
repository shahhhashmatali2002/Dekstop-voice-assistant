"""Hear the voices before a client does, and pin the ones you like.

    python tools/check_voices.py                 # what the account has + samples
    python tools/check_voices.py --list          # just the table, no synthesis
    python tools/check_voices.py --urdu-voice <voice_id> --english-voice <voice_id>

WHY THIS EXISTS
---------------
Two things decide whether the output sounds natural, and neither is visible
from inside the application:

  * THE MODEL. Urdu is not in Flash v2.5's 32 languages. Urdu sent to Flash
    does not fail - it comes back as a voice sounding out letters it cannot
    read, which is heard as a terrible accent rather than as an error. Urdu
    needs the v3 family. The app now picks the model from the language
    (ELEVEN_MODELS in app/providers/tts); this prints what your account can
    actually reach, which is the part that changes without warning.

  * THE VOICE. Every ElevenLabs voice can pronounce Urdu under v3. They do not
    all sound like an Urdu speaker doing it. The app guesses by looking for a
    South Asian voice in your library; a guess is worse than a choice, so this
    writes samples to WAV files and you pick by ear.

Once you have picked, put the IDs in .env and nothing has to guess again:

    ELEVEN_UR_VOICE_ID=...
    ELEVEN_EN_VOICE_ID=...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.error
import urllib.request
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.config import get_secret            # noqa: E402
from app.providers.base import VoiceId              # noqa: E402
from app.providers.tts import (                     # noqa: E402
    ELEVEN_MODELS,
    SOUTH_ASIAN_HINTS,
    ElevenLabsTextToSpeech,
)

RATE = 24_000

URDU_SAMPLE = (
    "السلام علیکم، میں جان ہوں۔ میں میکروسافٹ پاکستان میں کام کرتا ہوں، "
    "اور ہم اگلے ہفتے یہ فیچر پروڈکشن میں بھیج رہے ہیں۔"
)
ENGLISH_SAMPLE = (
    "Hello, I am John Wick. I work at Macrosoft Pakistan, and we are shipping "
    "this feature to production next week."
)


def api(path: str, key: str):
    request = urllib.request.Request(
        f"https://api.elevenlabs.io{path}", headers={"xi-api-key": key}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"  {path} -> HTTP {exc.code}: {exc.read()[:200].decode('utf-8','replace')}")
    except Exception as exc:
        print(f"  {path} -> {exc}")
    return None


def show_models(key: str) -> None:
    print("MODELS ON THIS ACCOUNT")
    data = api("/v1/models", key)
    if not data:
        return
    for model in data:
        languages = {
            (entry.get("language_id") or "").lower()
            for entry in (model.get("languages") or [])
        }
        marks = []
        if "ur" in languages:
            marks.append("URDU")
        if "en" in languages:
            marks.append("english")
        if model.get("can_do_text_to_speech"):
            marks.append("tts")
        print(f"  {model.get('model_id',''):<32} {len(languages):>3} langs  "
              f"{', '.join(marks)}")
    print(f"\n  the app will use: urdu={ELEVEN_MODELS['ur']}  "
          f"english={ELEVEN_MODELS['en']}")


def show_voices(key: str) -> list[dict]:
    print("\nVOICES ON THIS ACCOUNT")
    data = api("/v2/voices?page_size=100", key)
    voices = (data or {}).get("voices") or []
    if not voices:
        print("  none found")
        return []
    for entry in voices:
        labels = " ".join(str(v) for v in (entry.get("labels") or {}).values())
        haystack = f"{entry.get('name','')} {entry.get('description','')} {labels}".lower()
        flag = "  <-- South Asian" if any(h in haystack for h in SOUTH_ASIAN_HINTS) else ""
        print(f"  {entry.get('voice_id',''):<24} {entry.get('name',''):<22} "
              f"{labels[:40]}{flag}")
    return voices


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())


async def sample(key: str, text: str, language: str, voice_id: str | None) -> str:
    tts = ElevenLabsTextToSpeech(key, rate=RATE)
    voice = VoiceId(
        vendor="elevenlabs",
        name=voice_id or "",
        rate=RATE,
        language=language,
    )
    if not voice_id:
        try:
            resolved = await tts.voice_for(voice)
        except Exception as exc:
            return f"  {language}: no voice could be resolved ({exc})"
        voice = VoiceId("elevenlabs", resolved, RATE, None, language)
    chunks: list[np.ndarray] = []
    try:
        async for chunk in tts.synthesize(text, voice):
            if chunk.pcm.size:
                chunks.append(chunk.pcm)
    except Exception as exc:
        return (f"  {language}: FAILED with model "
                f"{tts.model_for(language)} and voice {voice.name}\n     {exc}")
    if not chunks:
        return f"  {language}: nothing came back"
    path = f"sample_{language}.wav"
    write_wav(path, np.concatenate(chunks), RATE)
    seconds = sum(c.size for c in chunks) / RATE
    return (f"  {language}: {path}  ({seconds:.1f}s, model "
            f"{tts.model_for(language)}, voice {voice.name})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect and audition the voices.")
    ap.add_argument("--list", action="store_true", help="no synthesis, just the table")
    ap.add_argument("--urdu-voice", help="voice ID to audition for Urdu")
    ap.add_argument("--english-voice", help="voice ID to audition for English")
    args = ap.parse_args()

    key = get_secret("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVENLABS_API_KEY is not set.", file=sys.stderr)
        return 2

    show_models(key)
    show_voices(key)
    if args.list:
        return 0

    print("\nSAMPLES - listen to these, then pin what you like in .env")
    print(asyncio.run(sample(key, URDU_SAMPLE, "ur", args.urdu_voice)))
    print(asyncio.run(sample(key, ENGLISH_SAMPLE, "en", args.english_voice)))
    print(
        "\n  ELEVEN_UR_VOICE_ID=<id>   in .env pins the Urdu voice\n"
        "  ELEVEN_EN_VOICE_ID=<id>   pins the English one\n"
        "  ELEVEN_URDU_MODEL=<id>    if you want a different model for Urdu"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
