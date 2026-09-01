"""Which language should the recogniser be listening in? Ask it.

    python tools/bench_language.py --record 8
    python tools/bench_language.py deepgram_tap.wav
    python tools/bench_language.py --record 8 --keyterm Macrosoft --keyterm "John Wick"

WHY THIS EXISTS
---------------
"It heard my name wrong" is not a bug report a person can act on, because
there are four different faults that produce it and they need opposite fixes:

    the recogniser was pinned to the wrong language
    the recogniser was right but the name is not in its vocabulary
    the name was heard correctly and the TRANSLATOR tidied it
    the audio going in is too quiet or clipped to hear anything from

This sends ONE recording to Deepgram several times - once per language, once
with keyterm prompting and once without - and prints every answer side by
side with the model's own confidence, plus what app/providers/dual_asr would
have picked. Ten seconds of your own voice settles the question that no amount
of reading the code will.

Say the sentence that goes wrong. Include the name.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.config import get_secret                 # noqa: E402
from app.models.terminology import Terminology           # noqa: E402
from app.providers.dual_asr import choose, score         # noqa: E402

ENDPOINT = "https://api.deepgram.com/v1/listen"
RATE = 16_000


# ----------------------------------------------------------------- capture


def record(seconds: float) -> bytes:
    """Grab a few seconds from the default microphone as 16 kHz mono PCM."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        try:
            import pyaudio  # type: ignore
        except ImportError:
            print(
                "Recording needs PyAudioWPatch: pip install PyAudioWPatch\n"
                "Or pass a .wav file instead.",
                file=sys.stderr,
            )
            raise SystemExit(2)

    audio = pyaudio.PyAudio()
    try:
        info = audio.get_default_input_device_info()
        print(f"microphone   {info['name']}")
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=1024,
        )
        print(f"\n>>> SPEAK NOW - {seconds:.0f} seconds <<<\n")
        frames = []
        for _ in range(int(RATE / 1024 * seconds)):
            frames.append(stream.read(1024, exception_on_overflow=False))
        stream.stop_stream()
        stream.close()
        print("...done.\n")
        return b"".join(frames)
    finally:
        audio.terminate()


def load_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as handle:
        if handle.getnchannels() != 1:
            print(f"{path}: expected mono, got {handle.getnchannels()} channels")
        return handle.readframes(handle.getnframes()), handle.getframerate()


def as_wav(pcm: bytes, rate: int) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def analyse(pcm: bytes, rate: int) -> list[str]:
    """Is this recording good enough for ANY recogniser? Answer that first.

    Almost every "it heard me wrong" turns out to be one of four things, and
    three of them are visible here without sending a byte to a provider:

      * nothing was captured (wrong device)
      * it is too quiet, or clipped
      * it is spectrally wrecked - and this is the one nobody looks for

    Intelligibility lives between about 500 Hz and 4 kHz. That is where the
    second and third formants are, and formants are how a listener tells
    "Wick" from "Bek". A capture chain that keeps the low rumble and throws
    that band away still sounds like a voice - you can hear that someone is
    talking, and even who - while carrying almost none of the information a
    recogniser uses. Aggressive noise suppression, a headset in hands-free
    mode, and a bad resample all do this.
    """
    import numpy as np

    samples = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype=np.int16)
    if samples.size == 0:
        return ["SILENT - nothing was captured. Wrong input device?"]

    a = samples.astype(np.float64)
    notes = []
    peak = float(np.abs(a).max())
    win = int(rate * 0.02) or 1
    frames = a[: a.size // win * win].reshape(-1, win)
    loud = frames[np.sqrt((frames ** 2).mean(axis=1)) >= 300]

    notes.append(
        f"peak {peak / 327.68:.0f}% of full scale, "
        f"speech present in {100 * len(loud) / max(1, len(frames)):.0f}% of it"
    )
    if peak >= 32_700:
        notes.append("  !! CLIPPING - turn the input gain (or MIC_GAIN) down")
    if not len(loud):
        notes.append("  !! no speech-level audio at all - wrong device, or muted")
        return notes

    speech = loud.ravel()
    rms_db = 20 * np.log10(speech.std() / 32768 + 1e-12)
    notes.append(f"speech level {rms_db:.0f} dBFS")
    if rms_db < -30:
        notes.append("  !! very quiet - raise the mic level in Windows first")

    spectrum = np.abs(np.fft.rfft(loud * np.hanning(win), axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(win, 1 / rate)
    total = spectrum.sum() or 1.0
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    shares = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        share = 100 * spectrum[mask].sum() / total
        shares.append(share)
        notes.append(f"  {lo:>5}-{hi:<5} Hz  {share:5.1f}%  {'#' * int(share / 2)}")

    speech_band = shares[1] + shares[2] + shares[3]  # 500 Hz - 4 kHz
    if speech_band < 25:
        notes.append(
            f"  !! only {speech_band:.0f}% of the energy is in the 500 Hz - 4 kHz\n"
            "     band where intelligibility lives. This recording is spectrally\n"
            "     wrecked and NO recogniser will read it reliably. Look at the\n"
            "     capture chain, not the model: a Bluetooth headset in hands-free\n"
            "     mode, Windows or Teams noise suppression, a 'voice isolation'\n"
            "     effect, or a resample somewhere in app/audio."
        )
    elif shares[0] > 55:
        notes.append(
            f"  !  {shares[0]:.0f}% of the energy is below 500 Hz - rumble, "
            "handling noise or a proximity effect is dominating the voice."
        )
    return notes


def save(pcm: bytes, rate: int, path: str) -> None:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    print(f"saved        {path}  <- LISTEN TO THIS. If it does not sound like\n"
          f"             you in a quiet room, stop and fix that first.")


# --------------------------------------------------------------- recognise


def transcribe(key: str, wav: bytes, language: str, keyterms: list[str]) -> dict:
    params = {
        "model": "nova-3",
        "language": language,
        "punctuate": "true",
        "smart_format": "true",
    }
    query = urllib.parse.urlencode(params)
    for term in keyterms:
        query += "&" + urllib.parse.urlencode({"keyterm": term})
    request = urllib.request.Request(
        f"{ENDPOINT}?{query}",
        data=wav,
        headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            body = json.loads(resp.read())
    except Exception as exc:
        detail = getattr(exc, "read", lambda: b"")()
        return {"error": f"{exc} {detail[:200].decode('utf-8', 'replace')}"}
    alt = body["results"]["channels"][0]["alternatives"][0]
    return {
        "text": alt.get("transcript", "").strip(),
        "confidence": alt.get("confidence"),
    }


def show(label: str, result: dict) -> None:
    if "error" in result:
        print(f"  {label:<26} ERROR  {result['error']}")
        return
    conf = result.get("confidence")
    conf_text = f"{conf:.3f}" if isinstance(conf, (int, float)) else "  -  "
    print(f"  {label:<26} conf {conf_text}  {result['text'] or '(nothing)'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare what Deepgram hears in each language."
    )
    ap.add_argument("wav", nargs="?", help="a mono WAV to test; omit with --record")
    ap.add_argument("--record", type=float, metavar="SECONDS",
                    help="record from the default microphone instead")
    ap.add_argument("--languages", default="en,ur,multi",
                    help="comma separated (default en,ur,multi)")
    ap.add_argument("--keyterm", action="append", default=[],
                    help="a name to bias towards; repeat for several")
    ap.add_argument("--no-terminology", action="store_true",
                    help="do not add the app's own keyterm list")
    args = ap.parse_args()

    key = get_secret("DEEPGRAM_API_KEY")
    if not key:
        print("DEEPGRAM_API_KEY is not set.", file=sys.stderr)
        return 2

    if args.record:
        pcm, rate = record(args.record), RATE
    elif args.wav:
        pcm, rate = load_wav(args.wav)
    else:
        ap.error("give a .wav or use --record SECONDS")
        return 2

    print(f"audio        {len(pcm) / 2 / rate:.1f}s at {rate} Hz")
    for line in analyse(pcm, rate):
        print(f"             {line}" if not line.startswith(" ") else line)
    save(pcm, rate, "bench_input.wav")

    keyterms = list(args.keyterm)
    if not args.no_terminology:
        keyterms += [t for t in Terminology().keyterms() if t not in keyterms]
    print(f"keyterms     {len(keyterms)}: {', '.join(keyterms[:12])}"
          f"{' ...' if len(keyterms) > 12 else ''}")

    wav = as_wav(pcm, rate)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]

    print("\nWITHOUT keyterm prompting")
    plain = {}
    for language in languages:
        result = transcribe(key, wav, language, [])
        plain[language] = result
        show(f"language={language}", result)

    print("\nWITH keyterm prompting")
    hinted = {}
    for language in languages:
        result = transcribe(key, wav, language, keyterms)
        hinted[language] = result
        show(f"language={language}", result)

    print("\nWHAT THE APP WOULD DO (app/providers/dual_asr)")
    candidates = {
        language: (r.get("text", ""), r.get("confidence"))
        for language, r in hinted.items()
        if language in ("en", "ur") and "error" not in r
    }
    for language, (text, confidence) in candidates.items():
        print(f"  {language:<4} score {score(language, text, confidence):.3f}")
    language, text, _ = choose(candidates)
    print(f"\n  -> picks {language or '(nothing)'}: {text or '(nothing)'}")

    print(
        "\nREADING THIS\n"
        "  If the English line is right and the Urdu line is wrong, the old\n"
        "  behaviour (pinned to Urdu) was the bug and the referee now fixes it.\n"
        "  If a NAME is wrong on every line, it is not the language - add it\n"
        "  as a keyterm and run this again with --keyterm.\n"
        "  If the name is right here but wrong in a call, the translator is\n"
        "  rewriting it; that is rule 6b in app/providers/llm_translator.\n"
        "  If the spectrum check complained, fix THAT first and ignore\n"
        "  everything else on this page - a recogniser cannot read formants\n"
        "  that were removed before it saw them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
