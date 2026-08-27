"""Score recognisers against the evaluation set.

Spec reference: §8.5, §15.2, §18 O2.

This is the tool that settles the single biggest open risk in the project.
Providers document Urdu. None of them document code-switched Urdu-English -
Urdu grammar carrying English technical nouns - which is how the consultant
actually speaks. That gap, not translation quality, is where this product is
most likely to fail, and it is invisible in any test using clean read speech.

REPORT TWO NUMBERS, NOT ONE
---------------------------
    overall WER              how good the recogniser looks
    technical-token WER      whether the product actually works

The second is restricted to the English technical vocabulary embedded in Urdu
speech. It is the number that predicts success, and it is hidden inside the
first. R1 in the risk register trips at roughly 15%.

Layout expected under testdata/eval-set/:

    clip001.wav
    clip001.txt        hand transcript, exactly as spoken
    clip002.wav
    clip002.txt
    ...

Usage:

    python tools/bench_asr.py --list
    python tools/bench_asr.py --provider deepgram --lang ur
    python tools/bench_asr.py --provider deepgram --lang ur --keyterms
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.formats import RATE_ASR  # noqa: E402
from app.providers.base import AsrConfig  # noqa: E402
from app.models.config import get_secret  # noqa: E402
from app.models.terminology import Terminology  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import read_wav  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "testdata" / "eval-set"

#: A token counts as "technical" if it is ASCII-alphabetic - i.e. an English
#: word sitting inside Urdu script. Crude, and deliberately so: it needs no
#: curation and it catches exactly the code-switched vocabulary in question.
ASCII_WORD = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")


def normalise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[.,!?;:۔،؟\"'()\[\]]", " ", text)
    return [t for t in text.split() if t]


def is_technical(token: str) -> bool:
    return bool(ASCII_WORD.match(token))


def wer(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    """Levenshtein distance over tokens. Returns (errors, reference length)."""
    if not reference:
        return (len(hypothesis), 0)
    prev = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        cur = [i]
        for j, h in enumerate(hypothesis, 1):
            cur.append(
                min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
            )
        prev = cur
    return (prev[-1], len(reference))


def technical_wer(reference: list[str], hypothesis: list[str]) -> tuple[int, int]:
    """Error rate restricted to the English technical tokens (§8.5 step 2)."""
    ref_tech = [t for t in reference if is_technical(t)]
    if not ref_tech:
        return (0, 0)
    hyp_set = set(hypothesis)
    missed = sum(1 for t in ref_tech if t not in hyp_set)
    return (missed, len(ref_tech))


@dataclass
class Result:
    clip: str
    reference: str = ""
    hypothesis: str = ""
    errors: int = 0
    ref_len: int = 0
    tech_errors: int = 0
    tech_len: int = 0
    error: str = ""


@dataclass
class Totals:
    results: list[Result] = field(default_factory=list)

    def add(self, r: Result) -> None:
        self.results.append(r)

    @property
    def overall(self) -> float:
        e = sum(r.errors for r in self.results)
        n = sum(r.ref_len for r in self.results)
        return e / n if n else float("nan")

    @property
    def technical(self) -> float:
        e = sum(r.tech_errors for r in self.results)
        n = sum(r.tech_len for r in self.results)
        return e / n if n else float("nan")


def build_provider(name: str):
    if name == "deepgram":
        from app.providers.deepgram_asr import DeepgramSpeechToText

        return DeepgramSpeechToText(get_secret("DEEPGRAM_API_KEY"))
    raise SystemExit(
        f"unknown provider {name!r}. Add it in tools/bench_asr.py once the "
        "client exists under va/providers/."
    )


async def transcribe(provider, wav: Path, lang: str, keyterms: list[str]) -> str:
    pcm = read_wav(str(wav), RATE_ASR)
    cfg = AsrConfig(language=lang, rate=RATE_ASR, keyterms=keyterms)
    stream = await provider.open(cfg)
    finals: list[str] = []

    async def collect():
        async for h in stream.hypotheses():
            if h.is_final and h.text.strip():
                finals.append(h.text.strip())

    task = asyncio.create_task(collect())
    step = RATE_ASR // 5  # 200 ms
    for i in range(0, len(pcm), step):
        await stream.push_audio(pcm[i : i + step])
        await asyncio.sleep(0.05)  # roughly real time; providers expect it
    await asyncio.sleep(1.5)
    await stream.aclose()
    task.cancel()
    return " ".join(finals)


async def run(args) -> int:
    clips = sorted(EVAL_DIR.glob("*.wav"))
    if not clips:
        print(
            f"\n  No evaluation set found at {EVAL_DIR}\n\n"
            "  This is the Phase 1 deliverable and the most valuable asset in\n"
            "  the project (§15.2). Record 30-60 minutes of your real speaking\n"
            "  style - real technical content, real code-switching, real pace,\n"
            "  on the real headset - and hand-transcribe it as clipNNN.txt\n"
            "  beside each clipNNN.wav.\n"
        )
        return 1

    if args.list:
        for c in clips:
            has_ref = "ok" if c.with_suffix(".txt").exists() else "NO TRANSCRIPT"
            print(f"  {c.name:<24} {has_ref}")
        return 0

    terms = Terminology.load(EVAL_DIR.parent / "terminology.json")
    keyterms = terms.keyterms() if args.keyterms else []
    provider = build_provider(args.provider)
    totals = Totals()

    print(f"\n  provider={args.provider}  lang={args.lang}  keyterms={bool(keyterms)}\n")

    for clip in clips:
        ref_path = clip.with_suffix(".txt")
        if not ref_path.exists():
            continue
        reference = normalise(ref_path.read_text(encoding="utf-8"))
        r = Result(clip.name, reference=" ".join(reference))
        try:
            hyp_text = await transcribe(provider, clip, args.lang, keyterms)
        except Exception as exc:
            r.error = str(exc)
            totals.add(r)
            print(f"  {clip.name:<20} ERROR {exc}")
            continue

        hypothesis = normalise(hyp_text)
        r.hypothesis = " ".join(hypothesis)
        r.errors, r.ref_len = wer(reference, hypothesis)
        r.tech_errors, r.tech_len = technical_wer(reference, hypothesis)
        totals.add(r)

        clip_wer = r.errors / r.ref_len if r.ref_len else float("nan")
        tech = f"{r.tech_errors}/{r.tech_len}" if r.tech_len else "-"
        print(f"  {clip.name:<20} WER {clip_wer:6.1%}   technical {tech}")
        if args.verbose:
            print(f"      ref: {r.reference}")
            print(f"      hyp: {r.hypothesis}\n")

    print("\n  " + "=" * 52)
    print(f"  overall WER            {totals.overall:6.1%}")
    print(f"  technical-token WER    {totals.technical:6.1%}   <- the one that matters")
    print("  " + "=" * 52)
    if totals.technical == totals.technical and totals.technical > 0.15:
        print(
            "\n  R1 TRIPPED: technical-token error above 15%.\n"
            "  Try --keyterms, then the speech-to-speech path, before\n"
            "  committing to this provider (§8.5).\n"
        )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--provider", default="deepgram")
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--keyterms", action="store_true", help="load terminology hints")
    ap.add_argument("--list", action="store_true", help="show the evaluation set")
    ap.add_argument("-v", "--verbose", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
