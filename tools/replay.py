"""Offline pipeline replay.

Spec reference: §10.5, §15.1, §9.3.

Feeds a WAV file through the VAD, the stability filter, the segmenter and the
latency instrumentation, with no audio devices and no GUI, printing a
UnitTrace per unit. This is what makes latency tuning and regression testing
possible without a meeting, a headset or another human.

    python tools/replay.py sample.wav -v
    python tools/replay.py sample.wav --hangover 250 --stability 200
    python tools/replay.py sample.wav --sweep stability

The last two forms are the point of the tool: the endpointing settings are the
dominant latency term (§9.3), and they must be tuned against recordings rather
than in a live meeting.

WHAT THIS TOOL DOES NOT DO
--------------------------
It does not measure recognition accuracy - the hypotheses are synthetic. Word
error rate on real speech is tools/bench_asr.py against the evaluation set
(§15.2), and that is the number that decides whether the product works.
"""

from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.audio import convert  # noqa: E402
from app.audio.formats import DTYPE, RATE_ASR  # noqa: E402
from app.controllers.segmenter import Hypothesis, Segmenter  # noqa: E402
from app.controllers.vad import (  # noqa: E402
    EnergyDetector,
    SpeechEvent,
    VoiceActivityGate,
    frames_of,
)
from app.models.telemetry import (  # noqa: E402
    Direction,
    MetricsBus,
    TraceFactory,
    format_debug_view,
)

# Modelled network legs, from the §9.2 p50 column.
MT_S = 0.17
TTS_S = 0.14
PLAYBACK_S = 0.02

DEFAULT_SCRIPT = [
    "Can you finish the API integration by Friday?",
    "We already tested the endpoint but not in production.",
    "Let us move the deployment to the next sprint and confirm the scope.",
]


def read_wav(path: str, target_rate: int = RATE_ASR) -> np.ndarray:
    with wave.open(path, "rb") as w:
        channels, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    return convert.endpoint_to_pipeline(
        np.frombuffer(raw, dtype=DTYPE), rate, channels, target_rate
    )


def mock_hypotheses(text: str, emit_final: bool) -> list[str]:
    """Growing prefixes, one revision, then repetition while the speaker pauses.

    The trailing repetition matters. A real recogniser keeps returning the same
    hypothesis while it waits for its own endpointing, and that is exactly what
    lets the stability filter accumulate. Without it, the stable path can never
    fire and the endpointing settings would appear to do nothing - which is how
    the first version of this tool quietly lied during tuning.
    """
    words = text.split()
    out: list[str] = []
    for i in range(1, len(words) + 1):
        prefix = " ".join(words[:i])
        out.append(prefix)
        if i == max(1, len(words) // 2):
            out.append(prefix)  # a revision pass, as real recognisers do
    out.extend([text] * 4)  # holding steady while the speaker trails off
    return out


@dataclass
class Pending:
    at: float
    text: str
    is_final: bool


def run(args) -> dict:
    pcm = read_wav(args.wav)
    gate = VoiceActivityGate(
        EnergyDetector(args.threshold),
        rate=RATE_ASR,
        hangover_ms=args.hangover,
        preroll_ms=args.preroll,
    )
    segmenter = Segmenter(
        min_hypotheses=args.min_hyp,
        stable_ms=args.stability,
        max_words=args.max_words,
    )
    metrics = MetricsBus()
    traces = TraceFactory()
    script = args.script or DEFAULT_SCRIPT

    queue: list[Pending] = []
    utterances = 0
    units = 0
    onset_at = 0.0
    trace = None
    t = 0.0
    frame_s = 0.020

    def emit(unit, at: float) -> None:
        nonlocal units, trace
        units += 1
        tr = trace or traces.new(Direction.INCOMING)
        tr.t_speech_onset = onset_at
        tr.mark("unit_emitted", at)
        tr.words = unit.words
        tr.mark("translated", at + MT_S)
        tr.mark("tts_first_chunk", at + MT_S + TTS_S)
        tr.mark("playback_start", at + MT_S + TTS_S + PLAYBACK_S)
        metrics.record(tr)
        if args.verbose:
            print(f"    [{unit.reason:>9}] +{tr.ttfta:5.2f}s  {unit.text}")
        trace = None

    for frame in frames_of(pcm, RATE_ASR):
        res = gate.push(frame, at=t)

        if res.event is SpeechEvent.ONSET:
            utterances += 1
            onset_at = t
            trace = traces.new(Direction.INCOMING)
            trace.t_speech_onset = t
            trace.mark("frame_sent", t)
            # Schedule the hypotheses across real time rather than dumping them
            # in one loop - otherwise the stability timing is never exercised.
            texts = mock_hypotheses(script[(utterances - 1) % len(script)], args.emit_final)
            step = args.hyp_interval / 1000.0
            queue = [
                Pending(t + 0.15 + i * step, txt, args.emit_final and i == len(texts) - 1)
                for i, txt in enumerate(texts)
            ]

        # Deliver any hypotheses that have come due.
        while queue and queue[0].at <= t:
            p = queue.pop(0)
            if trace is not None and trace.t_hypothesis == 0.0:
                trace.mark("hypothesis", p.at)
            for unit in segmenter.push(Hypothesis(p.text, p.is_final, p.at)):
                emit(unit, p.at)

        if res.event is SpeechEvent.OFFSET:
            queue.clear()
            for unit in segmenter.flush():
                emit(unit, t)  # VAD-flushed units are traced too

        t += frame_s

    for unit in segmenter.flush():
        emit(unit, t)

    snap = metrics.snapshot()
    snap["_utterances"] = utterances
    snap["_units"] = units
    return snap


def report(args, snap: dict) -> int:
    print(f"\n  utterances detected: {snap['_utterances']}")
    print(f"  units emitted:       {snap['_units']}\n")
    print(format_debug_view(snap, "incoming"))
    t = snap["incoming"]["ttfta"]
    print(
        f"\n  hangover={args.hangover:.0f}ms  stability={args.stability:.0f}ms  "
        f"min_hyp={args.min_hyp}  max_words={args.max_words}"
    )
    if t["p50"] is None:
        print("  no TTFTA recorded - nothing was emitted\n")
        return 1
    verdict = "PASS" if t["p50"] <= 1.5 else "OVER BUDGET"
    print(f"  TTFTA p50 {t['p50']:.2f}s  p90 {t['p90']:.2f}s   {verdict} (§1.5)\n")
    return 0


def sweep(args, knob: str) -> int:
    values = {
        "stability": [100, 200, 350, 500, 700],
        "hangover": [200, 300, 400, 600, 800],
        "max_words": [6, 8, 12, 16, 24],
        "min_hyp": [1, 2, 3, 4],
    }[knob]
    print(f"\n  Sweeping {knob} — the §9.3 tuning exercise\n")
    print(f"  {knob:>10}   units   TTFTA p50   TTFTA p90")
    print("  " + "-" * 46)
    for v in values:
        setattr(args, knob, v)
        snap = run(args)
        t = snap["incoming"]["ttfta"]
        p50 = "—" if t["p50"] is None else f"{t['p50']:.2f}s"
        p90 = "—" if t["p90"] is None else f"{t['p90']:.2f}s"
        print(f"  {v:>10}   {snap['_units']:>5}   {p50:>9}   {p90:>9}")
    print(
        "\n  Lower is faster, but each step down raises the risk of "
        "translating half a sentence.\n"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("wav", help="16-bit PCM WAV file")
    ap.add_argument("--hangover", type=float, default=400.0, help="VAD hangover ms")
    ap.add_argument("--preroll", type=float, default=250.0, help="VAD pre-roll ms")
    ap.add_argument("--stability", type=float, default=350.0, help="stability ms")
    ap.add_argument("--min-hyp", type=int, default=3, dest="min_hyp")
    ap.add_argument("--max-words", type=int, default=12, dest="max_words")
    ap.add_argument("--threshold", type=float, default=-45.0, help="VAD dBFS")
    ap.add_argument(
        "--hyp-interval",
        type=float,
        default=120.0,
        dest="hyp_interval",
        help="ms between simulated hypotheses",
    )
    ap.add_argument("--script", nargs="*", help="utterance texts to simulate")
    ap.add_argument(
        "--emit-final",
        action="store_true",
        dest="emit_final",
        help="let the mock recogniser mark the last hypothesis final; this "
        "bypasses the stability path, so the endpointing knobs stop mattering",
    )
    ap.add_argument(
        "--sweep",
        choices=["stability", "hangover", "max_words", "min_hyp"],
        help="try a range of values for one knob and tabulate the effect",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        return sweep(args, args.sweep)
    print(f"\n  {args.wav}\n")
    return report(args, run(args))


if __name__ == "__main__":
    raise SystemExit(main())
