"""Speech-rate drift control.

Spec reference: §7.6.

THE PROBLEM NOBODY EXPECTS UNTIL IT HAPPENS
-------------------------------------------
A sentence in English and its Urdu translation do not take the same time to
speak. If the translation is systematically 15% longer, then during a
five-minute client monologue the playback queue grows by roughly forty-five
seconds. The consultant ends up listening to the beginning of a paragraph the
client finished a minute ago, and every subsequent turn is broken.

    queue depth        action                              perceptible?
    ---------------------------------------------------------------------
    < 400 ms           play at 1.00x                       no
    400 ms - 1.5 s     1.05-1.15x, pitch preserved         barely
    1.5 s - 3 s        up to 1.25x, ask for brevity        noticeable
    > 3 s              drop the oldest unit, tell the user yes

Two rules. The time-stretch must preserve pitch - naive resampling makes the
voice comic and is instantly disqualifying in a client meeting. And dropping
always beats growing: a missed sentence the consultant can ask about is
recoverable, a conversation running a minute behind is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DriftAction(str, Enum):
    NORMAL = "normal"
    SPEED_UP = "speed_up"
    SPEED_UP_AND_SHORTEN = "speed_up_and_shorten"
    DROP_OLDEST = "drop_oldest"


@dataclass(frozen=True)
class DriftDecision:
    action: DriftAction
    rate: float  # playback rate multiplier, pitch preserved
    prefer_brevity: bool  # instruct the translator to be more concise
    drop: bool

    @property
    def perceptible(self) -> bool:
        return self.rate > 1.05


class DriftController:
    """Maps playback-queue depth onto a playback rate and a drop decision."""

    def __init__(
        self,
        *,
        comfortable_ms: float = 400.0,
        stretch_ms: float = 1500.0,
        aggressive_ms: float = 3000.0,
        max_rate: float = 1.25,
    ) -> None:
        if not comfortable_ms < stretch_ms < aggressive_ms:
            raise ValueError("drift thresholds must be strictly increasing")
        self.comfortable_ms = comfortable_ms
        self.stretch_ms = stretch_ms
        self.aggressive_ms = aggressive_ms
        self.max_rate = max_rate
        self.drops = 0

    def decide(self, queue_ms: float) -> DriftDecision:
        if queue_ms <= self.comfortable_ms:
            return DriftDecision(DriftAction.NORMAL, 1.0, False, False)

        if queue_ms <= self.stretch_ms:
            # Ramp 1.00 -> 1.15 across the band.
            span = self.stretch_ms - self.comfortable_ms
            frac = (queue_ms - self.comfortable_ms) / span
            return DriftDecision(
                DriftAction.SPEED_UP, 1.0 + 0.15 * frac, False, False
            )

        if queue_ms <= self.aggressive_ms:
            span = self.aggressive_ms - self.stretch_ms
            frac = (queue_ms - self.stretch_ms) / span
            rate = min(self.max_rate, 1.15 + (self.max_rate - 1.15) * frac)
            return DriftDecision(
                DriftAction.SPEED_UP_AND_SHORTEN, rate, True, False
            )

        self.drops += 1
        return DriftDecision(DriftAction.DROP_OLDEST, self.max_rate, True, True)


# ----------------------------------------------------------------------


def time_stretch(pcm: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """Speed up `pcm` by `rate` while preserving pitch.

    Prefers a real time-stretch library. Falls back to WSOLA-lite - overlap-add
    at pitch-synchronous-ish offsets - which is not broadcast quality but is
    perfectly acceptable at the 1.0-1.25x range this is used for, and crucially
    does NOT shift pitch the way resampling would.

    Never resamples: a chipmunk voice in a client meeting is disqualifying.
    """
    if rate <= 1.0001:
        return pcm
    if pcm.size == 0:
        return pcm

    try:  # pragma: no cover - optional dependency
        import pyrubberband

        return pyrubberband.time_stretch(pcm, sample_rate, rate).astype(pcm.dtype)
    except Exception:
        pass

    return _wsola(pcm, rate, sample_rate)


def _wsola(pcm: np.ndarray, rate: float, sample_rate: int) -> np.ndarray:
    """Overlap-add time compression with a cross-correlation search.

    Window ~30 ms with 50% overlap. For each output frame we look for the input
    offset whose waveform best matches the tail we already emitted, which keeps
    successive windows phase-aligned and avoids the metallic buzz of naive OLA.
    """
    dtype = pcm.dtype
    x = pcm.astype(np.float32)
    n = x.size

    win_n = max(64, int(0.030 * sample_rate))
    hop_out = win_n // 2
    hop_in = int(round(hop_out * rate))
    search = max(1, win_n // 8)

    if n < win_n * 2:
        return pcm

    window = np.hanning(win_n).astype(np.float32)
    out = np.zeros(int(n / rate) + win_n, dtype=np.float32)
    norm = np.zeros_like(out)

    read = 0
    write = 0
    tail = np.zeros(hop_out, dtype=np.float32)

    while read + win_n + search < n and write + win_n < out.size:
        lo = max(0, read - search)
        hi = min(n - win_n, read + search)
        if hi > lo and tail.any():
            # Pick the offset whose head best matches what we just wrote.
            best, best_score = read, -np.inf
            for cand in range(lo, hi + 1, max(1, search // 4)):
                head = x[cand : cand + hop_out]
                if head.size != tail.size:
                    continue
                score = float(np.dot(head, tail))
                if score > best_score:
                    best_score, best = score, cand
            read = best

        seg = x[read : read + win_n] * window
        out[write : write + win_n] += seg
        norm[write : write + win_n] += window

        tail = x[read + hop_out : read + hop_out + hop_out]
        if tail.size != hop_out:
            tail = np.zeros(hop_out, dtype=np.float32)

        read += hop_in
        write += hop_out

    out = out[: write + win_n]
    norm = norm[: write + win_n]
    np.divide(out, np.maximum(norm, 1e-6), out=out)

    if dtype == np.int16:
        return np.clip(out, -32768, 32767).astype(np.int16)
    return out.astype(dtype)
