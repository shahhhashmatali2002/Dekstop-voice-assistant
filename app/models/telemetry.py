"""Latency measurement.

Spec reference: §9.1, §9.4, §11.3.

TTFTA - Time To First Translated Audio - is the interval from the instant a
speaker begins an utterance to the instant the first sample of its translation
reaches the listener's ear. It is the only latency number that describes the
user experience, and it is what the overlay displays.

Report it as a distribution, never as a single figure. p50 describes how the
product feels; p90 describes how often it is annoying.

Everything here uses time.perf_counter(), never time.time(): the wall clock
can step backwards and would produce negative durations.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields
from enum import Enum


class Direction(str, Enum):
    """Which way a unit is travelling."""

    INCOMING = "incoming"  # EN -> UR, client to consultant
    OUTGOING = "outgoing"  # UR -> EN, consultant to client

    @property
    def label(self) -> str:
        return "EN -> UR" if self is Direction.INCOMING else "UR -> EN"


def now() -> float:
    """Monotonic timestamp in seconds."""
    return time.perf_counter()


# ----------------------------------------------------------------------


@dataclass
class UnitTrace:
    """Timestamps for one translation unit, written at each stage boundary.

    Nothing here is estimated. A stage that never ran leaves its field at 0.0
    and is reported as absent rather than as zero duration.
    """

    unit_id: int
    direction: Direction

    t_speech_onset: float = 0.0  # VAD said speech started
    t_frame_sent: float = 0.0  # first frame handed to the provider
    t_hypothesis: float = 0.0  # first hypothesis received
    t_unit_emitted: float = 0.0  # segmenter released the unit
    t_translated: float = 0.0  # translation returned
    t_tts_first_chunk: float = 0.0  # first PCM chunk received
    t_playback_start: float = 0.0  # first sample written to the render ring

    words: int = 0
    dropped: bool = False
    drop_reason: str = ""

    # ------------------------------------------------------------------

    def mark(self, stage: str, at: float | None = None) -> None:
        """Stamp `t_<stage>` with the current time."""
        attr = f"t_{stage}"
        if not hasattr(self, attr):
            raise AttributeError(f"UnitTrace has no stage {stage!r}")
        setattr(self, attr, at if at is not None else now())

    #: Sentinel for "this stage never ran". perf_counter() is never exactly
    #: zero in practice, but comparing explicitly beats relying on truthiness -
    #: a 0.0 that *is* meaningful would otherwise vanish silently.
    UNSET = 0.0

    @property
    def ttfta(self) -> float | None:
        """Seconds from speech onset to first translated audio."""
        if self.t_speech_onset == self.UNSET or self.t_playback_start == self.UNSET:
            return None
        return self.t_playback_start - self.t_speech_onset

    def stages(self) -> dict[str, float]:
        """Per-stage durations in seconds, skipping stages that did not run."""
        seq = [
            ("capture", "t_speech_onset", "t_frame_sent"),
            ("asr", "t_frame_sent", "t_hypothesis"),
            ("segmentation", "t_hypothesis", "t_unit_emitted"),
            ("translation", "t_unit_emitted", "t_translated"),
            ("tts", "t_translated", "t_tts_first_chunk"),
            ("playback", "t_tts_first_chunk", "t_playback_start"),
        ]
        out: dict[str, float] = {}
        for name, a, b in seq:
            ta, tb = getattr(self, a), getattr(self, b)
            if ta != self.UNSET and tb != self.UNSET and tb >= ta:
                out[name] = tb - ta
        return out

    def as_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["direction"] = self.direction.value
        d["ttfta"] = self.ttfta
        return d


# ----------------------------------------------------------------------


class Percentiles:
    """Sliding-window percentile tracker.

    Exact rather than approximate: the window is small (a minute of speech is
    a few hundred units at most) and sorting it is far cheaper than the audio
    work happening alongside.
    """

    __slots__ = ("_window_s", "_samples")

    def __init__(self, window_s: float = 60.0) -> None:
        self._window_s = window_s
        self._samples: deque[tuple[float, float]] = deque()

    def add(self, value: float, at: float | None = None) -> None:
        at = at if at is not None else now()
        self._samples.append((at, value))
        self._evict(at)

    def _evict(self, at: float) -> None:
        cutoff = at - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def __len__(self) -> int:
        return len(self._samples)

    def snapshot(self, at: float | None = None) -> dict[str, float | None]:
        # `at` is injectable so the window can be reasoned about in tests and
        # in offline replay, where timestamps come from a file rather than the
        # clock. Production always passes None and gets perf_counter().
        self._evict(at if at is not None else now())
        if not self._samples:
            return {"n": 0, "p50": None, "p90": None, "max": None}
        vals = sorted(v for _, v in self._samples)
        n = len(vals)

        def pct(p: float) -> float:
            # Nearest-rank; with a window this small, interpolation is noise.
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return vals[idx]

        return {"n": n, "p50": pct(0.50), "p90": pct(0.90), "max": vals[-1]}


# ----------------------------------------------------------------------


@dataclass
class DirectionMetrics:
    """Rolling metrics for one direction."""

    direction: Direction
    window_s: float = 60.0

    ttfta: Percentiles = field(init=False)
    stages: dict[str, Percentiles] = field(init=False)
    units: int = 0
    dropped: int = 0
    reconnects: int = 0
    queue_depth_ms: float = 0.0
    drift_rate: float = 1.0
    provider_rtt_ms: float = 0.0

    def __post_init__(self) -> None:
        self.ttfta = Percentiles(self.window_s)
        self.stages = {}

    def record(self, trace: UnitTrace) -> None:
        self.units += 1
        if trace.dropped:
            self.dropped += 1
            return
        t = trace.ttfta
        if t is not None:
            self.ttfta.add(t)
        for name, dur in trace.stages().items():
            self.stages.setdefault(name, Percentiles(self.window_s)).add(dur)

    def snapshot(self) -> dict:
        return {
            "direction": self.direction.value,
            "ttfta": self.ttfta.snapshot(),
            "stages": {k: v.snapshot() for k, v in self.stages.items()},
            "units": self.units,
            "dropped": self.dropped,
            "reconnects": self.reconnects,
            "queue_depth_ms": round(self.queue_depth_ms, 1),
            "drift_rate": round(self.drift_rate, 3),
            "provider_rtt_ms": round(self.provider_rtt_ms, 1),
        }


class MetricsBus:
    """Aggregates traces and counters; emits snapshots to the UI at ~4 Hz.

    Every stage writes here. No stage computes a percentile itself and no stage
    takes a lock to record a measurement - the lock below is held only for the
    duration of a dict update, never across I/O.
    """

    def __init__(self, window_s: float = 60.0) -> None:
        self._lock = threading.Lock()
        self._by_direction = {
            d: DirectionMetrics(d, window_s) for d in Direction
        }
        self._audio: dict[str, dict] = {}
        self._started = now()

    def record(self, trace: UnitTrace) -> None:
        with self._lock:
            self._by_direction[trace.direction].record(trace)

    def set_queue_depth(self, direction: Direction, ms: float) -> None:
        with self._lock:
            self._by_direction[direction].queue_depth_ms = ms

    def set_drift(self, direction: Direction, rate: float) -> None:
        with self._lock:
            self._by_direction[direction].drift_rate = rate

    def set_rtt(self, direction: Direction, ms: float) -> None:
        with self._lock:
            self._by_direction[direction].provider_rtt_ms = ms

    def note_reconnect(self, direction: Direction) -> None:
        with self._lock:
            self._by_direction[direction].reconnects += 1

    def set_audio_health(self, name: str, health: dict) -> None:
        with self._lock:
            self._audio[name] = dict(health)

    def ttfta(self, direction: Direction) -> dict:
        with self._lock:
            return self._by_direction[direction].ttfta.snapshot()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "uptime_s": round(now() - self._started, 1),
                "incoming": self._by_direction[Direction.INCOMING].snapshot(),
                "outgoing": self._by_direction[Direction.OUTGOING].snapshot(),
                "audio": dict(self._audio),
            }


# ----------------------------------------------------------------------


class TraceFactory:
    """Hands out monotonically increasing unit IDs, per direction."""

    def __init__(self) -> None:
        self._counters = {d: 0 for d in Direction}
        self._lock = threading.Lock()

    def new(self, direction: Direction) -> UnitTrace:
        with self._lock:
            self._counters[direction] += 1
            uid = self._counters[direction]
        t = UnitTrace(unit_id=uid, direction=direction)
        t.mark("speech_onset")
        return t


def format_debug_view(snapshot: dict, direction: str = "incoming") -> str:
    """The pipeline debugger from Figure 12, as text."""
    d = snapshot[direction]
    order = ["capture", "asr", "segmentation", "translation", "tts", "playback"]
    lines = [
        "+" + "-" * 61 + "+",
        f"|  PIPELINE DEBUGGER{'':>18}direction: {direction:<12}|",
        "+" + "-" * 61 + "+",
        f"|{'':36}p50      p90      max     |",
    ]

    def fmt(v):
        if v is None:
            return "    -   "
        return f"{v * 1000:6.0f} ms" if v < 1 else f"{v:6.2f} s "

    for name in order:
        s = d["stages"].get(name)
        if not s:
            continue
        lines.append(
            f"|   {name:<22}{fmt(s['p50'])} {fmt(s['p90'])} {fmt(s['max'])}   |"
        )
    t = d["ttfta"]
    lines.append("|   " + "-" * 55 + "  |")
    lines.append(
        f"|   {'TTFTA':<22}{fmt(t['p50'])} {fmt(t['p90'])} {fmt(t['max'])}   |"
    )
    lines.append("|" + " " * 61 + "|")
    lines.append(
        f"|   units {d['units']:<6} dropped {d['dropped']:<4} "
        f"reconnects {d['reconnects']:<4}{'':10}|"
    )
    lines.append(
        f"|   queue {d['queue_depth_ms']:>6.0f} ms   drift {d['drift_rate']:.2f}x"
        f"   rtt {d['provider_rtt_ms']:>5.0f} ms   |"
    )
    lines.append("+" + "-" * 61 + "+")
    return "\n".join(lines)
