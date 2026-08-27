"""Stability filtering and segmentation.

Spec reference: §6.3, §6.3.1.

Streaming recognition revises its own output:

    "I think we should cha..."  ->  "I think we should change"
                               ->  "I think we should charge"

Sending every partial to the translator is wasteful and produces jitter.
Waiting for a full pause is accurate but slow. The compromise is a segmenter
that emits translation units at natural boundaries:

    emit a unit when ANY of:
      (a) the recogniser marks the hypothesis final
      (b) the prefix has been stable for >= stable_ms
          AND ends at a clause boundary
      (c) the pending text has grown past max_words
      (d) VAD reports end of speech

    and never emit a prefix of something already emitted.

ONCE EMITTED, TEXT IS FROZEN
----------------------------
A later revision to already-spoken words is discarded, because the Urdu audio
has already left the building. That makes the stability threshold a direct
quality-for-latency trade: start conservative and tighten only with
measurement. A visible mistranslation in front of a client costs far more than
200 ms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from app.models.telemetry import now

# Sentence-ish boundaries in both scripts. Urdu uses the danda (۔), the Arabic
# comma (،) and question mark (؟); English punctuation appears constantly in
# code-switched speech, so both sets are live at once.
CLAUSE_ENDS = ".!?;:۔؟…"
SOFT_ENDS = ",،-—"


@dataclass
class Hypothesis:
    """One partial or final result from a streaming recogniser."""

    text: str
    is_final: bool = False
    at: float = field(default_factory=now)
    confidence: float | None = None
    speaker: str | None = None


@dataclass
class Unit:
    """A translation unit: a frozen span of source text, emitted exactly once."""

    text: str
    reason: str  # which rule fired: final / stable / max_words / vad
    at: float = field(default_factory=now)
    speaker: str | None = None

    @property
    def words(self) -> int:
        return len(self.text.split())


def _common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def _ends_at_word_boundary(prefix: str, texts) -> bool:
    """True when no candidate continues the prefix mid-word.

    "I think" is a clean boundary against "I think we" (the next char is a
    space) but "I think we should cha" is not, against "...change".
    """
    for t in texts:
        if len(t) > len(prefix) and not t[len(prefix)].isspace():
            return False
    return True


def _trim_to_word_boundary(s: str) -> str:
    """Never cut a word in half - a half word poisons the translation."""
    if not s or s[-1].isspace():
        return s.rstrip()
    idx = s.rfind(" ")
    return s[:idx].rstrip() if idx > 0 else ""


def _last_clause_end(s: str) -> int:
    """Index just past the last clause-ending punctuation, or -1."""
    best = -1
    for ch in CLAUSE_ENDS:
        best = max(best, s.rfind(ch))
    return best + 1 if best >= 0 else -1


class StabilityFilter:
    """Tracks how long the leading prefix of the hypothesis has been unchanged.

    A prefix is stable when it is a prefix of ALL of the last `min_hypotheses`
    hypotheses AND has been so for `stable_ms`.

    Implementation note, learned the hard way
    -----------------------------------------
    The obvious implementation - fold the running prefix with each new
    hypothesis - is wrong. A common prefix computed that way can only ever
    shrink, so for text that grows monotonically ("Can", "Can you", "Can you
    finish") it locks onto the first word and never recovers, and the stable
    path silently never fires again.

    Keeping a window of the last N hypotheses and recomputing their common
    prefix each time fixes it: while the speaker is still producing new words
    the prefix keeps moving and nothing is confirmed, and the moment they pause
    and the recogniser starts repeating itself, the prefix settles and the
    whole utterance becomes eligible at once. That is exactly the behaviour we
    want.
    """

    def __init__(self, min_hypotheses: int = 3, stable_ms: float = 350.0) -> None:
        self.min_hypotheses = max(1, min_hypotheses)
        self.stable_ms = stable_ms
        self._window: deque[str] = deque(maxlen=self.min_hypotheses)
        self._prefix = ""
        self._since = 0.0
        self._latest = ""

    @property
    def latest(self) -> str:
        """The most recent hypothesis in full, stable or not."""
        return self._latest

    def update(self, text: str, at: float | None = None) -> str:
        """Feed a hypothesis; return the currently stable prefix."""
        at = at if at is not None else now()
        self._latest = text
        self._window.append(text)

        if len(self._window) < self.min_hypotheses:
            return ""

        prefix = self._window[0]
        for other in list(self._window)[1:]:
            prefix = _common_prefix(prefix, other)
            if not prefix:
                break
        # Never confirm a half word: "chang" would poison the translation.
        # But only trim when the prefix actually splits one - trimming
        # unconditionally throws away the last complete word every time.
        if not _ends_at_word_boundary(prefix, self._window):
            prefix = _trim_to_word_boundary(prefix)

        if prefix != self._prefix:
            self._prefix = prefix
            self._since = at

        # A freshly-changed prefix still has to hold for stable_ms. With
        # stable_ms == 0 it is confirmed immediately, which is what the tests
        # and the replay tool rely on.
        if (at - self._since) * 1000.0 < self.stable_ms:
            return ""
        return self._prefix

    def reset(self) -> None:
        self._window.clear()
        self._prefix = ""
        self._since = 0.0
        self._latest = ""


class Segmenter:
    """Turns a stream of hypotheses into a stream of frozen units."""

    def __init__(
        self,
        *,
        min_hypotheses: int = 3,
        stable_ms: float = 350.0,
        max_words: int = 12,
        min_words: int = 2,
    ) -> None:
        self.stability = StabilityFilter(min_hypotheses, stable_ms)
        self.max_words = max_words
        self.min_words = min_words
        self._emitted = ""  # frozen; never revisited

    # ------------------------------------------------------------------

    @property
    def emitted(self) -> str:
        return self._emitted

    @property
    def pending(self) -> str:
        return ""

    def reset(self) -> None:
        """Call at end of utterance, so the next one starts clean."""
        self.stability.reset()
        self._emitted = ""

    # ------------------------------------------------------------------

    def push(self, hyp: Hypothesis) -> list[Unit]:
        """Feed one hypothesis. Returns zero or more units to translate."""
        units: list[Unit] = []
        text = hyp.text.strip()

        # Rule (a): the recogniser says this is final. Emit the remainder and
        # reset - nothing more is coming for this utterance.
        if hyp.is_final:
            rest = self._remainder(text)
            if rest:
                units.append(Unit(rest, "final", hyp.at, hyp.speaker))
                self._emitted = (self._emitted + " " + rest).strip()
            self.stability.reset()
            return units

        stable = self.stability.update(text, hyp.at)
        rest = self._remainder(stable)
        if not rest:
            return units

        # Rule (c): too long to keep holding. Cut at the last clause boundary
        # if there is one, otherwise at a word boundary.
        if len(rest.split()) >= self.max_words:
            cut = _last_clause_end(rest)
            chunk = rest[:cut] if cut > 0 else _trim_to_word_boundary(rest)
            chunk = chunk.strip()
            if chunk and len(chunk.split()) >= self.min_words:
                units.append(Unit(chunk, "max_words", hyp.at, hyp.speaker))
                self._emitted = (self._emitted + " " + chunk).strip()
                return units

        # Rule (b): stable AND ends at a clause boundary.
        if rest and rest[-1] in CLAUSE_ENDS:
            if len(rest.split()) >= self.min_words:
                units.append(Unit(rest, "stable", hyp.at, hyp.speaker))
                self._emitted = (self._emitted + " " + rest).strip()
        return units

    def flush(self, reason: str = "vad") -> list[Unit]:
        """Rule (d): VAD reported end of speech. Emit whatever is held.

        This is the safety net. Everything not yet emitted goes now, regardless
        of stability, because no further hypotheses are coming - so it flushes
        the LATEST hypothesis in full, not the stable prefix. Flushing only the
        stable prefix would silently truncate every utterance to its first few
        confirmed words.
        """
        held = self.stability.latest.strip()
        rest = self._remainder(held)
        if rest and len(rest.split()) >= 1:
            self._emitted = (self._emitted + " " + rest).strip()
            self.stability.reset()
            return [Unit(rest, reason)]
        return []


    # ------------------------------------------------------------------

    def _remainder(self, text: str) -> str:
        """The part of `text` beyond what has already been emitted."""
        if not text:
            return ""
        text_clean = text.strip()
        if not self._emitted:
            return text_clean
        emitted_clean = self._emitted.strip()

        t_norm = text_clean.rstrip(".!?;:۔؟…").lower()
        e_norm = emitted_clean.rstrip(".!?;:۔؟…").lower()

        if t_norm == e_norm or t_norm.endswith(e_norm) or e_norm.endswith(t_norm):
            return ""
        if text_clean.startswith(emitted_clean):
            return text_clean[len(emitted_clean):].strip()
        if e_norm and t_norm.startswith(e_norm):
            return text_clean[len(emitted_clean):].strip()

        common = _common_prefix(emitted_clean, text_clean)
        rem = text_clean[len(common):].strip()
        if rem.lower().rstrip(".!?;:۔؟…") == t_norm:
            return ""
        return rem

