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
      (b2) the prefix has been stable for >= stable_ms, ends at a SOFT
           boundary (a comma), and is at least soft_min_words long -
           fluent speech produces commas for whole minutes at a time
           without a single full stop, and waiting for one means every
           unit is cut by max_words instead, seconds after the clause
           was actually finished
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


def _words(text: str) -> list[str]:
    """Comparable words: lowercase, stripped of the punctuation ASR moves around."""
    trailing = ".,!?;:\u06d4\u060c\u061f\u2026\"\u2019'"
    return [w.strip(trailing).lower() for w in text.split() if w.strip(trailing)]


def _starts_with(whole: list[str], prefix: list[str]) -> bool:
    return len(prefix) <= len(whole) and whole[: len(prefix)] == prefix


def _same(a: str, b: str) -> bool:
    """Two renderings of the same sentence, ignoring trailing punctuation."""
    trailing = ".!?;:\u06d4\u061f\u2026 "
    return a.strip().rstrip(trailing).lower() == b.strip().rstrip(trailing).lower()


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
        self._history: list[tuple[str, float]] = []
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

        if not prefix:
            return self._prefix

        # AN EXTENSION IS NOT A CHANGE.
        #
        # This is the distinction the filter used to miss, and missing it made
        # it silent. "the prefix changed, so restart the clock" treats
        #
        #     "Hey welcome back"  ->  "Hey welcome back so today"
        #
        # as a disagreement. It is not: every word in the first is still
        # agreed, and has been agreed for LONGER. But hypotheses arrive every
        # few hundred milliseconds and each one extends the prefix, so the
        # clock restarted on every single one and `at - _since` was always
        # zero. Nothing was ever confirmed while somebody kept talking: 61
        # words of continuous speech produced no units at all, and the only
        # line that came out was the one the VAD flushed at the first pause.
        #
        # So the clock belongs to each prefix separately. A word is confirmed
        # once it has been in the common prefix for `stable_ms`, whatever has
        # been appended after it since.
        if self._history and not prefix.startswith(self._history[-1][0]):
            # A genuine revision - the recogniser changed a word it had
            # already agreed on. Nothing is confirmed until this settles.
            self._history = [(prefix, at)]
            self._prefix = ""
        elif not self._history or prefix != self._history[-1][0]:
            self._history.append((prefix, at))

        deadline = at - self.stable_ms / 1000.0
        for candidate, first_seen in self._history:
            if first_seen <= deadline and len(candidate) > len(self._prefix):
                self._prefix = candidate
                self._since = first_seen

        # A long utterance would otherwise keep one entry per hypothesis.
        # Older entries are shorter, and a longer confirmed prefix supersedes
        # them.
        if len(self._history) > 64:
            del self._history[:-32]
        return self._prefix

    def reset(self) -> None:
        self._window.clear()
        self._prefix = ""
        self._since = 0.0
        self._history: list[tuple[str, float]] = []
        self._latest = ""


class Segmenter:
    """Turns a stream of hypotheses into a stream of frozen units."""

    def __init__(
        self,
        *,
        min_hypotheses: int = 3,
        stable_ms: float = 350.0,
        max_words: int = 12,
        min_words: int = 3,
        soft_min_words: int = 6,
    ) -> None:
        self.stability = StabilityFilter(min_hypotheses, stable_ms)
        self.max_words = max_words
        self.min_words = min_words
        self.soft_min_words = soft_min_words
        self._emitted = ""  # frozen; never revisited - WITHIN one utterance
        self._last_spoken = ""  # the previous utterance, to catch repeat finals
        self._normal: tuple[float, int] | None = None  # settings held during a hurry

    # ------------------------------------------------------------------

    @property
    def emitted(self) -> str:
        return self._emitted

    @property
    def pending(self) -> str:
        return ""

    def hurry(self, *, stable_ms: float = 120.0, max_words: int = 7) -> None:
        """Emit the FIRST unit early, then go back to the configured settings.

        Pressing Start in the middle of a sentence and then waiting for a
        whole clause to be confirmed feels broken, however correct it is: the
        listener has no way to tell "working, nothing to say yet" from "not
        working". Cutting the first unit short gets a voice into their ears
        quickly; everything after it is worth the wait it costs.
        """
        if self._normal is not None:
            return
        self._normal = (self.stability.stable_ms, self.max_words)
        self.stability.stable_ms = stable_ms
        self.max_words = max_words

    def _settle(self) -> None:
        """Something was spoken. Back to the settings the user chose."""
        if self._normal is None:
            return
        self.stability.stable_ms, self.max_words = self._normal
        self._normal = None

    def reset(self) -> None:
        """Start completely clean. Called when the pipeline stops."""
        self.stability.reset()
        self._emitted = ""
        self._last_spoken = ""

    def _end_utterance(self) -> None:
        """The speaker finished. Clear per-utterance state, remember the text.

        `_emitted` exists to stop the SAME utterance being spoken twice while
        the recogniser is still revising it. It is per-utterance state, and
        carrying it into the next one is not a stricter version of that rule -
        it is a different rule, and a fatal one: `_remainder` then compares
        brand-new speech against a sentence that has nothing to do with it,
        finds no common prefix, and returns nothing. The first utterance of a
        session is spoken and every utterance after it is silently swallowed.

        What genuinely does need to survive is the last utterance's text, so
        that a recogniser which sends its final result twice - or a VAD flush
        landing on top of a final - does not say the same sentence twice.
        """
        # Only if something was actually said. An utterance that produced
        # nothing - a duplicate final, say - must not wipe the guard that
        # suppressed it, or the third copy gets through.
        if self._emitted:
            self._last_spoken = self._emitted
        self.stability.reset()
        self._emitted = ""

    # ------------------------------------------------------------------

    def push(self, hyp: Hypothesis) -> list[Unit]:
        """Feed one hypothesis. Returns zero or more units to translate."""
        units: list[Unit] = []
        text = hyp.text.strip()

        # An EMPTY final is Deepgram's UtteranceEnd event (a word-gap close,
        # see utterance_end_ms): it carries no transcript of its own and
        # means "the clause you have been hearing is finished." Flush what is
        # held rather than treating the empty text as a real final, which
        # would emit nothing and could drop the clause. This is what makes
        # reading a paragraph work - the closes arrive on natural word gaps,
        # not only on silences a fluent reader never leaves.
        if hyp.is_final and not text:
            return self.flush("utterance_end")

        # Rule (a): the recogniser says this is final. Emit the remainder and
        # reset - nothing more is coming for this utterance.
        if hyp.is_final:
            rest = self._remainder(text)
            if rest:
                units.append(Unit(rest, "final", hyp.at, hyp.speaker))
                self._emitted = (self._emitted + " " + rest).strip()
            self._end_utterance()
            if units:
                self._settle()
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
                self._settle()
                return units

        # Rule (b): stable AND ends at a clause boundary.
        if rest and rest[-1] in CLAUSE_ENDS:
            if len(rest.split()) >= self.min_words:
                units.append(Unit(rest, "stable", hyp.at, hyp.speaker))
                self._emitted = (self._emitted + " " + rest).strip()
                self._settle()
            return units

        # Rule (b2): stable AND ends at a soft boundary, with enough words
        # that the clause stands alone as a translation unit. This is what
        # keeps CONTINUOUS speech flowing: a fluent speaker produces clause
        # after clause separated by commas, and a segmenter that only knows
        # hard punctuation sits on all of it until max_words truncates the
        # thought mid-clause. The word floor is higher than min_words on
        # purpose - "yes," alone is worth holding for its sentence, while a
        # six-word clause is already worth hearing.
        if (
            rest
            and rest[-1] in SOFT_ENDS
            and len(rest.split()) >= self.soft_min_words
        ):
            units.append(Unit(rest, "soft", hyp.at, hyp.speaker))
            self._emitted = (self._emitted + " " + rest).strip()
            self._settle()
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
        # `_emitted` is NOT cleared here, and that is the whole point of this
        # method's existence being separate from `_end_utterance`. The VAD and
        # the recogniser disagree about where an utterance ends, constantly: a
        # 400 ms breath is silence to the gate, while the recogniser keeps
        # extending the same hypothesis across it. Forgetting what we already
        # spoke at every breath means the next hypothesis - which still
        # contains all of it - gets spoken again from the top, with a few new
        # words on the end. That is the stutter of half-repeated sentences.
        self.stability.reset()
        if rest and len(rest.split()) >= 1:
            self._emitted = (self._emitted + " " + rest).strip()
            self._settle()
            return [Unit(rest, reason)]
        return []


    # ------------------------------------------------------------------

    def _remainder(self, text: str) -> str:
        """The part of `text` we have not already spoken.

        Compared word by word, ignoring punctuation and case, because the
        recogniser revises both freely while it is still listening -
        "student" becomes "student?" becomes "student," and none of those is
        new speech.

        The last branch matters as much as the first. If a hypothesis has
        nothing in common with what we have spoken, the recogniser has moved
        on to new speech without ever telling us the old utterance ended.
        Returning nothing there - on the reasoning that we cannot align it -
        swallows that sentence, and then every sentence after it.
        """
        if not text:
            return ""
        text_clean = text.strip()
        emitted = self._emitted.strip()

        if not emitted:
            # Only the previous utterance arriving twice can suppress this.
            if self._last_spoken and _same(text_clean, self._last_spoken):
                return ""
            return text_clean

        spoken = _words(emitted)
        words = text_clean.split()
        heard = _words(text_clean)

        if heard == spoken or _starts_with(spoken, heard):
            # Nothing new, or the recogniser retracted. We cannot unsay it.
            return ""

        if _starts_with(heard, spoken):
            return " ".join(words[len(spoken):]).strip()

        # Check if a suffix of what was already emitted overlaps with the start of heard
        # Handles cases where ASR dropped initial words on revision (e.g. "I'm in ground software" -> "in ground software technical manager")
        for k in range(min(len(spoken), len(heard)), 1, -1):
            if spoken[-k:] == heard[:k]:
                return " ".join(words[k:]).strip()

        shared = 0
        for mine, theirs in zip(spoken, heard):
            if mine != theirs:
                break
            shared += 1

        if shared >= 2:
            # A revision of the same speech: keep our version, take what is new.
            return " ".join(words[shared:]).strip()

        # Nothing in common. This is different speech, not a revision of ours.
        self._emitted = ""
        return text_clean
