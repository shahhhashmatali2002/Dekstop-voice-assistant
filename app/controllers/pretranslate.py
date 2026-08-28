"""Pre-translated playback: Urdu on the video's own clock.

WHY THIS EXISTS
---------------
Live translation cannot be in sync with live video, and that is a property of
time rather than a shortfall in the engineering. To say a sentence in Urdu you
must first hear the English sentence; Urdu puts the verb where English does
not, so you often cannot start until the clause is nearly done; and then
recognition, translation and synthesis each take their turn. The picture,
meanwhile, arrives instantly. Three seconds of lag is the floor, and the
listener spends the whole video reconciling what they hear with what they saw
a moment ago.

A RECORDING HAS A FUTURE
------------------------
Its words already exist before you press play. So this mode does not stream at
all: it reads the whole transcript, translates every line with the rest of the
video for context, synthesises all of the Urdu, and only then starts playing -
each line fired at the timestamp it belongs to. The translation is finished
before playback begins, so there is nothing to wait for and nothing to be
behind. The Urdu lands where the English was.

It costs a wait before playback, and it cannot work on anything live: a Teams
call has no future to read ahead into. That is the whole trade.

WHAT IT DOES NOT DO
-------------------
It does not silence the original - routing does that, exactly as in listen
mode. Point the player at the cable and this speaks into your speakers.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from app.audio import convert
from app.audio.formats import DTYPE, RATE_S2ST
from app.models.terminology import Terminology
from app.providers.base import (
    ProviderError,
    RollingContext,
    TranslationUnit,
    VoiceId,
)

#: Sentence ends, English and Urdu.
SENTENCE_END = ".!?۔؟…"


class PrepareError(RuntimeError):
    """The video could not be prepared. The message is for a human."""


# ----------------------------------------------------------------------
# the transcript
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    """One timed piece of the original transcript."""

    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def video_id(url: str) -> str:
    """Pull the id out of any of the shapes YouTube hands people.

    Accepts a bare id too, so a user who pasted only that is not told their
    perfectly good input is wrong.
    """
    url = (url or "").strip()
    if not url:
        raise PrepareError("No video given.")
    patterns = (
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
        r"/live/([A-Za-z0-9_-]{11})",
    )
    for pattern in patterns:
        found = re.search(pattern, url)
        if found:
            return found.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise PrepareError(f"{url!r} does not look like a YouTube video or id.")


def fetch_cues(video: str, languages: Sequence[str] = ("en",)) -> list[Cue]:
    """The video's caption track, as timed cues.

    Auto-generated captions are fine here: their TIMING is what this mode
    depends on, and it comes from the same alignment YouTube uses to display
    them. Wording is improved by the translation step, which sees whole
    sentences.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:
        raise PrepareError(
            "pip install youtube-transcript-api"
        ) from exc

    try:
        api = YouTubeTranscriptApi()
        raw = api.fetch(video, languages=list(languages)).to_raw_data()
    except AttributeError:  # pragma: no cover - older releases
        raw = YouTubeTranscriptApi.get_transcript(video, languages=list(languages))
    except Exception as exc:
        raise PrepareError(
            f"Could not get captions for this video ({exc}). Videos with "
            "captions disabled cannot be pre-translated."
        ) from exc

    cues: list[Cue] = []
    for item in raw:
        text = (item.get("text") or "").replace("\n", " ").strip()
        if not text or text.startswith("["):  # [Music], [Applause]
            continue
        start = float(item.get("start", 0.0))
        cues.append(Cue(start, start + float(item.get("duration", 0.0)), text))
    if not cues:
        raise PrepareError("The caption track is empty.")
    return cues


def merge_sentences(
    cues: Iterable[Cue], *, max_seconds: float = 14.0, max_chars: int = 260
) -> list[Cue]:
    """Join caption fragments into whole sentences.

    Caption cues are cut to fit on screen, not at meaning: "skill. For
    example, your" is a cue. Translating those as-is produces the same
    disjointed Urdu the live path produces, for none of the latency reason -
    here we can see the whole sentence before deciding anything, so we should.

    A merged cue keeps the START of its first fragment, because that is the
    moment the speaker began saying it.
    """
    merged: list[Cue] = []
    start = end = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal start, end, parts
        if parts and start is not None:
            merged.append(Cue(start, end or start, " ".join(parts).strip()))
        start = end = None
        parts = []

    for cue in cues:
        if start is None:
            start = cue.start
        end = cue.end
        parts.append(cue.text)

        joined = " ".join(parts)
        long_enough = (end - start) >= max_seconds or len(joined) >= max_chars
        if joined.rstrip().endswith(tuple(SENTENCE_END)) or long_enough:
            flush()
    flush()
    return merged


# ----------------------------------------------------------------------
# the prepared script
# ----------------------------------------------------------------------


@dataclass
class Line:
    """One sentence, translated and synthesised, with the time it belongs at."""

    start: float
    english: str
    urdu: str = ""
    pcm: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=DTYPE))
    rate: int = RATE_S2ST

    @property
    def ready(self) -> bool:
        return self.pcm.size > 0


async def prepare(
    cues: Sequence[Cue],
    *,
    translator,
    tts,
    voice: VoiceId,
    terminology: Terminology | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Line]:
    """Translate and synthesise every line before anything is played.

    Sequential on purpose. The translator is given a rolling context so each
    sentence is rendered knowing what came before it - the advantage this mode
    has over the live path, and the reason its Urdu reads as continuous speech
    rather than a series of captions.
    """
    terms = terminology or Terminology()
    context = RollingContext()
    lines: list[Line] = []
    total = len(cues)

    for index, cue in enumerate(cues, start=1):
        line = Line(start=cue.start, english=cue.text)
        if on_progress:
            with contextlib.suppress(Exception):
                on_progress(index, total, cue.text)

        unit = TranslationUnit(
            text=cue.text, source_language="en", target_language="ur"
        )
        try:
            translated = await translator.translate(unit, context, terms)
        except ProviderError as exc:
            raise PrepareError(f"Translation failed on line {index}: {exc}") from exc

        line.urdu = translated.text.strip()
        if not line.urdu:
            continue
        context.add("speaker", cue.text)

        chunks: list[np.ndarray] = []
        rate = voice.rate
        try:
            async for chunk in tts.synthesize(line.urdu, voice):
                if chunk.pcm.size:
                    chunks.append(chunk.pcm)
                    rate = chunk.rate
        except ProviderError as exc:
            raise PrepareError(f"Synthesis failed on line {index}: {exc}") from exc

        if chunks:
            line.pcm = np.concatenate(chunks)
            line.rate = rate
            lines.append(line)

    if not lines:
        raise PrepareError("Nothing was translated; there is nothing to play.")
    return lines


def overruns(lines: Sequence[Line]) -> list[tuple[Line, float]]:
    """Lines whose Urdu is longer than the gap before the next one.

    Urdu is often wordier than English, so a sentence can run into the next
    one's slot. Reported rather than silently truncated: the honest options
    are to let it overlap slightly or to start it early, and both are the
    listener's call.
    """
    late: list[tuple[Line, float]] = []
    for line, following in zip(lines, list(lines[1:]) + [None]):
        spoken = line.pcm.size / line.rate
        room = (following.start - line.start) if following else spoken
        if spoken > room:
            late.append((line, spoken - room))
    return late


# ----------------------------------------------------------------------
# playback on the video's clock
# ----------------------------------------------------------------------


class Player:
    """Fires each line at its timestamp against a clock you control.

    The clock is injected, and so is sleeping, because the whole value of this
    class is its timing and timing that can only be observed in real time
    cannot be tested.
    """

    def __init__(
        self,
        sink,
        lines: Sequence[Line],
        *,
        offset: float = 0.0,
        clock: Callable[[], float] | None = None,
        sleep=None,
    ) -> None:
        self.sink = sink
        self.lines = sorted(lines, key=lambda line: line.start)
        #: Added to every timestamp. Positive speaks later, negative earlier -
        #: the correction for having started the video a moment before or
        #: after pressing play here.
        self.offset = offset
        self._clock = clock or asyncio.get_event_loop().time
        self._sleep = sleep or asyncio.sleep
        self.spoken = 0
        self.skipped = 0

    async def run(self, *, position: float = 0.0, on_line=None) -> None:
        """Play from `position` seconds into the video until the end."""
        started = self._clock() - position

        for line in self.lines:
            due = started + line.start + self.offset
            wait = due - self._clock()
            if wait < -2.0:
                # Long past. Speaking it now would put it against the wrong
                # picture, which is worse than not speaking it.
                self.skipped += 1
                continue
            if wait > 0:
                await self._sleep(wait)

            if on_line is not None:
                with contextlib.suppress(Exception):
                    on_line(line)
            await self._write(line)
            self.spoken += 1

    async def _write(self, line: Line) -> None:
        pcm = line.pcm
        if line.rate != self.sink.rate:
            pcm = convert.resample(pcm, line.rate, self.sink.rate)
        channels = max(1, getattr(self.sink, "channels", 1))
        if channels > 1:
            pcm = convert.to_channels(pcm, channels)

        ring = getattr(self.sink, "ring", None)
        if ring is None:
            self.sink.write(pcm)
            return
        limit = max(1, ring.capacity // 2)
        for start in range(0, pcm.size, limit):
            piece = pcm[start : start + limit]
            while ring.free < piece.size:
                await self._sleep(0.01)
            self.sink.write(piece)
