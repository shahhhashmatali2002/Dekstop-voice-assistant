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
from pathlib import Path
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


def _caption_advice(exc: Exception) -> str:
    """Turn the library's generic complaint into the actual cause.

    Its default message leads with "the video is no longer available", which
    is almost never true for a video the user is watching in another window,
    and it buries the causes that actually happen.
    """
    from youtube_transcript_api import _errors as errors

    kind = type(exc).__name__
    known = {
        "TranscriptsDisabled":
            "This video has captions turned off, so there is no transcript to "
            "read.\n      Use --transcribe to have Deepgram transcribe the "
            "audio instead.",
        "NoTranscriptFound":
            "No caption track in the language asked for. Try --lang with one "
            "of the\n      languages listed above, or --transcribe.",
        "PoTokenRequired":
            "YouTube is demanding a browser token for this request - it will "
            "not serve\n      captions to a plain HTTP client. Use "
            "--transcribe, which does not go\n      through the caption API "
            "at all.",
        "RequestBlocked":
            "YouTube is blocking this request (it does this to non-browser "
            "clients).\n      Use --transcribe.",
        "IpBlocked":
            "YouTube is blocking this IP address. Use --transcribe.",
        "AgeRestricted":
            "The video is age restricted, so captions need a signed-in "
            "session.\n      Use --transcribe.",
        "VideoUnplayable":
            "YouTube says the video is unplayable for this client. If it plays "
            "in your\n      browser, this is a bot check rather than the "
            "video - use --transcribe.",
        "VideoUnavailable":
            "YouTube says the video is unavailable. If it plays in your "
            "browser, that is\n      a bot check rather than the truth - use "
            "--transcribe.",
        "YouTubeDataUnparsable":
            "YouTube returned a page this library does not understand, which "
            "usually\n      means a consent or bot-check page. Use "
            "--transcribe.",
    }
    if kind in known:
        return known[kind]
    del errors  # imported only to make the dependency explicit
    return (
        f"{kind}: {exc}\n      Use --transcribe to skip the caption API and "
        "transcribe the audio."
    )


def caption_languages(video: str) -> list[str]:
    """What caption tracks exist, for the error message. Never raises."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        found = []
        for transcript in YouTubeTranscriptApi().list(video):
            tag = " (auto)" if transcript.is_generated else ""
            found.append(f"{transcript.language_code}{tag}")
        return found
    except Exception:
        return []


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
        available = caption_languages(video)
        listing = (
            f"\n      Caption tracks on this video: {', '.join(available)}"
            if available else ""
        )
        raise PrepareError(
            f"Captions could not be read.{listing}\n      "
            + _caption_advice(exc)
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
# transcribing the audio ourselves
# ----------------------------------------------------------------------
#
# The caption API is the fast path and it is not dependable: YouTube serves
# bot checks to non-browser clients, some videos have captions disabled, and
# none of that is under our control. Transcribing the audio is slower and
# costs Deepgram credit, but it works on any video - and on a local file,
# which the caption path can never do.
#
# Deepgram's pre-recorded endpoint with `utterances=true` returns exactly the
# structure this mode needs: sentence-level segments, each with a start time.
# That is better timing than a caption track, which is cut to fit a screen.

DEEPGRAM_PRERECORDED = "https://api.deepgram.com/v1/listen"


def download_audio(url: str, into: Path, *, on_progress=None) -> Path:
    """Fetch a video's audio. Returns the file it wrote.

    Uses yt-dlp as a LIBRARY rather than a command. On Windows, pip puts
    console scripts in a per-user Scripts directory that is routinely not on
    PATH - pip says so in a warning nobody reads - so `yt-dlp` the command is
    missing on a machine where `import yt_dlp` works perfectly. Shelling out
    turns a working install into "yt-dlp is needed: pip install yt-dlp",
    which is advice the user has already followed.
    """
    into.mkdir(parents=True, exist_ok=True)
    template = str(into / "audio.%(ext)s")

    try:
        import yt_dlp
    except ImportError:
        return _download_audio_via_command(url, into, template)

    options = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "consoletitle": False,
    }
    if on_progress is not None:
        options["progress_hooks"] = [on_progress]

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            written = Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise PrepareError(f"yt-dlp could not download the audio: {exc}") from exc

    if written.exists():
        return written
    # yt-dlp reports the pre-merge name in some formats; take what is there.
    found = sorted(into.glob("audio.*"))
    if not found:
        raise PrepareError("yt-dlp reported success but wrote no file.")
    return found[0]


def _download_audio_via_command(url: str, into: Path, template: str) -> Path:
    """Last resort: the command, if the module is genuinely absent."""
    import shutil
    import subprocess
    import sys

    command = None
    if shutil.which("yt-dlp"):
        command = ["yt-dlp"]
    else:
        # `python -m yt_dlp` works whenever the module is importable by the
        # interpreter running us, PATH or no PATH.
        command = [sys.executable, "-m", "yt_dlp"]

    try:
        subprocess.run(
            command + ["-f", "bestaudio/best", "--no-playlist", "--quiet",
                       "-o", template, url],
            check=True, capture_output=True, timeout=1800,
        )
    except FileNotFoundError:
        raise PrepareError(
            "yt-dlp is needed to get the audio of a YouTube video.\n"
            "      pip install yt-dlp"
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
        if "No module named" in detail:
            raise PrepareError(
                "yt-dlp is needed to get the audio of a YouTube video.\n"
                "      pip install yt-dlp"
            )
        raise PrepareError(f"yt-dlp could not download the audio: {detail}")
    except subprocess.TimeoutExpired:
        raise PrepareError("yt-dlp timed out downloading the audio.")

    found = sorted(into.glob("audio.*"))
    if not found:
        raise PrepareError("yt-dlp reported success but wrote no file.")
    return found[0]


def transcribe(audio: Path, api_key: str, *, model: str = "nova-3") -> list[Cue]:
    """Deepgram pre-recorded transcription, as timed sentences."""
    import json

    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise PrepareError("pip install requests") from exc

    if not api_key:
        raise PrepareError("DEEPGRAM_API_KEY is not set, so audio cannot be "
                           "transcribed.")

    params = {
        "model": model,
        "smart_format": "true",
        "utterances": "true",
        "language": "en",
    }
    try:
        with audio.open("rb") as handle:
            response = requests.post(
                DEEPGRAM_PRERECORDED,
                params=params,
                headers={
                    "Authorization": f"Token {api_key}",
                    "Content-Type": "application/octet-stream",
                },
                data=handle,
                timeout=1800,
            )
    except Exception as exc:
        raise PrepareError(f"Deepgram could not be reached: {exc}") from exc

    if response.status_code != 200:
        raise PrepareError(
            f"Deepgram refused the audio ({response.status_code}): "
            f"{response.text[:300]}"
        )
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise PrepareError("Deepgram returned something that is not JSON.") from exc

    return cues_from_deepgram(payload)


def cues_from_deepgram(payload: dict) -> list[Cue]:
    """Pull timed sentences out of a Deepgram pre-recorded response.

    Kept separate from the HTTP call so the shape of the response is tested
    without a network, which is where response-parsing bugs actually live.
    """
    utterances = (payload.get("results") or {}).get("utterances") or []
    cues = [
        Cue(float(u.get("start", 0.0)), float(u.get("end", 0.0)),
            (u.get("transcript") or "").strip())
        for u in utterances
        if (u.get("transcript") or "").strip()
    ]
    if cues:
        return cues

    # No utterances (the parameter was not honoured): fall back to words,
    # which every response has, and group them into sentences ourselves.
    channels = (payload.get("results") or {}).get("channels") or []
    words = []
    for channel in channels:
        for alternative in channel.get("alternatives") or []:
            words.extend(alternative.get("words") or [])
    if not words:
        raise PrepareError("Deepgram returned no transcript for this audio.")

    grouped: list[Cue] = []
    start = None
    end = 0.0
    parts: list[str] = []
    for word in words:
        text = word.get("punctuated_word") or word.get("word") or ""
        if start is None:
            start = float(word.get("start", 0.0))
        end = float(word.get("end", end))
        parts.append(text)
        if text.rstrip().endswith(tuple(SENTENCE_END)):
            grouped.append(Cue(start, end, " ".join(parts).strip()))
            start, parts = None, []
    if parts and start is not None:
        grouped.append(Cue(start, end, " ".join(parts).strip()))
    return grouped


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


async def prepare_into(
    script: "Script",
    cues: Sequence[Cue],
    *,
    translator,
    tts,
    voice: VoiceId,
    terminology: Terminology | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    converter=None,
    reference=None,
) -> None:
    """Fill `script` line by line, so playback can begin on the first one.

    Runs as a background task beside the player. Preparing one line takes a
    second or two and lines are spaced seconds apart, so this stays ahead of
    playback comfortably - and if it ever does fall behind, the player waits
    for the line rather than skipping it.
    """
    try:
        async for line in prepare_lines(
            cues, translator=translator, tts=tts, voice=voice,
            terminology=terminology, on_progress=on_progress,
            converter=converter, reference=reference,
        ):
            script.add(line)
    except Exception as exc:
        script.close(exc)
        return
    script.close()


async def prepare_lines(
    cues: Sequence[Cue],
    *,
    translator,
    tts,
    voice: VoiceId,
    terminology: Terminology | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    converter=None,
    reference=None,
):
    """Translate and synthesise each cue, yielding lines as they become ready.

    `converter` and `reference` are the optional voice-cloning path: the Urdu
    is synthesised by the normal voice and then converted to sound like
    `reference`, which for a video is a clip of the video's own audio. It
    belongs here rather than on the live path because conversion costs real
    compute per line and this is the one mode where that is paid before
    playback rather than during it.
    """
    terms = terminology or Terminology()
    context = RollingContext()
    total = len(cues)

    for index, cue in enumerate(cues, start=1):
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

        urdu = translated.text.strip()
        if not urdu:
            continue
        context.add("speaker", cue.text)

        chunks: list[np.ndarray] = []
        rate = voice.rate
        try:
            async for chunk in tts.synthesize(urdu, voice):
                if chunk.pcm.size:
                    chunks.append(chunk.pcm)
                    rate = chunk.rate
        except ProviderError as exc:
            raise PrepareError(f"Synthesis failed on line {index}: {exc}") from exc

        if not chunks:
            continue

        pcm = np.concatenate(chunks)
        if converter is not None and reference is not None:
            try:
                pcm, rate = converter.convert(pcm, rate, reference)
            except ProviderError as exc:
                # The stock voice saying it is far better than nothing saying
                # it. Reported once and then dropped for the rest of the run,
                # because a per-line warning on a 600-line video is noise.
                raise PrepareError(
                    f"Voice conversion failed on line {index}: {exc}"
                ) from exc

        yield Line(
            start=cue.start, english=cue.text, urdu=urdu,
            pcm=pcm, rate=rate,
        )


async def prepare(
    cues: Sequence[Cue],
    *,
    translator,
    tts,
    voice: VoiceId,
    terminology: Terminology | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    converter=None,
    reference=None,
) -> list[Line]:
    """Every line, before anything is played. Used where a wait is fine."""
    lines = [
        line
        async for line in prepare_lines(
            cues, translator=translator, tts=tts, voice=voice,
            terminology=terminology, on_progress=on_progress,
            converter=converter, reference=reference,
        )
    ]
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
# the video's clock
# ----------------------------------------------------------------------


class PausableClock:
    """Where the video is, in seconds - not where the wall clock is.

    Playback used to be scheduled against wall time from the moment you
    pressed Enter, which assumes the video never stops. It does stop: you
    pause it, it buffers, you take a call. Every one of those left the Urdu
    running on against a frozen picture, and it never recovered, because
    nothing was measuring the video at all.
    """

    def __init__(self, position: float = 0.0, clock: Callable[[], float] | None = None):
        import time

        self._clock = clock or time.monotonic
        self._position = position
        self._since: float | None = None

    @property
    def running(self) -> bool:
        return self._since is not None

    def position(self) -> float:
        if self._since is None:
            return self._position
        return self._position + (self._clock() - self._since)

    def start(self) -> None:
        if self._since is None:
            self._since = self._clock()

    def pause(self) -> None:
        if self._since is not None:
            self._position = self.position()
            self._since = None

    def seek(self, position: float) -> None:
        self._position = position
        if self._since is not None:
            self._since = self._clock()


class SourceMonitor:
    """Follows the video by watching whether its audio is still arriving.

    WHY ARRIVAL AND NOT LOUDNESS
    ----------------------------
    A paused video does not go quiet, it goes ABSENT: WASAPI stops delivering
    callbacks for an endpoint nothing is rendering into, so no frames arrive
    at all. A quiet passage in a PLAYING video still delivers frames, they are
    just near zero. Measuring amplitude would confuse the two and pause the
    clock every time the speaker drew breath, which would put the clock
    permanently behind the picture.

    So this needs no button and no keypress: it starts the clock when audio
    first appears and holds it the moment audio stops.
    """

    def __init__(
        self,
        source,
        clock: PausableClock,
        *,
        idle_after: float = 0.35,
        interval: float = 0.05,
        on_change: Callable[[bool], None] | None = None,
        sleep=None,
    ) -> None:
        self.source = source
        self.clock = clock
        self.idle_after = idle_after
        self.interval = interval
        self.on_change = on_change
        self._sleep = sleep or asyncio.sleep
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def poll(self) -> int:
        """One look at the ring. Returns how many samples had arrived."""
        ring = getattr(self.source, "ring", None)
        if ring is None:
            return 0
        return int(ring.read_available().size)

    async def run(self) -> None:
        idle = 0.0
        while not self._stop:
            if self.poll():
                idle = 0.0
                if not self.clock.running:
                    self.clock.start()
                    self._announce(True)
            else:
                idle += self.interval
                if idle >= self.idle_after and self.clock.running:
                    self.clock.pause()
                    self._announce(False)
            await self._sleep(self.interval)

    def _announce(self, playing: bool) -> None:
        if self.on_change is not None:
            with contextlib.suppress(Exception):
                self.on_change(playing)


# ----------------------------------------------------------------------
# the script, filled in while it is being played
# ----------------------------------------------------------------------


class Script:
    """Lines in video order, appended as they become ready.

    Preparing a whole video before speaking a word means a wait proportional
    to its length - and if the viewer starts the video during that wait, the
    first line is spoken a minute after the moment it belonged to. Playback
    only ever needs the NEXT line, and preparing one takes a second or two
    while lines are spaced seconds apart, so preparation stays ahead easily.
    """

    def __init__(self) -> None:
        self.lines: list[Line] = []
        self.finished = False
        self.error: Exception | None = None

    def add(self, line: Line) -> None:
        self.lines.append(line)

    def close(self, error: Exception | None = None) -> None:
        self.error = error
        self.finished = True

    @property
    def ready(self) -> int:
        return len(self.lines)


# ----------------------------------------------------------------------
# playback
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
        late_grace: float = 2.0,
        clock: Callable[[], float] | None = None,
        sleep=None,
    ) -> None:
        self.sink = sink
        self.lines = sorted(lines, key=lambda line: line.start)
        #: Added to every timestamp. Positive speaks later, negative earlier -
        #: the correction for having started the video a moment before or
        #: after pressing play here.
        self.offset = offset
        #: How far past a line's moment it may still be spoken. Beyond this
        #: the picture has moved on and silence is the better answer.
        self.late_grace = late_grace
        self._clock = clock
        self._sleep = sleep or asyncio.sleep
        self.spoken = 0
        self.skipped = 0

    async def run(self, *, position: float = 0.0, on_line=None) -> None:
        """Play a finished list against a clock that never stops.

        Kept for callers that have every line in hand and nothing to follow.
        """
        script = Script()
        for line in self.lines:
            script.add(line)
        script.close()
        clock = PausableClock(position, clock=self._clock)
        clock.start()
        await self.follow(clock, script, on_line=on_line)

    async def follow(self, clock: PausableClock, script: Script, *, on_line=None):
        """Play `script` against `clock`, which may pause, resume or seek.

        Polled rather than scheduled. Sleeping until a line's wall-clock due
        time assumes the video keeps playing for exactly that long; asking the
        clock where the video is, over and over, makes a pause simply mean the
        answer stops changing - and a resume means it starts again, with no
        state to unwind.
        """
        index = 0
        while True:
            if index >= len(script.lines):
                if script.finished:
                    return
                await self._sleep(0.05)  # preparation has not caught up yet
                continue

            line = script.lines[index]
            due = line.start + self.offset
            position = clock.position()

            if position < due:
                if not clock.running:
                    await self._sleep(0.05)  # the video is paused; so are we
                    continue
                await self._sleep(min(0.1, due - position))
                continue

            index += 1
            if position - due > self.late_grace:
                # Long past. Speaking it now would put it against the wrong
                # picture, which is worse than not speaking it.
                self.skipped += 1
                continue

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
