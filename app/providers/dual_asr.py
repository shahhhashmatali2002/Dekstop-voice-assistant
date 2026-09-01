"""Two recognisers on one microphone, and a rule for believing one of them.

Spec reference: §8.3, §7.7 - and a defect this file exists to fix.

THE PROBLEM
-----------
The outgoing direction opened the recogniser with `language="ur"` and never
asked again. That is correct only for somebody who speaks nothing but Urdu.
Hand a monolingual Urdu model an English sentence and it does not decline: it
maps English phonemes onto the nearest Urdu-model output it has, confidently.
Reported from a live call:

    said "I am John Wick"              heard "John Bek"
    said "I work at Macrosoft Pakistan" heard "Microsoft"

Neither is a small error. A name is the one token in a sentence a listener
cannot reconstruct from context, so a mangled name is not "slightly worse
transcription" - it is a wrong fact delivered in the speaker's own voice.

WHY NOT JUST ONE MULTILINGUAL MODEL
-----------------------------------
Because there is not one that covers this pair. Nova-3's code-switching mode
(`language=multi`) is English, Spanish, French, German, Hindi, Russian,
Portuguese, Japanese, Italian and Dutch - Urdu is not in it. Urdu exists on
Nova-3 only as a MONOLINGUAL option. So a single stream is a choice between
hearing Urdu and hearing English, and this speaker does both, in the same
call and sometimes in the same sentence.

WHAT THIS DOES
--------------
Opens one stream per language, sends every frame to all of them, and decides
per utterance which one to believe. Recognition is billed per minute of audio
and is the cheapest component in the pipeline by an order of magnitude, so
listening twice costs less than a single mangled client name.

The decision is deliberately a small pure function - `choose` - so it can be
argued with and unit-tested without a network.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re

import numpy as np

from app.providers.base import (
    AsrConfig,
    AsrStream,
    SpeechToText,
    TranscriptDelta,
)

#: Perso-Arabic. Urdu is written in it; English is not. The single most
#: reliable signal available here, and it costs nothing: if the Urdu stream
#: came back without it, the Urdu stream did not hear Urdu.
ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

#: How long to wait, after the first recogniser finalises an utterance, for
#: the other one to finalise the same utterance. They share endpointing
#: settings and the same audio, so they land within a few tens of
#: milliseconds; this is slack, not a budget being spent.
SETTLE_MS = 450.0

#: Added to English's score when the two are close. Not a preference for
#: English - a correction for an asymmetry in the cost of being wrong. Names,
#: companies and technical terms are spoken in English by this speaker even
#: mid-Urdu, and the Urdu model renders them into script from which the
#: original cannot be recovered by anything downstream. An English stream that
#: is wrong about an Urdu clause still leaves the translator something to work
#: with; the reverse destroys the token that mattered.
ENGLISH_BIAS = float(os.environ.get("ASR_ENGLISH_BIAS", "0.05"))


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return len(ARABIC.findall("".join(letters))) / len(letters)


def score(language: str, text: str, confidence: float | None) -> float:
    """How much to believe one recogniser's version of one utterance.

    Three things, in order of how much they matter:

    1. The recogniser's own confidence, which is the only number both models
       produce and therefore the only one they can be compared on.
    2. How much was heard. A model that returns one word at 0.95 has not
       out-performed one that returned the whole sentence at 0.85 - it has
       failed differently. Short outputs are discounted rather than trusted.
    3. Script. An Urdu result in Latin letters is not an Urdu result.
    """
    words = len((text or "").split())
    if words == 0:
        return 0.0
    conf = 0.5 if confidence is None else max(0.0, min(1.0, confidence))
    length_weight = 0.55 + 0.45 * min(words / 6.0, 1.0)
    value = conf * length_weight

    if language.startswith("ur"):
        # The Urdu model handed English audio produces Latin text or a thin
        # scattering of script. Either way it is guessing, and the English
        # stream is listening to the same second of audio with a model that
        # is not.
        ratio = arabic_ratio(text)
        if ratio < 0.30:
            value *= 0.35
        elif ratio < 0.60:
            value *= 0.80
    elif language.startswith("en"):
        value += ENGLISH_BIAS

    return value


def choose(candidates: dict[str, tuple[str, float | None]]) -> tuple[str, str, float]:
    """Pick a language and its transcript. Returns (language, text, score).

    `candidates` maps language code to (text, confidence). Empty transcripts
    are ignored; an empty input returns ("", "", 0.0) and the caller emits
    nothing, which is the right answer for a cough.
    """
    best_lang, best_text, best_score = "", "", 0.0
    for language, (text, confidence) in candidates.items():
        text = (text or "").strip()
        if not text:
            continue
        s = score(language, text, confidence)
        if s > best_score:
            best_lang, best_text, best_score = language, text, s
    return best_lang, best_text, best_score


# ----------------------------------------------------------------------


class DualLanguageAsrStream(AsrStream):
    """One session that is really N sessions with a referee."""

    def __init__(
        self,
        streams: dict[str, AsrStream],
        *,
        primary: str = "en",
        settle_ms: float = SETTLE_MS,
    ) -> None:
        self._streams = streams
        self._settle = settle_ms / 1000.0
        self._leader = primary if primary in streams else next(iter(streams))
        self._out: asyncio.Queue[TranscriptDelta | None] = asyncio.Queue(maxsize=128)
        self._pending: dict[str, list[tuple[str, float | None]]] = {
            lang: [] for lang in streams
        }
        self._final_seen = asyncio.Event()
        self._lock = asyncio.Lock()
        self._closed = False
        self._live = len(streams)
        self._readers = [
            asyncio.create_task(self._read(lang, stream), name=f"asr-{lang}")
            for lang, stream in streams.items()
        ]
        self._referee = asyncio.create_task(self._decide_loop(), name="asr-referee")

    # ---------------------------------------------------------------- input

    async def push_audio(self, pcm: np.ndarray) -> None:
        """Every recogniser gets every frame.

        One of them failing is survivable and is not reported: the other is
        still listening, and a call that keeps working in one language beats a
        call that stops. ALL of them failing is the outgoing pipeline's
        reconnect signal, so that one is raised - swallowing it is how a dead
        connection becomes "the app quietly stopped translating".
        """
        if self._closed:
            return
        results = await asyncio.gather(
            *(s.push_audio(pcm) for s in self._streams.values()),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures and len(failures) == len(results):
            raise failures[0]

    async def keepalive(self) -> None:
        if self._closed:
            return
        await asyncio.gather(
            *(
                s.keepalive()
                for s in self._streams.values()
                if hasattr(s, "keepalive")
            ),
            return_exceptions=True,
        )

    # --------------------------------------------------------------- output

    async def hypotheses(self):
        while True:
            item = await self._out.get()
            if item is None:
                return
            yield item

    # ---------------------------------------------------------------- pumps

    async def _read(self, language: str, stream: AsrStream) -> None:
        try:
            async for delta in stream.hypotheses():
                text = (delta.text or "").strip()
                if not text:
                    continue
                if delta.is_final:
                    async with self._lock:
                        self._pending[language].append((text, delta.confidence))
                    self._final_seen.set()
                elif language == self._leader:
                    # Something has to reach the transcript window while the
                    # sentence is still being spoken, and an interim from the
                    # language that won last time is the least-wrong guess
                    # available before the referee has anything to referee.
                    with contextlib.suppress(asyncio.QueueFull):
                        self._out.put_nowait(
                            TranscriptDelta(
                                text=text,
                                is_final=False,
                                language=language,
                                confidence=delta.confidence,
                            )
                        )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            self._live -= 1
            if self._live <= 0:
                self._final_seen.set()

    async def _decide_loop(self) -> None:
        try:
            while not self._closed:
                await self._final_seen.wait()
                if self._closed:
                    return
                # Let the other recognisers land on the same utterance before
                # comparing. Without this the faster stream always "wins" by
                # being the only candidate.
                await asyncio.sleep(self._settle)
                async with self._lock:
                    candidates = {
                        lang: (" ".join(t for t, _ in parts), _mean_conf(parts))
                        for lang, parts in self._pending.items()
                        if parts
                    }
                    for parts in self._pending.values():
                        parts.clear()
                    self._final_seen.clear()

                if not candidates:
                    if self._live <= 0:
                        return
                    continue

                language, text, _ = choose(candidates)
                if not text:
                    continue
                self._leader = language or self._leader
                await self._emit(
                    TranscriptDelta(
                        text=text,
                        is_final=True,
                        language=language,
                        confidence=candidates.get(language, ("", None))[1],
                    )
                )
                if self._live <= 0:
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._out.put_nowait(None)

    async def _emit(self, delta: TranscriptDelta) -> None:
        try:
            self._out.put_nowait(delta)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._out.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._out.put_nowait(delta)

    # ---------------------------------------------------------------- close

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._final_seen.set()
        for task in [*self._readers, self._referee]:
            task.cancel()
        for task in [*self._readers, self._referee]:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        await asyncio.gather(
            *(s.aclose() for s in self._streams.values()), return_exceptions=True
        )
        with contextlib.suppress(Exception):
            self._out.put_nowait(None)


def _mean_conf(parts: list[tuple[str, float | None]]) -> float | None:
    known = [c for _, c in parts if c is not None]
    if not known:
        return None
    return sum(known) / len(known)


# ----------------------------------------------------------------------


class DualLanguageSpeechToText(SpeechToText):
    """`SpeechToText` that listens in several languages at once.

    Presents exactly the single-recogniser interface, so the cascade, the
    outgoing pipeline and the transcript window are unchanged and unaware -
    the same trick `cascade_s2st` plays on the realtime model.
    """

    name = "dual-language-asr"

    def __init__(
        self,
        asr: SpeechToText,
        languages: tuple[str, ...] = ("en", "ur"),
        *,
        primary: str = "en",
    ) -> None:
        self.asr = asr
        self.languages = tuple(dict.fromkeys(languages))
        self.primary = primary

    async def open(self, cfg: AsrConfig) -> AsrStream:
        if len(self.languages) < 2:
            return await self.asr.open(
                AsrConfig(**{**cfg.__dict__, "language": self.languages[0]})
            )
        streams: dict[str, AsrStream] = {}
        try:
            for language in self.languages:
                streams[language] = await self.asr.open(
                    AsrConfig(
                        language=language,
                        rate=cfg.rate,
                        interim_results=cfg.interim_results,
                        # Keyterm prompting is supported on Nova-3 for both
                        # English and Urdu, and the terms that get mangled -
                        # names, employers, products - are the same list in
                        # both. Send it to both.
                        keyterms=list(cfg.keyterms),
                        model=cfg.model,
                        region=cfg.region,
                    )
                )
        except Exception:
            await asyncio.gather(
                *(s.aclose() for s in streams.values()), return_exceptions=True
            )
            raise
        return DualLanguageAsrStream(streams, primary=self.primary)
