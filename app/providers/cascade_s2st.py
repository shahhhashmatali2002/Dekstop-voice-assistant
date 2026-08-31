"""Speech-to-speech translation assembled from three cheaper parts.

Spec reference: §14.2 - the fallback for when the single-model path is
unavailable, "worse in every way except that it works".

    Urdu speech -> Deepgram nova-3 (ur) -> translator -> English TTS

It exists for a blunt commercial reason. The single-model path is OpenAI's
realtime translation model, billed per minute of audio, with no free tier. A
day of testing empties a small balance, and when it does the outgoing
direction stops mid-call - which is indistinguishable, from the outside, from
the software breaking. This path runs on recognition the project already pays
for, a translator with a free tier, and system TTS that costs nothing.

The important design decision is that this is NOT a new pipeline. It presents
exactly the interface the single-model provider presents - `open()` returning
something with `push_audio`, `audio_out`, `transcripts` and `aclose` - so the
outgoing pipeline, the barge-in handler, the jitter handling and the transcript
window are all unchanged and unaware. Choosing between the two is one line in
the factory, and adding credit later switches back with nothing to undo.

Two things are deliberately simple:

  * Segmentation is Deepgram's own `is_final`. The recogniser already decides
    where an utterance ended, and a second opinion on top of it was, in this
    codebase, the source of duplicated and half-spoken lines.
  * Nothing is translated twice. Interim hypotheses are shown in the
    transcript as they arrive but only finals are sent to the translator and
    to TTS, because a phrase spoken twice is far worse than a phrase heard
    half a second later.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time

import numpy as np

from app.audio import convert
from app.audio.formats import DTYPE
from app.providers.base import (
    AsrConfig,
    PcmChunk,
    ProviderError,
    ProviderMisconfigured,
    RollingContext,
    S2stConfig,
    S2stStream,
    SpeechTranslator,
    TranscriptDelta,
    TranslationUnit,
    VoiceId,
)

#: What the recogniser wants, regardless of what the pipeline hands us.
RATE_ASR = 16_000

#: The voice the client hears. An Edge voice, so it costs nothing; override
#: with ENGLISH_VOICE for a different accent.
DEFAULT_ENGLISH_VOICE = "en-US-AndrewNeural"

#: Bounded so a stalled consumer cannot grow memory without limit. Audio is
#: large: sixty chunks is already several seconds of speech.
AUDIO_QUEUE = 64
TEXT_QUEUE = 64


def english_voice(rate: int) -> VoiceId:
    return VoiceId(
        vendor="edge",
        name=os.environ.get("ENGLISH_VOICE") or DEFAULT_ENGLISH_VOICE,
        rate=rate,
    )


class CascadeS2stStream(S2stStream):
    """One session: audio in, translated audio and text out."""

    def __init__(self, asr_stream, translator, tts, *, cfg: S2stConfig,
                 terminology=None) -> None:
        self._asr = asr_stream
        self._translator = translator
        self._tts = tts
        self._cfg = cfg
        self._terminology = terminology
        self._context = RollingContext()

        self._audio: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE)
        self._text: asyncio.Queue = asyncio.Queue(maxsize=TEXT_QUEUE)
        self._closed = False
        self.failure: Exception | None = None

        #: Serialises translation and synthesis. Two utterances synthesised at
        #: once arrive interleaved at the sink, which is heard as two people
        #: talking over each other in one voice.
        self._speaking = asyncio.Lock()
        self._jobs: set[asyncio.Task] = set()
        self._last_push = 0.0
        self._worker = asyncio.create_task(self._read_hypotheses())
        self._heartbeat = asyncio.create_task(self._keep_alive())

    # ---------------------------------------------------------------- input

    async def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE)
        if self._cfg.rate != RATE_ASR:
            pcm = convert.resample(pcm, self._cfg.rate, RATE_ASR)
        self._last_push = time.monotonic()
        await self._asr.push_audio(pcm)

    #: Deepgram drops a stream after ten seconds with nothing on it. Half that
    #: leaves room for a slow event loop without wasting messages.
    KEEPALIVE_EVERY_S = 4.0

    async def _keep_alive(self) -> None:
        """Hold the recogniser's stream open across a pause in the call.

        Without this, ten quiet seconds - one person thinking, a document being
        opened, anything - kills the connection with NET-0001, and the next
        thing said is lost while the stream reconnects. The pause between
        sentences in a real meeting is routinely longer than ten seconds.
        """
        keepalive = getattr(self._asr, "keepalive", None)
        if keepalive is None:
            return
        while not self._closed:
            await asyncio.sleep(self.KEEPALIVE_EVERY_S / 2.0)
            if self._closed:
                return
            if time.monotonic() - self._last_push < self.KEEPALIVE_EVERY_S:
                continue
            try:
                await keepalive()
            except Exception:
                # The reader owns the connection's health and will report it.
                return

    # --------------------------------------------------------------- output

    async def audio_out(self):
        while True:
            item = await self._audio.get()
            if item is None:
                self._raise_failure()
                return
            yield item

    async def transcripts(self):
        while True:
            item = await self._text.get()
            if item is None:
                return
            yield item

    def _raise_failure(self) -> None:
        """Say why the stream ended, if it ended badly.

        The alternative - returning quietly - is what turns a dead provider
        into "the translator has silently stopped translating", which is the
        worst failure this component has.
        """
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure

    # ---------------------------------------------------------------- pumps

    async def _read_hypotheses(self) -> None:
        try:
            async for hypothesis in self._asr.hypotheses():
                text = (hypothesis.text or "").strip()
                if not text:
                    continue

                # NOT emitted: the Urdu we heard. This channel means "the
                # translation" to everything downstream - the outgoing
                # pipeline labels it [you] and shows it as what the client
                # hears. Putting interim source text here filled the window
                # with Urdu fragments that looked like the app translating
                # into the wrong language, when it had not translated yet at
                # all. The realtime model has the same two channels and only
                # ever sends its OUTPUT transcript here; this must match.
                if not hypothesis.is_final:
                    continue

                job = asyncio.create_task(self._speak_translation(text))
                self._jobs.add(job)
                job.add_done_callback(self._jobs.discard)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = exc
        finally:
            # The recogniser is finished, but utterances it already produced
            # may still be mid-translation or mid-synthesis. Closing the audio
            # stream here would cut off the last sentence of the call - and it
            # would do it silently, which is the worst kind. Wait for them.
            # Not when we are being torn down: aclose() cancels these on
            # purpose and awaiting them then would hang the shutdown.
            if not self._closed:
                for job in list(self._jobs):
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await job
            with contextlib.suppress(Exception):
                self._audio.put_nowait(None)
            with contextlib.suppress(Exception):
                self._text.put_nowait(None)

    async def _speak_translation(self, said: str) -> None:
        """Translate one finished utterance and synthesise it, in order."""
        async with self._speaking:
            if self._closed:
                return
            unit = TranslationUnit(
                text=said,
                source_language=self._cfg.source_language or "ur",
                target_language=self._cfg.target_language,
            )
            try:
                translated = await self._translator.translate(
                    unit, self._context, self._terminology
                )
            except ProviderError as exc:
                self.failure = exc
                with contextlib.suppress(Exception):
                    self._audio.put_nowait(None)
                return
            except Exception:
                return

            english = (translated.text or "").strip()
            if not english:
                return

            self._context.add("you", english)
            # Both halves, together: this path heard the Urdu and produced
            # the English, so the window can show them one under the other.
            await self._put(
                self._text,
                TranscriptDelta(text=english, is_final=True,
                                language=self._cfg.target_language,
                                source_text=said),
            )

            voice = english_voice(self._cfg.rate)
            try:
                async for chunk in self._tts.synthesize(english, voice):
                    if self._closed or chunk.pcm.size == 0:
                        continue
                    await self._put(self._audio, chunk)
            except Exception as exc:
                self.failure = exc
                with contextlib.suppress(Exception):
                    self._audio.put_nowait(None)

    @staticmethod
    async def _put(queue: asyncio.Queue, item) -> None:
        """Never block the producer for long, never silently lose audio.

        A full audio queue means the sink is not draining; waiting briefly is
        correct, and dropping the oldest after that is better than stalling
        recognition, which would put the whole session behind the speaker.
        """
        try:
            queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass
        try:
            await asyncio.wait_for(queue.put(item), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            with contextlib.suppress(Exception, asyncio.QueueEmpty):
                queue.get_nowait()
            with contextlib.suppress(Exception, asyncio.QueueFull):
                queue.put_nowait(item)

    # ---------------------------------------------------------------- close

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True

        for task in (self._worker, self._heartbeat):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for job in list(self._jobs):
            job.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await job
        with contextlib.suppress(Exception):
            await self._asr.aclose()
        with contextlib.suppress(Exception):
            self._audio.put_nowait(None)
        with contextlib.suppress(Exception):
            self._text.put_nowait(None)


class CascadeSpeechTranslator(SpeechTranslator):
    """The single-model provider's interface, served by three cheap parts."""

    name = "cascade-s2st"

    #: Unlike the realtime model, this path has no fixed list of output
    #: languages: whatever the translator can write and the TTS can speak.
    #: English is what this product sends to the client.
    output_languages = ("en",)

    def __init__(self, asr, translator, tts, *, terminology=None) -> None:
        self.asr = asr
        self.translator = translator
        self.tts = tts
        self.terminology = terminology

    async def open(self, cfg: S2stConfig) -> S2stStream:
        if self.asr is None or self.translator is None or self.tts is None:
            raise ProviderMisconfigured(
                "the cascade needs a recogniser, a translator and a voice"
            )
        asr_stream = await self.asr.open(
            AsrConfig(
                language=cfg.source_language or "ur",
                rate=RATE_ASR,
                interim_results=True,
            )
        )
        return CascadeS2stStream(
            asr_stream,
            self.translator,
            self.tts,
            cfg=cfg,
            terminology=self.terminology,
        )
