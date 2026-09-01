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

#: The voice the client hears. An Edge voice by default (costs nothing), or
#: ElevenLabs voice ID when configured.
DEFAULT_ENGLISH_VOICE = "en-US-BrianNeural"
ELEVENLABS_DEFAULT_VOICE = "pNInz6obpgDQGcFmaJgB"  # Adam


def english_voice(rate: int, tts_vendor: str = "") -> VoiceId:
    if tts_vendor.startswith("eleven") or tts_vendor.startswith("fallback"):
        name = (
            os.environ.get("ELEVEN_EN_VOICE_ID")
            or os.environ.get("ELEVENLABS_VOICE_ID")
            or os.environ.get("ENGLISH_VOICE")
            # The known-working default, restored. Resolving a voice from the
            # account instead meant a network round trip inside the first
            # utterance of every call, which is latency in the one place this
            # pipeline cannot afford it.
            or ELEVENLABS_DEFAULT_VOICE
        )
        return VoiceId(vendor="elevenlabs", name=name, rate=rate, language="en")
    name = os.environ.get("ENGLISH_VOICE") or DEFAULT_ENGLISH_VOICE
    return VoiceId(
        vendor="edge",
        name=name,
        rate=rate,
        # Without this the English the client hears goes through the Urdu text
        # preparation and arrives as Urdu script inside an English voice.
        language="en",
    )


#: Bounded so a stalled consumer cannot grow memory without limit. Audio is
#: large: sixty chunks is already several seconds of speech.
AUDIO_QUEUE = 64
TEXT_QUEUE = 64


class CascadeS2stStream(S2stStream):
    """One session: audio in, translated audio and text out."""

    def __init__(self, asr_stream, translator, tts, *, cfg: S2stConfig,
                 terminology=None, reopen=None) -> None:
        self._asr = asr_stream
        self._translator = translator
        self._tts = tts
        self._cfg = cfg
        self._terminology = terminology
        #: Dials a fresh recogniser stream with the session's own settings.
        #: Recovery lives INSIDE the session because the alternative was
        #: watched happening: the recogniser hangs up (NET-0001 after a
        #: network stall), the reader ends quietly, and the death is only
        #: discovered when a push fails - ten seconds of speech later.
        self._reopen = reopen
        self._context = RollingContext()

        self._audio: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE)
        self._text: asyncio.Queue = asyncio.Queue(maxsize=TEXT_QUEUE)
        self._closed = False
        self.failure: Exception | None = None

        #: Two locks, not one, and the split is the pipeline. `_speaking`
        #: serialises synthesis - two utterances synthesised at once arrive
        #: interleaved at the sink, heard as two people talking over each
        #: other in one voice. `_translating` serialises the translator so
        #: the rolling context stays in spoken order. Holding ONE lock across
        #: both stages (as this originally did) meant utterance N+1's
        #: translation could not start until N had finished SYNTHESISING,
        #: so every utterance paid its predecessor's whole tail - during
        #: fluent speech, seconds per utterance, compounding.
        self._speaking = asyncio.Lock()
        self._translating = asyncio.Lock()
        #: Consecutive translation failures. One slow or failed round trip
        #: skips ONE utterance; only a run of them means the translator is
        #: actually down. Watched happening: a single translation past its
        #: deadline used to kill the whole outgoing direction mid-call.
        self._translate_failures = 0
        #: Clause-level segmentation over the recogniser's INTERIMS, exactly
        #: as the incoming pipeline does it. The original design here trusted
        #: Deepgram's is_final alone, and its chunk boundaries are arbitrary:
        #: fluent speech was cut into "I have recently" / "done graduation
        #: in AI" and each fragment translated and spoken as though it were
        #: a sentence. The segmenter's duplication bugs that once justified
        #: trusting is_final have since been fixed where the incoming
        #: pipeline lives on it. CASCADE_SEGMENTER=0 restores the old
        #: behaviour for A/B testing on a live call.
        self._segmenter = None
        if (os.environ.get("CASCADE_SEGMENTER") or "1").strip().lower() not in (
            "0", "off", "false", "no",
        ):
            from app.controllers.segmenter import Segmenter

            self._segmenter = Segmenter(
                min_hypotheses=2,
                stable_ms=250.0,
                max_words=14,
                soft_min_words=6,
            )
        self._jobs: set[asyncio.Task] = set()
        self._last_push = 0.0
        self._pushed_ms = 0.0
        self._worker = asyncio.create_task(self._read_hypotheses())
        self._heartbeat = asyncio.create_task(self._keep_alive())

    # ---------------------------------------------------------------- input

    async def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed or self._asr is None:
            # Mid-reopen. Dropping a frame here is the whole cost of the
            # reconnect; raising instead would tear down the pipeline the
            # reopen is about to save.
            return
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE)
        if self._cfg.rate != RATE_ASR:
            pcm = convert.resample(pcm, self._cfg.rate, RATE_ASR)
        self._last_push = time.monotonic()
        self._pushed_ms += len(pcm) * 1000.0 / RATE_ASR
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
        while not self._closed:
            await asyncio.sleep(self.KEEPALIVE_EVERY_S / 2.0)
            if self._closed:
                return
            if time.monotonic() - self._last_push < self.KEEPALIVE_EVERY_S:
                continue
            # Looked up per beat, not once: after an in-place reopen the
            # stream object is a different one, and a keepalive bound at
            # start-up would be dutifully keeping a corpse warm.
            keepalive = getattr(self._asr, "keepalive", None)
            if keepalive is None:
                continue
            try:
                await keepalive()
            except Exception:
                # The reader owns the connection's health and will report it.
                continue

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

    #: Recogniser accounting, printed every ten seconds. The question this
    #: answers is the one the ONSET/OFFSET log could not: when eleven speech
    #: events produce one transcript, is the recogniser being sent nothing, or
    #: being sent audio and returning nothing? Those have opposite fixes and
    #: look identical from outside.
    REPORT_EVERY_S = 10.0

    #: Consecutive failed translations before the outgoing direction is
    #: declared dead rather than merely unlucky.
    TRANSLATE_FAILURES_FATAL = 3

    #: Consecutive failed redials before the session is declared dead. A
    #: dropped socket is ordinary - a network stall, a provider node
    #: restarting - and ending the call for one is the worse answer.
    RECONNECT_ATTEMPTS = 3

    async def _reopen_asr(self, attempt: int) -> bool:
        """Dial the recogniser again, in place. False when it will not answer."""
        if self._reopen is None:
            return False
        await asyncio.sleep(min(2.0, 0.3 * attempt))
        old_stream, self._asr = self._asr, None
        if old_stream is not None:
            with contextlib.suppress(Exception):
                await old_stream.aclose()
        try:
            self._asr = await self._reopen()
        except Exception:
            return False
        print(
            "[DEBUG ASR] recogniser stream reopened after a drop - a few "
            "words around the gap may be lost",
            flush=True,
        )
        return True

    async def _read_hypotheses(self) -> None:
        import os

        verbose = (os.environ.get("DEBUG_ASR") or "").lower() in ("1", "true", "on")
        interims = finals = 0
        last_report = time.monotonic()
        failures = 0
        try:
          while not self._closed:
            broke: Exception | None = None
            try:
              async for hypothesis in self._asr.hypotheses():
                failures = 0
                text = (hypothesis.text or "").strip()

                if hypothesis.is_final:
                    finals += 1
                elif text:
                    interims += 1
                if verbose and text:
                    kind = "FINAL" if hypothesis.is_final else "interim"
                    print(
                        f"[DEBUG ASR] {kind:<7} "
                        f"conf={hypothesis.confidence if hypothesis.confidence is not None else '-'} "
                        f"{text[:90]}",
                        flush=True,
                    )
                if time.monotonic() - last_report >= self.REPORT_EVERY_S:
                    print(
                        f"[DEBUG ASR] last {time.monotonic() - last_report:.0f}s: "
                        f"{self._pushed_ms / 1000:.1f}s of audio sent, "
                        f"{interims} interim and {finals} final results back"
                        + ("" if interims or finals else
                           "  <-- audio is going out and NOTHING is coming "
                           "back: check the language setting, not the mic"),
                        flush=True,
                    )
                    interims = finals = 0
                    self._pushed_ms = 0.0
                    last_report = time.monotonic()

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
                if self._segmenter is not None:
                    from app.controllers.segmenter import Hypothesis

                    for unit in self._segmenter.push(
                        Hypothesis(text, hypothesis.is_final, hypothesis.at)
                    ):
                        self._speak_unit(unit.text, hypothesis.language)
                    continue

                if not hypothesis.is_final:
                    continue

                self._speak_unit(text, hypothesis.language)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                broke = exc
                self.failure = exc

            # A CLEAN end is the provider (or our own close) saying the
            # session is over - ending with it is correct. Only an ABNORMAL
            # end - the recogniser hanging up mid-session, surfaced as an
            # exception - is worth redialling, keeping the translator, the
            # synthesis queue and everything downstream: the alternative is
            # a quiet death discovered only when a push fails, ten lost
            # seconds later.
            if self._closed or broke is None:
                break
            failures += 1
            if failures > self.RECONNECT_ATTEMPTS:
                break
            if not await self._reopen_asr(failures):
                break
            self.failure = None
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

    def _speak_unit(self, text: str, language: str | None) -> None:
        """Hand one confirmed span of source speech to translate-and-speak."""
        job = asyncio.create_task(self._speak_translation(text, language))
        self._jobs.add(job)
        job.add_done_callback(self._jobs.discard)

    async def _speak_translation(
        self, said: str, heard_language: str | None = None
    ) -> None:
        """Translate one finished utterance and synthesise it, in order.

        `heard_language` is what the recogniser decided it was listening to.
        When that is already the language the client hears, the translator is
        SKIPPED entirely, and that is a correctness fix rather than an
        optimisation. Asked to render English into English, a language model
        does not copy - it edits, and the thing it edits most readily is the
        token it half-recognises: "Macrosoft" becomes "Microsoft", because
        "Microsoft" is what the training data says a person meant. A speaker
        introducing their own employer is then contradicted, fluently, in
        their own voice. Nothing said in the target language is worth a round
        trip through a model that is free to disagree with it.
        """
        target = self._cfg.target_language
        source = (heard_language or self._cfg.source_language or "ur").strip()

        # Stage one: translation, serialised only against other translations.
        # Utterance N+1 gets here while N is still inside `_speaking` below,
        # which is what lets its round trip overlap N's synthesis.
        async with self._translating:
            if self._closed:
                return

            if source.split("-")[0] == target.split("-")[0]:
                english = said.strip()
                if not english:
                    return
            else:
                unit = TranslationUnit(
                    text=said,
                    source_language=source,
                    target_language=target,
                )
                try:
                    translated = await self._translator.translate(
                        unit, self._context, self._terminology
                    )
                except ProviderError as exc:
                    self._translate_failures += 1
                    if self._translate_failures >= self.TRANSLATE_FAILURES_FATAL:
                        self.failure = exc
                        with contextlib.suppress(Exception):
                            self._audio.put_nowait(None)
                        return
                    print(
                        f"[DEBUG OUTGOING] translation failed for one "
                        f"utterance ({exc}) - skipped, the stream continues",
                        flush=True,
                    )
                    return
                except Exception:
                    return
                self._translate_failures = 0

                english = (translated.text or "").strip()
                if not english:
                    return

            self._context.add("you", english)

        # Stage two: synthesis, in the order stage one finished. A job can
        # only reach this acquire after its predecessor released
        # `_translating`, and that predecessor queues for `_speaking` before
        # yielding - so the order the utterances were heard in is the order
        # they are spoken in.
        async with self._speaking:
            if self._closed:
                return
            # Both halves, together: this path heard the Urdu and produced
            # the English, so the window can show them one under the other.
            await self._put(
                self._text,
                TranscriptDelta(text=english, is_final=True,
                                language=self._cfg.target_language,
                                source_text=said),
            )

            vendor = getattr(self._tts, "name", "edge")
            voice = english_voice(self._cfg.rate, tts_vendor=vendor)
            synthesized_any = False
            try:
                async for chunk in self._tts.synthesize(english, voice):
                    if self._closed or chunk.pcm.size == 0:
                        continue
                    synthesized_any = True
                    await self._put(self._audio, chunk)
            except Exception as exc:
                # `not synthesized_any` used to gate this, on the reasoning
                # that half a sentence in one voice and half in another is
                # worse than either. It is not worse than the client hearing
                # the first two words and then nothing, which is what a
                # mid-stream failure actually produced. Speaking the whole
                # line again in the free voice is the least bad of the three.
                if vendor.startswith("eleven"):
                    try:
                        from app.providers.tts import EdgeTextToSpeech
                        fallback_tts = EdgeTextToSpeech()
                        fallback_voice = english_voice(self._cfg.rate, tts_vendor="edge")
                        async for chunk in fallback_tts.synthesize(english, fallback_voice):
                            if self._closed or chunk.pcm.size == 0:
                                continue
                            await self._put(self._audio, chunk)
                        return
                    except Exception:
                        pass
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
        keyterms = self.terminology.keyterms() if self.terminology else []
        asr_cfg = AsrConfig(
            language=cfg.source_language or "ur",
            rate=RATE_ASR,
            interim_results=True,
            keyterms=keyterms,
        )
        asr_stream = await self.asr.open(asr_cfg)
        return CascadeS2stStream(
            asr_stream,
            self.translator,
            self.tts,
            cfg=cfg,
            terminology=self.terminology,
            reopen=lambda: self.asr.open(asr_cfg),
        )
