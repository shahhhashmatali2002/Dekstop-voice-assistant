"""Pipeline A - incoming, English to Urdu, cascade.

Spec reference: §2.1, §8.2, §6, §7.6.

    Teams audio -> VAD -> streaming ASR -> stability + segmentation
                -> translation (rolling context) -> streaming Urdu TTS
                -> drift control -> headphones

This direction MUST be a cascade. Urdu is not an available output language for
any realtime speech-to-speech model (§8.2), so there is no single-model option
here and there will not be one until that changes.

The consultant hears only Urdu: the original English is never mixed into the
headphone path at conversational level (§5.6 F3).
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np

from app.audio import convert
from app.audio.formats import RATE_ASR, RATE_S2ST, frame_samples
from app.providers.base import (
    AsrConfig,
    ProviderError,
    RollingContext,
    TranslationUnit,
    VoiceId,
)
from app.models.terminology import Terminology
from app.models.telemetry import Direction, MetricsBus, TraceFactory, UnitTrace, now
from app.controllers.drift import DriftController, time_stretch
from app.controllers.segmenter import Hypothesis, Segmenter
from app.controllers.vad import SpeechEvent, VoiceActivityGate


class IncomingPipeline:
    """English in, Urdu out. Owns its own provider connections (§4.4)."""

    def __init__(
        self,
        *,
        asr,
        translator,
        tts,
        voice: VoiceId,
        sink,  # RenderSink or any object with .write(pcm) / .ring
        metrics: MetricsBus,
        traces: TraceFactory,
        terminology: Terminology | None = None,
        gate: VoiceActivityGate | None = None,
        segmenter: Segmenter | None = None,
        drift: DriftController | None = None,
        on_transcript=None,
        on_degrade=None,
    ) -> None:
        self.asr = asr
        self.translator = translator
        self.tts = tts
        self.voice = voice
        self.sink = sink
        self.metrics = metrics
        self.traces = traces
        self.terminology = terminology or Terminology()
        self.gate = gate or VoiceActivityGate(rate=RATE_ASR)
        self.segmenter = segmenter or Segmenter()
        self.drift = drift or DriftController()
        self.on_transcript = on_transcript
        self.on_degrade = on_degrade

        self.context = RollingContext()
        self._audio_in: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        self._stream = None
        self._tasks: list[asyncio.Task] = []
        self._speaking_trace: UnitTrace | None = None
        self._running = False
        self._ducked = False

    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def feed(self, pcm: np.ndarray) -> None:
        """Hand in canonical 16 kHz mono int16. Called from the audio pump."""
        try:
            self._audio_in.put_nowait(pcm)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._audio_in.get_nowait()  # drop oldest (§6.4)
            with contextlib.suppress(asyncio.QueueFull):
                self._audio_in.put_nowait(pcm)

    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        cfg = AsrConfig(
            language="en",
            rate=RATE_ASR,
            interim_results=True,
            keyterms=self.terminology.keyterms(),
        )
        self._stream = await self.asr.open(cfg)
        self._running = True
        self._tasks = [
            asyncio.create_task(self._pump_audio(), name="incoming-audio"),
            asyncio.create_task(self._consume_hypotheses(), name="incoming-asr"),
        ]

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks.clear()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.aclose()
            self._stream = None
        self.segmenter.reset()
        self.gate.reset()

    # ------------------------------------------------------------------

    async def _pump_audio(self) -> None:
        n = frame_samples(RATE_ASR)
        carry = np.empty(0, dtype=np.int16)
        while self._running:
            chunk = await self._audio_in.get()
            buf = np.concatenate([carry, chunk]) if carry.size else chunk
            usable = len(buf) - (len(buf) % n)
            carry = buf[usable:]

            for i in range(0, usable, n):
                frame = buf[i : i + n]
                res = self.gate.push(frame)

                if res.event is SpeechEvent.ONSET:
                    self._speaking_trace = self.traces.new(Direction.INCOMING)
                    self._speaking_trace.mark("frame_sent")

                if res.audio is not None and self._stream is not None:
                    try:
                        await self._stream.push_audio(res.audio)
                    except ProviderError as exc:
                        await self._degrade(f"recognition: {exc}")
                        return

                if res.event is SpeechEvent.OFFSET:
                    for unit in self.segmenter.flush():
                        await self._translate_and_speak(unit)

    async def _consume_hypotheses(self) -> None:
        assert self._stream is not None
        try:
            async for hyp in self._stream.hypotheses():
                if self._speaking_trace and not self._speaking_trace.t_hypothesis:
                    self._speaking_trace.mark("hypothesis")
                for unit in self.segmenter.push(
                    Hypothesis(hyp.text, hyp.is_final, hyp.at, speaker=hyp.speaker)
                ):
                    await self._translate_and_speak(unit)
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            await self._degrade(f"recognition stream: {exc}")
        except Exception as exc:  # pragma: no cover
            await self._degrade(f"recognition stream: {exc}")

    # ------------------------------------------------------------------

    async def _translate_and_speak(self, unit) -> None:
        trace = self._speaking_trace or self.traces.new(Direction.INCOMING)
        trace.mark("unit_emitted")
        trace.words = unit.words

        queue_ms = self._queue_ms()
        decision = self.drift.decide(queue_ms)
        self.metrics.set_queue_depth(Direction.INCOMING, queue_ms)
        self.metrics.set_drift(Direction.INCOMING, decision.rate)

        if decision.drop:
            # §7.6: dropping always beats growing, and it is never silent.
            trace.dropped = True
            trace.drop_reason = "playback queue over 3 s"
            self.metrics.record(trace)
            self._emit_transcript(unit.text, "", dropped=True)
            return

        tu = TranslationUnit(
            text=unit.text,
            source_language="en",
            target_language="ur",
            speaker=unit.speaker,
            prefer_brevity=decision.prefer_brevity,
        )
        try:
            translated = await self.translator.translate(
                tu, self.context, self.terminology
            )
        except ProviderError as exc:
            trace.dropped = True
            trace.drop_reason = str(exc)
            self.metrics.record(trace)
            self._emit_transcript(unit.text, "", dropped=True)
            return

        trace.mark("translated")
        if not translated.text.strip():
            # The translator was told to emit nothing rather than guess (§8.4.2).
            self.metrics.record(trace)
            return

        self.context.add("client", unit.text)
        self._emit_transcript(unit.text, translated.text)

        await self._speak(translated.text, trace, decision.rate)

    async def _speak(self, text: str, trace: UnitTrace, rate: float) -> None:
        first = True
        try:
            async for chunk in self.tts.synthesize(text, self.voice):
                if chunk.pcm.size == 0:
                    continue
                if first:
                    trace.mark("tts_first_chunk")
                pcm = chunk.pcm
                if rate > 1.001:
                    pcm = time_stretch(pcm, rate, chunk.rate)
                if chunk.rate != self.sink.rate:
                    pcm = convert.resample(pcm, chunk.rate, self.sink.rate)
                if getattr(self.sink, "channels", 1) > 1:
                    pcm = convert.to_channels(pcm, self.sink.channels)
                if self._ducked:
                    pcm = convert.apply_gain_db(pcm, -20.0)
                self.sink.write(pcm)
                if first:
                    trace.mark("playback_start")
                    first = False
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            # §14.2: TTS down means text-only, not a dead pipeline.
            trace.dropped = True
            trace.drop_reason = f"synthesis: {exc}"
            if self.on_degrade:
                self.on_degrade("impair", f"synthesis unavailable: {exc}")
        finally:
            self.metrics.record(trace)
            self._speaking_trace = None

    # ------------------------------------------------------------------

    def duck(self, on: bool) -> None:
        """Barge-in step 1: attenuate immediately, reversibly (§7.2)."""
        self._ducked = on

    def flush_playback(self) -> None:
        """Barge-in step 3: the consultant really is talking. Drop the rest."""
        with contextlib.suppress(Exception):
            self.sink.flush()

    def _queue_ms(self) -> float:
        ring = getattr(self.sink, "ring", None)
        if ring is None:
            return 0.0
        return ring.depth_ms(self.sink.rate * getattr(self.sink, "channels", 1))

    def _emit_transcript(self, source: str, translated: str, dropped: bool = False):
        if self.on_transcript:
            with contextlib.suppress(Exception):
                self.on_transcript(
                    {
                        "direction": Direction.INCOMING.value,
                        "source": source,
                        "translated": translated,
                        "dropped": dropped,
                        "at": now(),
                    }
                )

    async def _degrade(self, reason: str) -> None:
        self._running = False
        if self.on_degrade:
            with contextlib.suppress(Exception):
                self.on_degrade("degrade", reason)
