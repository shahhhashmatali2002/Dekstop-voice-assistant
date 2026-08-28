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
from dataclasses import dataclass

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
from app.providers.urdu_speech import ends_sentence, for_speech


@dataclass
class _PlayItem:
    """One block of synthesised audio on its way to the sink.

    `trace` is set on the first block of an utterance only, and `started` is
    the event the synthesiser waits on: it is set the moment that first block
    is actually written, which is what makes the latency figure mean "sound
    came out" rather than "sound was queued".
    """

    pcm: np.ndarray
    trace: UnitTrace | None = None
    started: asyncio.Event | None = None
    ms: float = 0.0


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
        preroll_ms: float = 250.0,
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

        self.preroll_ms = max(0.0, preroll_ms)

        self.context = RollingContext()
        self._audio_in: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=100)
        self._play_q: asyncio.Queue["_PlayItem"] = asyncio.Queue()
        self._units: asyncio.Queue = asyncio.Queue(maxsize=32)
        self._queued_ms = 0.0
        self._stream = None
        self._tasks: list[asyncio.Task] = []
        self._speaking_trace: UnitTrace | None = None
        self._running = False
        self._ducked = False

        # Enough to tell the four silences apart. They look identical in the
        # transcript window - it is empty - and they have completely different
        # fixes: nothing routed to the device, something routed but muted, a
        # recogniser that is not answering, and a recogniser that is answering
        # but has not confirmed a phrase yet.
        self.fed_ms = 0.0          # audio taken off the capture ring
        self.speech_ms = 0.0       # of that, what the gate called speech
        self.hypotheses = 0        # what the recogniser sent back
        self.units = 0             # phrases confirmed and sent to translate

    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def feed(self, pcm: np.ndarray) -> None:
        """Hand in canonical 16 kHz mono int16. Called from the audio pump."""
        self.fed_ms += 1000.0 * pcm.size / RATE_ASR
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
        self._stream = await self.asr.open(self._asr_config())
        # Start was just pressed. Get something audible out quickly, then
        # settle into the configured trade.
        self.segmenter.hurry()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._pump_audio(), name="incoming-audio"),
            asyncio.create_task(self._consume_hypotheses(), name="incoming-asr"),
            asyncio.create_task(self._play_loop(), name="incoming-playback"),
            asyncio.create_task(self._unit_worker(), name="incoming-units"),
        ]

    def _asr_config(self) -> AsrConfig:
        return AsrConfig(
            language="en",
            rate=RATE_ASR,
            interim_results=True,
            keyterms=self.terminology.keyterms(),
        )

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
        self._drain_playback()

    # ------------------------------------------------------------------

    #: A streaming recogniser closes the socket when it hears nothing at all
    #: for long enough - Deepgram allows about ten seconds. That is a problem
    #: because an IDLE capture endpoint delivers no callbacks at all: WASAPI
    #: loopback on a device nothing is playing into is not a stream of
    #: silence, it is no stream. So the queue below simply stays empty, this
    #: task waits on it for ever, and the session is closed underneath us -
    #: which the user sees as translation dying the moment a video is paused,
    #: buffers, or loses its audio device.
    KEEPALIVE_S = 4.0

    async def _pump_audio(self) -> None:
        n = frame_samples(RATE_ASR)
        silence = np.zeros(n, dtype=np.int16)
        carry = np.empty(0, dtype=np.int16)
        last_push = now()

        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_in.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if (
                    self._stream is not None
                    and now() - last_push >= self.KEEPALIVE_S
                ):
                    with contextlib.suppress(ProviderError):
                        await self._stream.push_audio(silence)
                        last_push = now()
                continue

            buf = np.concatenate([carry, chunk]) if carry.size else chunk
            usable = len(buf) - (len(buf) % n)
            carry = buf[usable:]

            for i in range(0, usable, n):
                frame = buf[i : i + n]
                res = self.gate.push(frame)

                if res.speech:
                    self.speech_ms += 1000.0 * frame.size / RATE_ASR

                if res.event is SpeechEvent.ONSET:
                    # Deliberately NOT segmenter.reset(). The gate reports an
                    # onset after every breath, while the recogniser is still
                    # extending one utterance across it; clearing what we have
                    # spoken here makes the next hypothesis arrive as though
                    # it were new and get spoken from the top again.
                    self._speaking_trace = self.traces.new(Direction.INCOMING)
                    self._speaking_trace.mark("frame_sent")


                if self._stream is not None:
                    audio_frame = res.audio if res.audio is not None else frame
                    try:
                        await self._stream.push_audio(audio_frame)
                        last_push = now()
                    except ProviderError:
                        # The consumer owns the connection: it will notice the
                        # same failure and reopen. Tearing the pipeline down
                        # from here would race it and lose the audio that a
                        # reconnect is about to be able to use.
                        pass


                if res.event is SpeechEvent.OFFSET:
                    for unit in self.segmenter.flush():
                        self._enqueue(unit)

    #: Consecutive failed reopens before we give up and fall back to the
    #: original audio. A dropped websocket is ordinary - a laptop sleeping, a
    #: wifi handover, a provider restarting one node - and ending the session
    #: for one is a worse answer than waiting a second and dialling again.
    RECONNECT_ATTEMPTS = 3

    async def _consume_hypotheses(self) -> None:
        """Read hypotheses, and own the lifetime of the connection.

        The connection is owned HERE rather than shared with the audio pump
        because two tasks reacting to the same dropped socket race each other:
        one tears the pipeline down while the other is reopening it. The pump
        writes to whatever stream is current and ignores its own failures; this
        loop decides whether there is still a session.
        """
        failures = 0
        while self._running:
            try:
                if self._stream is None:
                    break
                async for hyp in self._stream.hypotheses():
                    failures = 0
                    self.hypotheses += 1
                    if self._speaking_trace and not self._speaking_trace.t_hypothesis:
                        self._speaking_trace.mark("hypothesis")
                    for unit in self.segmenter.push(
                        Hypothesis(hyp.text, hyp.is_final, hyp.at, speaker=hyp.speaker)
                    ):
                        self._enqueue(unit)
                reason = "the recogniser closed the stream"
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                reason = str(exc)
            except Exception as exc:  # pragma: no cover
                reason = str(exc)

            if not self._running:
                return

            failures += 1
            if failures > self.RECONNECT_ATTEMPTS or not await self._reopen(failures):
                await self._degrade(f"recognition stream: {reason}")
                return

    async def _reopen(self, attempt: int) -> bool:
        """Dial the recogniser again. False when it will not answer."""
        await asyncio.sleep(min(2.0, 0.3 * attempt))
        old, self._stream = self._stream, None
        if old is not None:
            with contextlib.suppress(Exception):
                await old.aclose()
        try:
            self._stream = await self.asr.open(self._asr_config())
        except Exception:
            return False
        with contextlib.suppress(Exception):
            self.metrics.note_reconnect(Direction.INCOMING)
        # A new session has heard nothing yet; anything half-recognised
        # belongs to a stream that no longer exists.
        self.segmenter.reset()
        self.segmenter.hurry()
        return True

    # ------------------------------------------------------------------

    def _enqueue(self, unit) -> None:
        """Hand a unit to the worker. Never blocks the caller.

        Counted here rather than in the worker: this is the moment a phrase was
        confirmed, and the four silences are told apart by where the count
        stops.

        THE RULE THIS EXISTS FOR: nothing on the recognition path may wait on
        the speech path.

        Translating and synthesising a unit takes one to several seconds, and
        both callers here are on the recognition path - one is the task that
        feeds audio to the recogniser. Awaiting synthesis inside them stops
        that feed, and a streaming recogniser that stops receiving audio does
        not wait politely: Deepgram closes the socket with 1011 after about
        ten seconds of silence, which surfaces as translation dying mid-video
        for no visible reason.
        """
        self.units += 1
        try:
            self._units.put_nowait(unit)
        except asyncio.QueueFull:
            # Falling this far behind is a real failure, but the newest speech
            # is worth more than the oldest.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._units.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._units.put_nowait(unit)

    async def _unit_worker(self) -> None:
        """One unit at a time, in order, off the recognition path."""
        while True:
            unit = await self._units.get()
            try:
                await self._translate_and_speak(unit)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - must never die
                if self.on_degrade:
                    with contextlib.suppress(Exception):
                        self.on_degrade("impair", f"unit failed: {exc}")

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
            trace.drop_reason = (
                f"playback is {queue_ms / 1000:.1f} s behind"
            )
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

        # The transcript keeps the translation as written; only the voice gets
        # the rewritten version - Latin words said in Urdu, and an ending that
        # matches whether the speaker actually finished the thought. See
        # app/providers/urdu_speech.
        spoken = for_speech(
            translated.text,
            final=ends_sentence(unit.text, getattr(unit, "reason", "")),
        )
        await self._speak(spoken, trace, decision.rate)

    async def _speak(self, text: str, trace: UnitTrace, rate: float) -> None:
        """Synthesise one unit and hand it to the player.

        TWO THINGS THIS DELIBERATELY DOES NOT DO
        ----------------------------------------
        It does not write to the sink, and it does not wait for the utterance
        to finish playing.

        Writing straight to the sink makes playback exactly as smooth as the
        network is: the render callback drains the ring every period whether
        or not anything is filling it, and an empty ring is served as silence,
        so one chunk that arrives 40 ms late is 40 ms of silence punched
        through the middle of a word. The player holds a cushion of
        `preroll_ms` before it starts, which is what that silence lands in
        instead.

        Waiting for the utterance to finish is worse, and is the difference
        between fluent speech and a sentence read out one phrase at a time by
        someone who stops to think between each. Recognition, translation and
        synthesis of the NEXT unit take one to two seconds, and if they only
        begin once this unit has finished playing, that pause is heard between
        every phrase. So this returns as soon as the first block is actually
        audible - the tail plays while the next unit is being prepared.
        """
        # Ask the voice to hurry rather than stretching what it has already
        # said. Per-chunk time-stretching has no state across the seam between
        # chunks, and a seam every few hundred milliseconds is the metallic
        # edge users hear as robotic. A neural voice reading faster just sounds
        # like a person reading faster. The DSP path stays for providers that
        # cannot be asked (see SpeedHint in app/providers/tts).
        hinted = False
        if rate > 1.001 and getattr(self.tts, "supports_speed", False):
            with contextlib.suppress(Exception):
                self.tts.set_speed(rate)
                hinted = True
                rate = 1.0

        first = True
        held: list[np.ndarray] = []
        held_ms = 0.0
        started: asyncio.Event | None = None
        channels = max(1, getattr(self.sink, "channels", 1))
        samples_per_ms = self.sink.rate * channels / 1000.0

        async def release() -> None:
            """Queue the cushion as one item and remember to wait for it."""
            nonlocal held, held_ms, started, first
            if not held:
                return
            block = np.concatenate(held) if len(held) > 1 else held[0]
            item = _PlayItem(
                pcm=block,
                trace=trace if first else None,
                started=asyncio.Event() if first else None,
                ms=block.size / samples_per_ms,
            )
            if first:
                started = item.started
                first = False
            self._queued_ms += item.ms
            await self._play_q.put(item)
            held = []
            held_ms = 0.0

        try:
            async for chunk in self.tts.synthesize(text, self.voice):
                if chunk.pcm.size == 0:
                    continue
                if not trace.t_tts_first_chunk:
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

                held.append(pcm)
                held_ms += pcm.size / samples_per_ms
                # The cushion applies to the start of the utterance. After
                # that, hand blocks over as they arrive - the player is the
                # one that paces them.
                if held_ms >= (self.preroll_ms if started is None else 0.0):
                    await release()

            # An utterance shorter than the cushion never reaches the
            # threshold. It still has to be spoken - "kya?" is a whole turn.
            await release()

            # Wait for sound, not for silence: the first block only.
            if started is not None:
                await started.wait()
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            # §14.2: TTS down means text-only, not a dead pipeline.
            trace.dropped = True
            trace.drop_reason = f"synthesis: {exc}"
            if self.on_degrade:
                self.on_degrade("impair", f"synthesis unavailable: {exc}")
        finally:
            # The hint is per utterance. Leaving it set would speed up every
            # unit after one busy moment.
            if hinted:
                with contextlib.suppress(Exception):
                    self.tts.set_speed(1.0)
            self.metrics.record(trace)
            self._speaking_trace = None

    # ------------------------------------------------------------------

    def duck(self, on: bool) -> None:
        """Barge-in step 1: attenuate immediately, reversibly (§7.2)."""
        self._ducked = on

    async def _play_loop(self) -> None:
        """The only writer to the sink. Paces audio and absorbs jitter.

        Backpressure lives here rather than in the synthesiser: the ring is
        half a second deep, and a writer that ignores that simply overwrites
        audio the listener has not heard yet. Waiting for room means the queue
        - and therefore the measured lag - grows honestly instead, which is
        what the drift controller needs to see (§7.6).
        """
        while True:
            item = await self._play_q.get()
            ring = getattr(self.sink, "ring", None)

            def sounded() -> None:
                """The listener is now hearing this block."""
                if item.trace is not None and not item.trace.t_playback_start:
                    item.trace.mark("playback_start")
                if item.started is not None and not item.started.is_set():
                    item.started.set()

            if ring is None:
                self.sink.write(item.pcm)
                sounded()
            else:
                # A block bigger than the whole ring can never "fit", so
                # waiting for room would wait for ever. Feed it in slices.
                limit = max(1, ring.capacity // 2)
                for start in range(0, item.pcm.size, limit):
                    slice_ = item.pcm[start : start + limit]
                    while ring.free < slice_.size:
                        await asyncio.sleep(0.010)
                    self.sink.write(slice_)
                    # Sound is out as soon as the first slice lands; the
                    # synthesiser has no reason to wait for the rest.
                    sounded()
            self._queued_ms = max(0.0, self._queued_ms - item.ms)

    def flush_playback(self) -> None:
        """Barge-in step 3: the consultant really is talking. Drop the rest."""
        self._drain_playback()
        with contextlib.suppress(Exception):
            self.sink.flush()

    def _drain_playback(self) -> None:
        """Throw away everything queued but not yet written."""
        while True:
            try:
                item = self._play_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            # Anything waiting on this block must not wait forever.
            if item.started is not None:
                item.started.set()
        self._queued_ms = 0.0

    def _queue_ms(self) -> float:
        """How far behind the speaker playback actually is.

        The sink ring alone is not that number. The ring is a fixed-size
        buffer that the player deliberately keeps full, so reading it reports
        a constant half second of "lag" whether the pipeline is keeping up or
        minutes behind - and a drift controller fed that number time-stretches
        every utterance forever, for no reason, which is heard as a permanent
        metallic edge on the voice. Audio still waiting in the play queue is
        the part that actually represents falling behind.
        """
        depth = 0.0
        ring = getattr(self.sink, "ring", None)
        if ring is not None:
            depth = ring.depth_ms(self.sink.rate * getattr(self.sink, "channels", 1))
        return depth + self._queued_ms

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
