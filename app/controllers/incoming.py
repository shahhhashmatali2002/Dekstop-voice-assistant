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
from dataclasses import dataclass, replace

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
        converter=None,
        learner=None,
        clone_max_factor: float = 0.35,
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

        # Speaking in the source speaker's own voice. Both are optional and
        # both must be present for anything to happen: a converter with no
        # voice to imitate has nothing to do, and a learned voice with no
        # converter is just a recording nobody asked for.
        self.converter = converter
        self.learner = learner
        self.clone_max_factor = clone_max_factor
        self._clone_failed = False
        self._clone_ready = False

        #: Until when Urdu is expected to still be sounding in the room. Read
        #: by the outgoing direction so that, on speakers, our own translation
        #: is not picked up by the microphone and sent back to the client as
        #: English. Set from the play loop, because that is the only place that
        #: knows what has actually been handed to the hardware.
        self.speaking_until = 0.0
        self._clone_probe: asyncio.Task | None = None

        self.context = RollingContext()
        # Deep enough that continuous speech never overflows it. At ~20 ms
        # a frame this is ~40 s of audio held before the last-resort drop -
        # the pump empties it into the recogniser far faster than it fills,
        # so it only ever backs up while the event loop is blocked, and
        # dropping a reader's words mid-paragraph (the old maxsize=100, ~2 s)
        # was heard as "it missed some lines."
        self._audio_in: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=2000)
        self._play_q: asyncio.Queue["_PlayItem"] = asyncio.Queue()
        self._units: asyncio.Queue = asyncio.Queue(maxsize=32)
        #: Units whose translation is already in flight, in speaking order.
        #: The bound is the lookahead: how many translations may run ahead of
        #: the mouth. Small on purpose - each one is money and, if the queue
        #: ever drops units, wasted money.
        self._ready_q: asyncio.Queue = asyncio.Queue(maxsize=2)
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
        #: Dead-stream detection for the client -> you direction. Deepgram
        #: can go silently dead - the socket lives but no results come - and
        #: waiting for its ping timeout to notice was measured at up to forty
        #: seconds, seen by the user as "Incoming translation stopped". These
        #: let the pump force a reopen after a few seconds of speech with
        #: nothing back.
        self._loud_ms_since_result = 0.0
        self._last_result_at = now()
        self._force_reopen = False

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
            asyncio.create_task(self._translate_worker(), name="incoming-translate"),
            asyncio.create_task(self._speak_worker(), name="incoming-speak"),
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
        while True:
            try:
                *_ignored, job = self._ready_q.get_nowait()
            except asyncio.QueueEmpty:
                break
            job.cancel()
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
                    if self.learner is not None and not self._clone_failed:
                        # Only speech: a reference built from room tone
                        # converts every line into room tone.
                        self.learner.hear(frame)

                if res.event is SpeechEvent.ONSET:
                    # Deliberately NOT segmenter.reset(). The gate reports an
                    # onset after every breath, while the recogniser is still
                    # extending one utterance across it; clearing what we have
                    # spoken here makes the next hypothesis arrive as though
                    # it were new and get spoken from the top again.
                    self._speaking_trace = self.traces.new(Direction.INCOMING)
                    self._speaking_trace.mark("frame_sent")


                if res.speech:
                    self._loud_ms_since_result += 1000.0 * frame.size / RATE_ASR

                if self._stream is not None:
                    try:
                        await self._stream.push_audio(frame)
                        last_push = now()
                    except ProviderError:
                        # The consumer owns the connection: it will notice the
                        # same failure and reopen. Tearing the pipeline down
                        # from here would race it and lose the audio that a
                        # reconnect is about to be able to use.
                        pass

                # Seconds of SPEECH sent, nothing back: the stream is dead
                # whatever the socket claims. Close it so the consumer's
                # async-for ends and reopens, rather than blocking on results
                # that will never come.
                if (
                    self._stream is not None
                    and self._loud_ms_since_result > 5000.0
                    and now() - self._last_result_at > 15.0
                ):
                    print(
                        "[DEBUG INCOMING] speech sent for 15s with nothing "
                        "back - the recogniser is dead, reopening now",
                        flush=True,
                    )
                    self._loud_ms_since_result = 0.0
                    self._last_result_at = now()
                    self._force_reopen = True
                    with contextlib.suppress(Exception):
                        await self._stream.aclose()

                if res.event is SpeechEvent.OFFSET:
                    for unit in self.segmenter.flush():
                        self._enqueue(unit)

    #: Consecutive failed reopens before we give up and fall back to the
    #: original audio. A dropped websocket is ordinary - a laptop sleeping, a
    #: wifi handover, a provider restarting one node - and ending the session
    #: for one is a worse answer than waiting a second and dialling again.
    RECONNECT_ATTEMPTS = 6

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
                    self._loud_ms_since_result = 0.0
                    self._last_result_at = now()
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

            if self._force_reopen:
                # The pump closed a silently-dead stream on purpose. Reopen
                # without spending a failure - the stream did not fault, it
                # went quiet, and a healthy redial is expected to succeed.
                self._force_reopen = False
                if await self._reopen(1):
                    continue

            failures += 1
            if failures > self.RECONNECT_ATTEMPTS or not await self._reopen(failures):
                await self._degrade(f"recognition stream: {reason}")
                return

    async def _reopen(self, attempt: int) -> bool:
        """Dial the recogniser again. False when it will not answer."""
        await asyncio.sleep(min(4.0, 0.4 * attempt))
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

    async def _translate_worker(self) -> None:
        """Start a unit's translation the moment it is confirmed.

        THE SERIALISATION THIS REPLACES was the pipeline's dominant latency
        cost in continuous speech. The old worker did, per unit:

            translate -> synthesise -> wait for the first audible block

        one unit at a time, so unit N+1's translation did not START until
        unit N was already sounding. Each unit therefore paid its
        predecessor's full network cost on top of its own, the queue grew
        faster than it drained for as long as the speaker kept talking, and
        the drift controller's stretch-then-drop was the only thing keeping
        the lag finite.

        Translation is network-bound and does not touch the sink, so it does
        not need to be serialised with playback at all - only SPEAKING has to
        happen in order. This stage launches translations as units arrive
        (`_ready_q`'s bound is the lookahead) and hands them, still in
        order, to `_speak_worker`. While one unit is being synthesised and
        spoken, the next one or two are already in flight at the translator.
        """
        while True:
            unit = await self._units.get()
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
                continue

            # Context carries the SOURCE text, so it can be added before the
            # translation exists - which is exactly what lets unit N+1's
            # translation run concurrently and still see unit N in context.
            self.context.add("client", unit.text)
            job = asyncio.create_task(
                self._translate_unit(unit, trace, decision)
            )
            # Blocks when the lookahead is full - that is the backpressure
            # that stops money being spent translating text the drift
            # controller is about to drop.
            await self._ready_q.put((unit, trace, decision, job))

    async def _translate_unit(self, unit, trace, decision):
        """One translation round trip. Never raises: the result is awaited
        bare by the speech worker, and a unit that failed is a unit skipped,
        not a pipeline down."""
        tu = TranslationUnit(
            text=unit.text,
            source_language="en",
            target_language="ur",
            speaker=unit.speaker,
            prefer_brevity=decision.prefer_brevity,
        )
        try:
            return await self.translator.translate(
                tu, self.context, self.terminology
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            trace.dropped = True
            trace.drop_reason = str(exc)
            self.metrics.record(trace)
            self._emit_transcript(unit.text, "", dropped=True)
            return None

    async def _speak_worker(self) -> None:
        """Speak translated units strictly in order, one at a time.

        Order lives in `_ready_q`, not in when translations happen to finish:
        a fast translation of unit N+1 waits here behind a slow unit N.
        Serialising synthesis is deliberate - two utterances synthesised at
        once interleave their chunks in the play queue and are heard as two
        people talking over each other in one voice.
        """
        while True:
            unit, trace, decision, job = await self._ready_q.get()
            try:
                translated = await job
                if translated is None:
                    continue

                trace.mark("translated")
                if not translated.text.strip():
                    # Told to emit nothing rather than guess (§8.4.2).
                    self.metrics.record(trace)
                    continue

                self._emit_transcript(unit.text, translated.text)

                # The transcript keeps the translation as written; only the
                # voice gets the rewritten version - Latin words said in Urdu,
                # and an ending that matches whether the speaker actually
                # finished the thought. See app/providers/urdu_speech.
                spoken = for_speech(
                    translated.text,
                    final=ends_sentence(unit.text, getattr(unit, "reason", "")),
                )
                # The rate is decided NOW, not when the unit was confirmed:
                # the queue has moved while the translation was in flight,
                # and a rate measured before it is stale by exactly the
                # latency this pipeline is trying to hide.
                rate = self.drift.decide(self._queue_ms()).rate
                await self._speak(spoken, trace, rate)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - must never die
                if self.on_degrade:
                    with contextlib.suppress(Exception):
                        self.on_degrade("impair", f"unit failed: {exc}")

    def _clone_reference(self):
        """The voice to imitate, or None - which is the normal case.

        None means one of: the feature is off, the machine was too slow for it,
        conversion has already failed once this session, or not enough of the
        speaker has been heard yet. All four end the same way - the stock voice
        speaks - so they are one check rather than four.
        """
        if self.converter is None or self.learner is None or self._clone_failed:
            return None
        reference = self.learner.reference()
        if reference is None:
            return None
        if not self._clone_ready:
            if self._clone_probe is None or self._clone_probe.done():
                self._clone_probe = asyncio.ensure_future(self._probe(reference))
            return None
        return reference

    async def _clone(self, pcm: np.ndarray, rate: int, reference):
        """Convert one utterance to the speaker's voice, off the event loop.

        In a thread, and this is not a detail: conversion is seconds of solid
        CPU, and doing it inline would stop the audio pump feeding the
        recogniser - which closes the socket after about ten seconds of
        silence and takes the whole session down with it. That failure has
        already happened once in this project, from a different cause, and it
        looks like translation dying mid-video for no reason.

        A failure switches cloning off for the session rather than dropping the
        line. The stock voice saying it is worth far more than nothing saying
        it.
        """
        try:
            return await asyncio.to_thread(
                self.converter.convert, pcm, rate, reference
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._clone_failed = True
            if self.on_degrade:
                with contextlib.suppress(Exception):
                    self.on_degrade(
                        "impair",
                        f"speaking in the original voice is off: {exc}",
                    )
            return pcm, rate

    #: How much of the learned voice to convert as a trial. Long enough to be
    #: representative, short enough that a slow machine finds out quickly.
    CLONE_PROBE_S = 0.5


    async def _probe(self, reference) -> None:
        """Find out what conversion costs here, without a listener paying for it.

        The measurement has to happen on real speech - the converter trims
        non-speech before matching, so a synthetic clip is trimmed to nothing -
        and it must not happen on a line somebody is waiting to hear. The
        learned voice is both: real speech, and already ours.

        This is the second design. The first timed the first real utterance,
        which worked but cost the user that utterance and, on a machine four
        times too slow, a queue deep enough that the drift controller dropped
        the sentences behind it. A trial off to one side costs nobody anything.

        Two conversions, and only the second is timed: the first pays for
        loading the model, which is a one-off of tens of seconds and would
        otherwise be charged to a per-second figure and make every machine look
        hopeless.
        """
        self._match_voice_to_speaker(reference)

        pcm = reference.pcm[: int(self.CLONE_PROBE_S * reference.rate)]
        seconds = max(pcm.size / float(reference.rate or 1), 1e-6)

        factor, failure = await self._time_conversion(pcm, reference, seconds)
        if factor is not None and factor <= self.clone_max_factor:


            self._clone_ready = True
            self._say(
                f"now speaking in the original speaker's voice "
                f"({factor:.2f}s of conversion per second of audio)"
            )
            return

        # Too slow, or not installed. Rather than give up on the idea, drop to
        # the cheap version - pitch and tone matched to the speaker, which is a
        # resemblance rather than their voice, and which any machine can run.
        # Saying "no" to somebody who asked for this and can nearly have it is
        # a worse answer than saying "here is the part that fits".
        why = (
            f"the full voice clone needs {factor:.1f}s of compute per second "
            f"of audio here, which would add about {factor * 4:.0f}s to every "
            f"sentence"
            if factor is not None
            else f"the full voice clone is unavailable ({failure})"
        )
        if getattr(self.converter, "name", "") == "light-match":
            self._clone_failed = True
            self._say(f"speaking in the original voice is off: {failure or why}")
            return

        from app.providers.voice_clone import LightVoiceMatcher

        self.converter = LightVoiceMatcher()
        light, light_failure = await self._time_conversion(pcm, reference, seconds)
        if light_failure:
            self._clone_failed = True
            self._say(f"speaking in the original voice is off: {why}")
            return




        self._clone_ready = True
        self._say(
            f"{why}. Using the light voice match instead: the speaker's pitch "
            f"and tone on our voice, which is a resemblance rather than their "
            f"actual voice. For the real thing on this machine, use "
            f"pretranslate.py --clone-voice, which prepares everything before "
            f"playback."
        )

    async def _time_conversion(self, pcm, reference, seconds):
        """(factor, failure). Two conversions; only the second is timed.

        The first pays for loading the model - tens of seconds, once - and
        charging that to a per-second figure would make every machine look
        hopeless.
        """
        try:
            await asyncio.to_thread(
                self.converter.convert, pcm, reference.rate, reference
            )
            started = now()
            await asyncio.to_thread(
                self.converter.convert, pcm, reference.rate, reference
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return None, str(exc)
        return (now() - started) / seconds, ""

    def _match_voice_to_speaker(self, reference) -> None:
        """Read the speaker's pitch and pick the Urdu voice nearest to it.

        Worth more than the conversion that follows it. Pitch-shifting a female
        voice down an octave to stand in for a man produces a distorted woman,
        not a man; reading the line with the male voice starts an octave closer
        with no artefacts at all.
        """
        from app.providers.voice_clone import voice_for_pitch

        estimate = getattr(self.converter, "median_f0", None)
        if estimate is None:
            from app.providers.voice_clone import LightVoiceMatcher

            estimate = LightVoiceMatcher().median_f0
        try:
            f0 = estimate(reference.pcm, reference.rate)
            wanted = voice_for_pitch(f0)
        except Exception:
            return
        if not wanted or wanted == self.voice.name:
            return
        self.voice = replace(self.voice, name=wanted)
        self._say(
            f"the speaker sounds like a {'man' if f0 < 160 else 'woman'} "
            f"({f0:.0f} Hz) - switching to {wanted}"
        )

    def _say(self, message: str) -> None:
        if self.on_degrade:
            with contextlib.suppress(Exception):
                self.on_degrade("impair", message)

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

        def shape(pcm: np.ndarray, chunk_rate: int) -> np.ndarray:
            """Everything between the voice and the sink."""
            if rate > 1.001:
                pcm = time_stretch(pcm, rate, chunk_rate)
            if chunk_rate != self.sink.rate:
                pcm = convert.resample(pcm, chunk_rate, self.sink.rate)
            if getattr(self.sink, "channels", 1) > 1:
                pcm = convert.to_channels(pcm, self.sink.channels)
            if self._ducked:
                pcm = convert.apply_gain_db(pcm, -20.0)
            return pcm

        # Speaking in the source speaker's voice, if enough of it has been
        # heard yet. Conversion needs the WHOLE utterance - it matches frames
        # against the speaker's audio, and a fragment cut mid-word converts
        # into a fragment cut mid-word - so streaming is given up for this
        # utterance. That costs the streaming head start, which is why the
        # engine refuses to enable this at all unless conversion measured fast
        # enough on this machine to afford it.
        reference = self._clone_reference()
        raw: list[np.ndarray] = []
        raw_rate = 0

        try:
            async for chunk in self.tts.synthesize(text, self.voice):
                if chunk.pcm.size == 0:
                    continue
                if not trace.t_tts_first_chunk:
                    trace.mark("tts_first_chunk")

                if reference is not None:
                    raw.append(chunk.pcm)
                    raw_rate = chunk.rate
                    continue

                pcm = shape(chunk.pcm, chunk.rate)
                held.append(pcm)
                held_ms += pcm.size / samples_per_ms
                # The cushion applies to the start of the utterance. After
                # that, hand blocks over as they arrive - the player is the
                # one that paces them.
                if held_ms >= (self.preroll_ms if started is None else 0.0):
                    await release()

            if reference is not None and raw:
                whole = np.concatenate(raw) if len(raw) > 1 else raw[0]
                converted, converted_rate = await self._clone(
                    whole, raw_rate, reference
                )
                pcm = shape(converted, converted_rate)
                held.append(pcm)
                held_ms += pcm.size / samples_per_ms

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

    @property
    def is_speaking(self) -> bool:
        """Is Urdu AUDIBLE in the room right now (or within its tail)?

        This is the half-duplex signal, and it must be TIGHT: every extra
        moment it stays true is a moment the outgoing direction throws the
        user's own microphone away. It was watched discarding almost eight
        seconds in ten, because a broader version stayed true while Urdu was
        merely QUEUED - and during drift there is always something queued,
        so the microphone was muted more or less permanently.

        Only actual playback counts here. `speaking_until` is stamped a
        little past the end of each block that reaches the sink, so it spans
        one audible utterance without spanning the queue behind it."""
        return now() < self.speaking_until

    @property
    def is_speaking_or_pending(self) -> bool:
        """Urdu audible OR about to be - the barge-in signal.

        Barge-in wants the WIDER window: ducking is instant and reversible,
        so covering the gaps between blocks of one utterance (a slow TTS
        chunk, a full ring) costs nothing and stops a real interruption
        being missed. Kept separate from `is_speaking` so this breadth never
        leaks into the half-duplex mute, which pays for every extra
        millisecond in discarded speech."""
        return now() < self.speaking_until or self._queue_ms() > 1.0

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

            # A moment past the end of the block, not exactly the end: the
            # sink drains it over that time, and cutting the microphone back on
            # at the last sample means the tail is still in the air.
            self.speaking_until = max(self.speaking_until, now() + item.ms / 1000.0 + 0.25)

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
