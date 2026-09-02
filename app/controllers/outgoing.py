"""Pipeline B - outgoing, Urdu to English.

Spec reference: §2.1, §8.2, §7.2.

    microphone -> VAD -> speech-to-speech translation -> virtual microphone

This direction CAN collapse into a single model, because Urdu is an accepted
input language and English is one of the thirteen supported outputs. One hop
instead of three, and prosody survives.

A cascade fallback is provided for when the S2ST provider is unavailable
(§14.2) - it is worse in every way except that it works.

Barge-in lives here rather than in the incoming pipeline, because the trigger
is the consultant starting to speak, which is this pipeline's VAD.
"""

from __future__ import annotations

import asyncio
import contextlib
import os

import numpy as np

from app.audio import convert
from app.audio.formats import RATE_S2ST, frame_samples
from app.providers.base import (
    ProviderError,
    RollingContext,
    S2stConfig,
    TranslationUnit,
)
from app.models.telemetry import Direction, MetricsBus, TraceFactory, now
from app.controllers.vad import SpeechEvent, VoiceActivityGate

#: What we tell the speech-to-speech provider we are speaking.
#:
#: "ur" is the DEFAULT and it is a compromise, not a claim: it is what this
#: pipeline was tuned and tested against. Setting OUTGOING_LANGUAGE=auto turns
#: on the two-recogniser referee in app/providers/dual_asr, which is the right
#: answer for a speaker who uses Urdu and English in the same call - but it
#: opens a second recognition stream and adds a settle window per utterance,
#: so it is opt-in until it has been measured on a real call rather than
#: assumed. See tools/bench_language.py.
SOURCE_LANGUAGE = (os.environ.get("OUTGOING_LANGUAGE") or "ur").strip().lower()

#: §7.3 - short acknowledgement tokens in both languages. Translating each one
#: into a spoken interruption makes a meeting unusable.
BACKCHANNELS = {
    "hmm", "mm", "mhm", "uh", "um", "ah", "oh", "ok", "okay", "right", "yeah",
    "yes", "no", "sure", "achha", "acha", "jee", "ji", "haan", "han", "theek",
}


def is_backchannel(text: str, duration_ms: float, max_ms: float = 600.0) -> bool:
    """Both short in time AND a known acknowledgement token."""
    if duration_ms > max_ms:
        return False
    words = text.strip().lower().strip(".,!?۔،؟").split()
    return bool(words) and len(words) <= 2 and all(w in BACKCHANNELS for w in words)


class OutgoingPipeline:
    """Urdu in, English out. Single-model where possible."""

    def __init__(
        self,
        *,
        s2st,
        sink,
        metrics: MetricsBus,
        traces: TraceFactory,
        gate: VoiceActivityGate | None = None,
        on_transcript=None,
        on_degrade=None,
        on_speech=None,  # barge-in hook: called with True/False
        suppress_backchannels: bool = True,
    ) -> None:
        self.s2st = s2st
        self.sink = sink
        self.metrics = metrics
        self.traces = traces
        self.gate = gate or VoiceActivityGate(rate=RATE_S2ST, frame_ms=20)
        self.on_transcript = on_transcript
        self.on_degrade = on_degrade
        self.on_speech = on_speech
        self.suppress_backchannels = suppress_backchannels

        self.context = RollingContext()
        # Deep enough that continuous speech never overflows it. At ~20 ms
        # a frame this is ~40 s of audio held before the last-resort drop -
        # the pump empties it into the recogniser far faster than it fills,
        # so it only ever backs up while the event loop is blocked, and
        # dropping a reader's words mid-paragraph (the old maxsize=100, ~2 s)
        # was heard as "it missed some lines."
        self._audio_in: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=2000)
        self._stream = None
        self._tasks: list[asyncio.Task] = []
        self._trace = None
        self._running = False
        self._speech_started_at = 0.0
        self._last_output_time = 0.0

    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if user is currently speaking Urdu or outgoing English audio is playing."""
        if getattr(self.gate, "speaking", False):
            return True
        return (now() - self._last_output_time) < 1.5


    @property
    def running(self) -> bool:
        return self._running


    def feed(self, pcm: np.ndarray) -> None:
        """Hand in canonical 24 kHz mono int16 from the microphone."""
        import os
        gain = float(os.environ.get("MIC_GAIN", "2.0"))
        if gain != 1.0 and pcm.size > 0:
            pcm = np.clip(pcm.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        try:
            self._audio_in.put_nowait(pcm)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._audio_in.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._audio_in.put_nowait(pcm)

    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        # "auto", not "ur". This speaker uses Urdu and English, separately
        # and mixed, and pinning the recogniser to one of them was what turned
        # "I am John Wick" into "John Bek". See app/providers/dual_asr.
        cfg = S2stConfig(
            source_language=SOURCE_LANGUAGE, target_language="en", rate=RATE_S2ST
        )
        self._stream = await self.s2st.open(cfg)
        self._running = True
        self._tasks = [
            asyncio.create_task(self._pump_audio(), name="outgoing-audio"),
            asyncio.create_task(self._consume_audio(), name="outgoing-out"),
            asyncio.create_task(self._consume_text(), name="outgoing-text"),
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
        self.gate.reset()

    # ------------------------------------------------------------------

    async def _pump_audio(self) -> None:
        n = frame_samples(RATE_S2ST)
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
                    print("[DEBUG OUTGOING] Speech ONSET detected", flush=True)
                    self._trace = self.traces.new(Direction.OUTGOING)
                    self._trace.mark("frame_sent")
                    self._speech_started_at = now()
                    self._notify_speech(True)
                elif res.event is SpeechEvent.OFFSET:
                    print("[DEBUG OUTGOING] Speech OFFSET detected", flush=True)
                    self._notify_speech(False)

                # Send the actual audio frame continuously so Deepgram's neural
                # acoustic model has complete, unchopped phonemes and natural context.
                payload = frame
                if self._stream is not None:
                    try:
                        await self._stream.push_audio(payload)
                    except Exception as exc:
                        print(f"[DEBUG OUTGOING] Push audio error: {exc}", flush=True)
                        try:
                            await self._restart_stream()
                            if self._stream is not None:
                                await self._stream.push_audio(payload)
                        except Exception as e:
                            print(f"[DEBUG OUTGOING] Reopen stream error: {e}", flush=True)
                            await self._degrade(f"translation stream: {exc}")
                            return

    #: How many times to redial before giving the direction up. One attempt
    #: was watched failing on a live session: the network hiccup that killed
    #: the stream also failed the single immediate handshake, and the whole
    #: outgoing direction died for what was a twenty-second outage.
    RECONNECT_ATTEMPTS = 5

    async def _restart_stream(self) -> None:
        if self._stream is not None:
            with contextlib.suppress(Exception):
                await self._stream.aclose()
            self._stream = None
        cfg = S2stConfig(
            source_language=SOURCE_LANGUAGE, target_language="en", rate=RATE_S2ST
        )
        last: Exception | None = None
        for attempt in range(1, self.RECONNECT_ATTEMPTS + 1):
            # 0.5, 1, 2, 4, 8 seconds - long enough for a wifi handover or a
            # provider node restart, short enough that speech loss is small.
            await asyncio.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
            try:
                self._stream = await self.s2st.open(cfg)
                last = None
                break
            except Exception as exc:
                last = exc
                print(
                    f"[DEBUG OUTGOING] reconnect {attempt}/"
                    f"{self.RECONNECT_ATTEMPTS} failed: {exc}",
                    flush=True,
                )
        if last is not None:
            raise ProviderError(f"could not reopen the translation stream: {last}")
        for t in self._tasks[1:]:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._tasks = [
            self._tasks[0],
            asyncio.create_task(self._consume_audio(), name="outgoing-out"),
            asyncio.create_task(self._consume_text(), name="outgoing-text"),
        ]



    #: Longest we will wait for the sink to make room before writing anyway.
    #: A sink that has not moved in a second is broken, not busy, and silence
    #: is a worse failure than a dropped sample.
    MAX_WAIT_S = 1.0

    async def _write_paced(self, pcm: np.ndarray) -> None:
        """Write no faster than the sink can play, and never faster.

        MEASURED, not guessed. tools/say_english.py pushed one sentence through
        this exact path and reported:

            samples written      336384
            ring dropped          22080

        The sink's ring holds 500 ms. The voice delivers a three-and-a-half
        second sentence in a fraction of that, and the ring - which never
        blocks, by design - drops the OLDEST samples to take the newest. So
        almost the whole sentence was thrown away before the sound card could
        play it, and what reached the client was a fragment or nothing at all,
        while the transcript happily showed a complete translation.

        Nothing here costs latency: the first slice goes straight into an empty
        ring exactly as before. It only stops us overwriting audio the device
        has not played yet.
        """
        ring = getattr(self.sink, "ring", None)
        if ring is None:
            self.sink.write(pcm)
            self._last_output_time = now()
            return

        index, total, waited = 0, pcm.size, 0.0
        while index < total and self._running:
            room = ring.free
            if room <= 0:
                if waited >= self.MAX_WAIT_S:
                    # Not draining at all. Write and take the loss rather than
                    # wait for room that is never coming.
                    self.sink.write(pcm[index:])
                    self._last_output_time = now()
                    return
                await asyncio.sleep(0.01)
                waited += 0.01
                continue
            waited = 0.0
            take = min(room, total - index)
            self.sink.write(pcm[index : index + take])
            self._last_output_time = now()
            index += take

    async def _consume_audio(self) -> None:
        assert self._stream is not None
        first = True
        try:
            async for chunk in self._stream.audio_out():
                if chunk.pcm.size == 0:
                    continue
                print(f"[DEBUG OUTGOING] Received audio chunk {chunk.pcm.size} samples", flush=True)
                if self._trace and first:
                    self._trace.mark("tts_first_chunk")
                pcm = chunk.pcm
                if chunk.rate != self.sink.rate:
                    pcm = await asyncio.to_thread(
                        convert.resample, pcm, chunk.rate, self.sink.rate
                    )
                if getattr(self.sink, "channels", 1) > 1:
                    pcm = convert.to_channels(pcm, self.sink.channels)
                await self._write_paced(pcm)

                if self._trace and first:
                    self._trace.mark("playback_start")
                    self.metrics.record(self._trace)
                    self._trace = None
                    first = False
        except asyncio.CancelledError:
            raise
        except ProviderError as exc:
            print(f"[DEBUG OUTGOING] Provider error in audio out: {exc}", flush=True)
            # Into the transcript window as well as the console: the console is
            # not where somebody on a call is looking, and a translator that
            # has quietly stopped translating is the worst thing this can do.
            if self.on_degrade:
                with contextlib.suppress(Exception):
                    self.on_degrade(
                        "degrade",
                        f"your Urdu is not reaching the client: {exc}",
                    )
            await self._degrade(f"translation audio: {exc}")
        except Exception as exc:  # pragma: no cover
            print(f"[DEBUG OUTGOING] Exception in audio out: {exc}", flush=True)
            await self._degrade(f"translation audio: {exc}")

    async def _consume_text(self) -> None:
        assert self._stream is not None
        try:
            async for delta in self._stream.transcripts():
                if not delta.text.strip():
                    continue
                print(f"[DEBUG OUTGOING] Transcript text: {delta.text} (final={delta.is_final})", flush=True)
                if delta.is_final:
                    self.context.add("you", delta.text)
                self._emit_transcript(
                    delta.text, delta.is_final, source=delta.source_text or ""
                )

        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - transcripts are advisory only
            pass

    # ------------------------------------------------------------------

    def _notify_speech(self, speaking: bool) -> None:
        if self.on_speech:
            with contextlib.suppress(Exception):
                self.on_speech(speaking)

    def _emit_transcript(self, text: str, final: bool, source: str = "") -> None:
        if self.on_transcript:
            with contextlib.suppress(Exception):
                self.on_transcript(
                    {
                        "direction": Direction.OUTGOING.value,
                        "source": source,
                        "translated": text,
                        "final": final,
                        "at": now(),
                    }
                )

    async def _degrade(self, reason: str) -> None:
        self._running = False
        if self.on_degrade:
            with contextlib.suppress(Exception):
                self.on_degrade("degrade", reason)


# ----------------------------------------------------------------------


class BargeInHandler:
    """Duck first, cancel second.

    Spec reference: §7.2.

        1. duck incoming playback to -20 dB   <- immediate, ~20 ms
        2. wait 250 ms                        <- is it real speech?
        3. if still speaking: stop TTS, flush playback, cancel pending units
           else: restore level, continue

    Ducking is instant and reversible; cancellation is neither. A cough, a
    chair scrape or a single "mm-hm" should duck and recover, not tear down an
    in-flight utterance.
    """

    def __init__(self, incoming, confirm_ms: float = 250.0) -> None:
        self.incoming = incoming
        self.confirm_ms = confirm_ms
        self._task: asyncio.Task | None = None
        self._speaking = False

    def on_speech(self, speaking: bool) -> None:
        self._speaking = speaking
        if speaking:
            self.incoming.duck(True)
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._confirm())
        else:
            self.incoming.duck(False)
            if self._task and not self._task.done():
                self._task.cancel()

    async def _confirm(self) -> None:
        try:
            await asyncio.sleep(self.confirm_ms / 1000.0)
        except asyncio.CancelledError:
            return
        if self._speaking:
            # Still going. This is a real interruption, not a cough.
            self.incoming.flush_playback()
            self.incoming.duck(False)
