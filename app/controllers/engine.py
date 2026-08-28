"""The engine: owns the asyncio loop, both pipelines, and the always-on wire.

Spec reference: §4.2, §2.5, §11.5, §14.

THE INVARIANT
-------------
The engine starts with the application and stops with it, never with the Start
button. Audio flows in every mode. Translation is a mode that is switched into
and out of, layered on top of a wire that is always live.

    launch          -> audio streams open, wires pumping, mode = PASSTHROUGH
    Start           -> wires muted, pipelines translating, mode = TRANSLATING
    Pause           -> wires unmuted, connections kept warm, mode = PAUSED
    Stop            -> wires unmuted, connections closed, mode = PASSTHROUGH
    failure         -> the affected wire unmutes; the other keeps translating

Threading (§4.2): the engine owns one thread running an asyncio loop. The UI
calls into it from the Qt thread via `submit`, which is thread-safe. Nothing
here ever touches a Qt object.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import threading
from dataclasses import dataclass

import numpy as np

from app.audio import convert
from app.audio.formats import RATE_ASR, RATE_S2ST
from app.controllers.drift import DriftController
from app.controllers.incoming import IncomingPipeline
from app.controllers.outgoing import BargeInHandler, OutgoingPipeline
from app.controllers.vad import VoiceActivityGate
from app.providers.base import VoiceId
from app.models.telemetry import Direction, MetricsBus, TraceFactory
from app.models.config import Config
from app.models.state import ModeMachine
from app.models.terminology import Terminology


@dataclass
class AudioPorts:
    """The endpoints and the wires the engine drives.

    `mic_source` and `vmic_sink` are None in video mode. Nothing is being
    sent to anybody there: something is playing and you want to understand
    it, so the microphone is not merely unused - opening it would put a live
    capture of the room into a session that has no use for one.
    """

    mic_source: object
    vmic_sink: object
    teams_source: object
    phones_sink: object
    wires: object  # PassthroughPair


class Engine:
    """Owns everything that is not UI."""

    def __init__(
        self,
        config: Config,
        ports: AudioPorts,
        providers: dict,
        *,
        terminology: Terminology | None = None,
        outgoing: bool = True,
    ) -> None:
        self.config = config
        self.ports = ports
        self.providers = providers
        self.terminology = terminology or Terminology()
        #: Whether the Urdu -> English direction exists at all.
        self.outgoing = outgoing and ports.mic_source is not None

        self.modes = ModeMachine()
        self.metrics = MetricsBus()
        self.traces = TraceFactory()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._incoming: IncomingPipeline | None = None
        #: When this stretch of translating began, and whether its silence has
        #: already been explained.
        self._translating_since: float | None = None
        self._said_why = False
        self._outgoing: OutgoingPipeline | None = None
        self._bargein: BargeInHandler | None = None
        self._pump_task: asyncio.Task | None = None
        self._closing = False

        self.on_transcript = None
        self.on_notice = None

    # ------------------------------------------------------------------
    # lifecycle - thread-safe, callable from the Qt thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bring up the audio wire. Does NOT start translating."""
        if self._thread is not None:
            return
        self.ports.wires.start()
        self.modes.stop("application started")  # PASSTHROUGH, audio flowing
        self._thread = threading.Thread(
            target=self._run_loop, name="engine", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def shutdown(self) -> None:
        """Stop everything, in order, exactly once.

        Order matters and getting it wrong is noisy rather than fatal, which
        is why it survived unnoticed: the pump task has to be cancelled from
        INSIDE the loop before the loop stops, or it is still pending when the
        loop closes and asyncio complains about a destroyed task. Being
        idempotent matters too - both the tray Quit and the aboutToQuit
        handler reach here.
        """
        if self._loop is None or self._closing:
            return
        self._closing = True

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(
                self._shutdown_async(), self._loop
            ).result(timeout=5)

        with contextlib.suppress(Exception):
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        self._loop = None

        with contextlib.suppress(Exception):
            self.ports.wires.stop()

    async def _shutdown_async(self) -> None:
        """Runs on the engine loop: close pipelines, then cancel the pump."""
        await self._stop_translating()
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
            self._pump_task = None

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pump_task = self._loop.create_task(self._audio_pump())
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            # Anything still pending would otherwise be reported as destroyed.
            with contextlib.suppress(Exception):
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            with contextlib.suppress(Exception):
                self._loop.close()

    def submit(self, coro):
        """Schedule work on the engine loop from any thread."""
        if self._loop is None:
            raise RuntimeError("engine is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------
    # controls
    # ------------------------------------------------------------------

    def start_translating(self):
        return self.submit(self._start_translating())

    def pause(self):
        return self.submit(self._pause())

    def stop_translating(self):
        return self.submit(self._stop_translating())

    def toggle_pause(self):
        from app.models.state import Mode

        if self.modes.mode is Mode.PAUSED:
            return self.start_translating()
        if self.modes.mode is Mode.TRANSLATING:
            return self.pause()
        return self.start_translating()

    # ------------------------------------------------------------------

    async def _start_translating(self) -> None:
        p = self.config.pipeline
        voice = VoiceId(
            vendor=self.config.providers.tts,
            name=self.config.providers.urdu_voice,
            rate=RATE_S2ST,
        )

        self._incoming = IncomingPipeline(
            asr=self.providers["asr"],
            translator=self.providers["translator"],
            tts=self.providers["tts"],
            voice=voice,
            sink=self.ports.phones_sink,
            metrics=self.metrics,
            traces=self.traces,
            terminology=self.terminology,
            gate=VoiceActivityGate(
                rate=RATE_ASR,
                hangover_ms=p.vad_hangover_ms,
                preroll_ms=p.vad_preroll_ms,
            ),
            drift=DriftController(
                comfortable_ms=p.drift_comfortable_ms,
                stretch_ms=p.drift_stretch_ms,
                aggressive_ms=p.drift_aggressive_ms,
                max_rate=p.drift_max_rate,
            ),
            preroll_ms=p.playback_preroll_ms,
            on_transcript=self._transcript,
            on_degrade=lambda kind, why: self._degrade(Direction.INCOMING, kind, why),
        )
        self._outgoing = None if not self.outgoing else OutgoingPipeline(
            s2st=self.providers["s2st"],
            sink=self.ports.vmic_sink,
            metrics=self.metrics,
            traces=self.traces,
            gate=VoiceActivityGate(
                rate=RATE_S2ST,
                hangover_ms=p.vad_hangover_ms,
                preroll_ms=p.vad_preroll_ms,
            ),
            on_transcript=self._transcript,
            on_degrade=lambda kind, why: self._degrade(Direction.OUTGOING, kind, why),
            suppress_backchannels=p.suppress_backchannels,
        )
        if self._outgoing is not None:
            self._bargein = BargeInHandler(self._incoming, p.barge_in_confirm_ms)
            self._outgoing.on_speech = self._bargein.on_speech

        failed: list[tuple[Direction, str]] = []
        try:
            await self._incoming.start()
        except Exception as exc:
            failed.append((Direction.INCOMING, str(exc)))
        if self._outgoing is not None:
            try:
                await self._outgoing.start()
            except Exception as exc:
                failed.append((Direction.OUTGOING, str(exc)))

        self.modes.start()
        self._translating_since = None
        self._said_why = False
        # Mute only the wires whose pipeline actually came up. A wire whose
        # pipeline failed must keep passing raw audio (§2.5, §14.1).
        if self._incoming.running:
            self.ports.wires.incoming.mute()
        if self._outgoing is not None and self._outgoing.running:
            self.ports.wires.outgoing.mute()
        for direction, why in failed:
            self.modes.degrade(direction, why)
            self._notice("error", f"{direction.value} translation unavailable: {why}")

    async def _pause(self) -> None:
        # Connections stay open and warm - resuming must cost nothing (§11.3).
        self.ports.wires.unmute_all()
        self.modes.pause()
        self._translating_since = None
        self._said_why = False

    async def _stop_translating(self) -> None:
        self.ports.wires.unmute_all()
        for pipe in (self._incoming, self._outgoing):
            if pipe is not None:
                with contextlib.suppress(Exception):
                    await pipe.stop()
        self._incoming = None
        self._outgoing = None
        self._bargein = None
        self.modes.stop()
        self._translating_since = None
        self._said_why = False

    # ------------------------------------------------------------------
    # the audio pump: rings -> pipelines
    # ------------------------------------------------------------------

    async def _audio_pump(self) -> None:
        """Moves captured audio from the rings into whichever pipeline is live.

        Runs forever, independent of mode. When not translating it simply has
        nowhere to deliver to - the passthrough wires are doing the work.
        """
        mic = self.ports.mic_source
        teams = self.ports.teams_source
        while True:
            try:
                if (
                    mic is not None
                    and self._outgoing is not None
                    and self._outgoing.running
                ):
                    raw = mic.ring.read_available()
                    if raw.size:
                        self._outgoing.feed(
                            convert.endpoint_to_pipeline(
                                raw, mic.rate, mic.channels, RATE_S2ST
                            )
                        )
                if self._incoming is not None and self._incoming.running:
                    raw = teams.ring.read_available()
                    if raw.size:
                        self._incoming.feed(
                            convert.endpoint_to_pipeline(
                                raw, teams.rate, teams.channels, RATE_ASR
                            )
                        )
                self._publish_audio_health()
                self._watch_for_silence()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - must never die
                self._notice("warning", f"audio pump: {exc}")
            await asyncio.sleep(0.01)

    #: How long to translate before saying that nothing is arriving. See the
    #: same constant in listen.py - both entry points need this, because the
    #: failure is in the routing, which neither of them can see.
    SILENCE_NOTICE_S = 12.0

    def _watch_for_silence(self) -> None:
        """Say why nothing is happening, once, instead of showing nothing.

        Four unrelated failures produce the same picture - a transcript window
        that stays empty - and each has a different fix. They are told apart by
        where the pipeline's counters stop.
        """
        pipe = self._incoming
        if self._said_why or pipe is None or not pipe.running:
            return
        if self._translating_since is None:
            self._translating_since = time.monotonic()
            return
        if time.monotonic() - self._translating_since < self.SILENCE_NOTICE_S:
            return

        source = (
            getattr(getattr(self.config, "devices", None), "teams_speaker", "")
            or "the source device"
        )
        # Proof beats thresholds: anything back from the recogniser means the
        # path works and there is nothing to explain.
        if pipe.hypotheses or pipe.units:
            return

        if pipe.fed_ms < 500.0:
            self._notice("warning", (
                f"Nothing is arriving from {source!r}. The device is open but "
                "silent, so no application is playing to it - check the app's "
                "output in Volume mixer, and that another copy of this program "
                "is not already holding the device."
            ))
        elif pipe.speech_ms < 300.0:
            self._notice("warning", (
                f"Audio is arriving from {source!r} but there is no speech in "
                "it. Either the application is muted, or it is playing to a "
                "different device than the one we are listening to."
            ))
        else:
            self._notice("warning", (
                "Speech is reaching the recogniser but nothing is coming back. "
                "Check the network and DEEPGRAM_API_KEY."
            ))
        self._said_why = True

    def _publish_audio_health(self) -> None:
        for name, obj in (
            ("mic", self.ports.mic_source),
            ("virtual_mic", self.ports.vmic_sink),
            ("teams", self.ports.teams_source),
            ("headphones", self.ports.phones_sink),
        ):
            if obj is None:
                continue
            with contextlib.suppress(Exception):
                self.metrics.set_audio_health(name, obj.health())

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------

    def _transcript(self, payload: dict) -> None:
        if self.on_transcript:
            with contextlib.suppress(Exception):
                self.on_transcript(payload)

    def _notice(self, severity: str, message: str, action: str = "") -> None:
        if self.on_notice:
            with contextlib.suppress(Exception):
                self.on_notice(
                    {"severity": severity, "message": message, "action": action}
                )

    def _degrade(self, direction: Direction, kind: str, why: str) -> None:
        """A pipeline failed. Fall its wire back to raw audio - never silence."""
        wire = (
            self.ports.wires.incoming
            if direction is Direction.INCOMING
            else self.ports.wires.outgoing
        )
        if wire is None:  # a direction that does not exist cannot degrade
            return
        if kind == "impair":
            self.modes.impair(direction, why)
            self._notice("warning", why)
            return

        wire.unmute()
        self.modes.degrade(direction, why)
        if direction is Direction.OUTGOING:
            self._notice(
                "error",
                "Outgoing translation stopped — your own voice is going through "
                "untranslated.",
            )
        else:
            self._notice(
                "error",
                "Incoming translation stopped — you are hearing the original "
                "audio. You can still be heard.",
            )

    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "state": self.modes.snapshot().as_dict(),
            "metrics": self.metrics.snapshot(),
        }
