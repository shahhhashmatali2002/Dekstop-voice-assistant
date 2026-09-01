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

import os

import numpy as np

from app.audio import convert
from app.audio.formats import RATE_ASR, RATE_S2ST
from app.controllers.drift import DriftController
from app.controllers.incoming import IncomingPipeline
from app.providers.factory import build_voice_clone
from app.audio.routing import DefaultOutputRouter, RoutingUnavailable
from app.controllers.outgoing import BargeInHandler, OutgoingPipeline
from app.controllers.segmenter import Segmenter
from app.controllers.vad import VoiceActivityGate
from app.providers.base import VoiceId
from app.models.telemetry import Direction, MetricsBus, TraceFactory, now
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


class _Named:
    """The feedback helpers take device objects; the config has only names."""

    def __init__(self, name: str) -> None:
        self.name = name


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

        #: Moves the machine's default output to the cable while translating,
        #: so nobody has to open the volume mixer - and puts it back on stop,
        #: on shutdown, and from an atexit hook if we die badly. A default left
        #: pointing at a cable is indistinguishable from broken sound hardware.
        self.router = DefaultOutputRouter()
        self._routing = None
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
        # Pay for the voice-clone model NOW, at launch, while nobody is on a
        # call. Loading it lazily meant torch imported and a gigabyte of
        # weights loaded in the middle of the first conversation - the GIL
        # bursts froze the audio loop for seconds and both recognisers died
        # with NET-0001. A stall during startup costs nothing anyone hears.
        threading.Thread(
            target=self._warm_clone, name="clone-warm", daemon=True
        ).start()
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
            self.router.restore()
        with contextlib.suppress(Exception):
            self.ports.wires.stop()

    def _warm_clone(self) -> None:
        """Import torch and load the clone model before any call is live."""
        try:
            p = self.config.providers
            if not getattr(p, "clone_voice", False):
                return
            from app.providers.voice_clone import build_converter

            converter = build_converter(getattr(p, "clone_backend", "light"))
            load = getattr(converter, "_load", None)
            if load is not None:
                load()
        except Exception:
            # The pipeline probes and degrades on its own; warming is only
            # ever an optimisation.
            pass

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
        self._loop.create_task(self._watch_stalls())
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

    def _route_audio(self) -> None:
        """Point Windows at the cable, if the user asked us to do the routing.

        Never fatal. Failing to route means the user does it by hand, which is
        where they were before this existed; refusing to translate over it
        would be a strictly worse outcome.
        """
        from app.audio.devices import feedback_loop_warnings

        devices = self.config.devices
        for warning in feedback_loop_warnings(
            None, None, _Named(devices.virtual_mic), _Named(devices.teams_speaker)
        ):
            self._notice("warning", warning)

        if not getattr(devices, "auto_route", False):
            return
        target = devices.teams_speaker
        if not target:
            return
        try:
            was = self.router.route(target)
            # A call needs the microphone routed as well: Teams records from
            # the OUTPUT half of the cable we render our English into. With
            # both flows routed, Teams can be left on "Default" for both and
            # needs no configuring at all.
            if self.config.mode != "video" and devices.virtual_mic:
                from app.audio.routing import recording_partner

                self.router.route(
                    recording_partner(devices.virtual_mic), capture=True
                )
            self._notice(
                "info",
                f"Everything is going to {target!r} while this runs - you do "
                f"not need to touch the volume mixer. It goes back when you "
                f"stop.",
            )
            if not was:
                self._notice("warning", "could not read the previous default output")
        except RoutingUnavailable as exc:
            self._notice(
                "warning",
                f"Could not route the audio automatically ({exc}). Send the "
                f"app you want translated to {target!r} in Settings > System > "
                f"Sound > Volume mixer.",
            )


    async def _start_translating(self) -> None:
        # In a thread, and not awaited. Enumerating endpoints and creating a
        # COM object takes seconds on some machines, and Start has to feel
        # immediate - the routing landing a moment later is invisible, whereas
        # a Start button that hangs is the whole experience.
        self._routing = asyncio.ensure_future(asyncio.to_thread(self._route_audio))
        p = self.config.pipeline
        voice = VoiceId(
            vendor=self.config.providers.tts,
            name=self.config.providers.urdu_voice,
            rate=RATE_S2ST,
            language="ur",
        )

        converter, learner = build_voice_clone(
            self.config,
            on_notice=lambda message: self._notice("warning", message),
        )
        self._incoming = IncomingPipeline(
            converter=converter,
            learner=learner,
            clone_max_factor=getattr(
                self.config.providers, "clone_max_factor", 0.35
            ),
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
            # Built from config HERE. This used to be left to the pipeline's
            # default Segmenter(), which silently ignored every stability and
            # max_words setting in the meeting mode - only listen mode ever
            # honoured them.
            segmenter=Segmenter(
                min_hypotheses=p.stability_hypotheses,
                stable_ms=p.stability_ms,
                max_words=p.max_words,
                soft_min_words=getattr(p, "soft_min_words", 6),
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
        self.router.restore()
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
                    # HALF DUPLEX, and only while the Urdu is actually
                    # sounding. On headphones this never triggers. On speakers
                    # it is the difference between a working call and a loop:
                    # the Urdu we play is heard by the microphone, translated
                    # back to English, and sent to the client - who then hears
                    # their own words returned to them, louder each pass. The
                    # cost is that talking over the translation is ignored,
                    # which is exactly what a speakerphone does and for the
                    # same reason. The ring is still drained, so nothing piles
                    # up to be sent late.
                    self._watch_microphone(raw.size, mic)
                    muted = raw.size and self._muted_by_playback()
                    if muted:
                        # HALF DUPLEX, and the two bugs that were hiding here.
                        #
                        # ONE: this used to do `raw = raw[:0]` and then skip
                        # feed() entirely, so the recogniser received NOTHING -
                        # not silence, nothing. Deepgram drops a stream that has
                        # had neither audio nor a text frame for ten seconds
                        # (NET-0001), and every drop loses whatever utterance was
                        # in flight and costs a reconnect. Silence of the same
                        # length keeps the stream alive and keeps the VAD's sense
                        # of time honest, at no cost: silence is what the
                        # recogniser would have been sent anyway.
                        #
                        # TWO, and this is the one that mangles names: the gate
                        # is open whenever Urdu is audible, which in a real
                        # conversation is exactly when the user starts replying.
                        # The FRONT of their sentence is what gets discarded -
                        # and the front of the sentence is where "I am John
                        # Wick" and "I work at Macrosoft" begin. A recogniser
                        # handed "...crosoft" answers "Microsoft", confidently.
                        # So how much is being thrown away is now measured and
                        # said out loud rather than left to be guessed at.
                        self._muted_samples += int(raw.size / max(1, mic.channels))
                        raw = np.zeros_like(raw)
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
                self._report_muted()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - must never die
                self._notice("warning", f"audio pump: {exc}")
            await asyncio.sleep(0.01)

    async def _watch_stalls(self) -> None:
        """Print when the event loop itself freezes, with a number.

        Every network stream this application holds dies the same way when
        the loop blocks: the recogniser receives no audio for ten seconds
        and closes with NET-0001, which then gets blamed on the microphone
        or the network. Meanwhile the pushes that were queued during the
        freeze all execute in a burst the moment it ends, so the send
        counters claim audio flowed the whole time. A synchronous call on
        the loop - a model import, a driver enumerating endpoints, a
        decoder - is invisible in every other log. This makes it visible.
        """
        while True:
            before = time.monotonic()
            await asyncio.sleep(1.0)
            stall = time.monotonic() - before - 1.0
            if stall > 0.5:
                print(
                    f"[DEBUG ENGINE] the audio loop was blocked for "
                    f"{stall:.1f}s by a synchronous call - if NET-0001 "
                    f"follows, THIS was the cause, not the network",
                    flush=True,
                )

    #: How long to translate before saying that nothing is arriving. See the
    #: same constant in listen.py - both entry points need this, because the
    #: failure is in the routing, which neither of them can see.
    SILENCE_NOTICE_S = 60.0

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
            # INFO, not a warning, and worded as what it actually means. The
            # old wording - "there is no speech in it, either the application
            # is muted or it is playing to a different device" - described a
            # fault. But the overwhelmingly common cause is that nobody on the
            # other end has spoken yet, which is not a fault at all. Testing
            # alone triggers it every single run, and it sent a whole day into
            # hunting for a broken cable that measured perfect.
            self._notice("info", (
                f"Nothing has been said on the other side yet - {source!r} is "
                "connected and quiet. If someone IS talking and you cannot "
                "hear them translated, check the Speaker setting inside the "
                "meeting app."
            ))
        elif pipe.speech_ms < 2000.0:
            # A virtual cable that nothing is playing into still idles a little
            # above digital silence - this machine's measured -70.6 dB, against
            # a speech gate that trips around -69 dB. So a few frames of
            # nothing get counted as speech, and the old wording then blamed
            # the API key for it. Under two seconds in twelve is not somebody
            # talking; it is a quiet wire.
            self._notice("warning", (
                f"Only faint noise is arriving from {source!r} - not speech. "
                "Nothing is playing into that device: check the Speaker "
                "setting inside the meeting app, not just Windows."
            ))
        else:
            self._notice("warning", (
                "The recogniser has had several seconds of speech and returned "
                "nothing. If the client is definitely talking, the audio may be "
                "too quiet to recognise - check the level with "
                "tools/check_deepgram.py."
            ))
        self._said_why = True

    def _muted_by_playback(self) -> bool:
        """Should the microphone be ignored because we are speaking?

        HALF_DUPLEX=0 in the environment turns this off without editing
        settings.json, which is what you want when the question being asked is
        "is half duplex eating my sentences?" - on headphones there is no
        acoustic path from the earpiece to the microphone, so there is nothing
        for it to protect against and everything for it to cost.

        Note what this does NOT check: which output device is in use. The
        setting's own documentation claims it is "harmless on headphones, where
        it never triggers". That was never true - it triggers on any output,
        because all it looks at is whether Urdu is currently audible.
        """
        if os.environ.get("HALF_DUPLEX", "").strip() in ("0", "off", "false"):
            return False
        if not getattr(self.config.devices, "half_duplex", True):
            return False
        pipe = self._incoming
        return bool(pipe is not None and getattr(pipe, "is_speaking", False))

    #: Microphone watchdog state. See _watch_microphone.
    _mic_last_audio_at = 0.0
    _mic_gap_reported = False
    _mic_fed_samples = 0

    def _watch_microphone(self, got: int, mic) -> None:
        """Say when the microphone stops delivering, because nothing else does.

        THE FAILURE THIS MAKES VISIBLE
        ------------------------------
        The outgoing pump only calls feed() when the microphone ring has
        something in it. If the ring goes empty - the capture stream died, the
        device was taken by another application, the routing moved underneath
        it, or "Microphone" was resolved to a silent virtual cable - then
        nothing is fed, nothing is pushed, and about ten seconds later Deepgram
        closes the stream with NET-0001 because it has received neither audio
        nor a keepalive.

        Read from the outside, that looks exactly like a recogniser that cannot
        understand you: some words get through, then a stream drop takes the
        utterance that was in flight, then the next few words land on a fresh
        stream with no context. It is not a recognition problem at all, and
        until this printed there was no way to tell the two apart.

        `_watch_for_silence` does not cover this: it watches the INCOMING
        pipeline only, waits sixty seconds, and reports into a bubble rather
        than the console.
        """
        if got:
            self._mic_fed_samples += int(got / max(1, getattr(mic, "channels", 1)))
            if self._mic_gap_reported:
                gap = now() - self._mic_last_audio_at
                print(
                    f"[DEBUG OUTGOING] microphone recovered after {gap:.1f}s",
                    flush=True,
                )
                self._mic_gap_reported = False
            self._mic_last_audio_at = now()
            return

        if not self._mic_last_audio_at:
            self._mic_last_audio_at = now()
            return
        if self._mic_gap_reported:
            return
        gap = now() - self._mic_last_audio_at
        if gap < 3.0:
            return
        self._mic_gap_reported = True
        rate = getattr(mic, "rate", 0) or 48_000
        name = getattr(getattr(mic, "device", None), "name", "?")
        print(
            f"[DEBUG OUTGOING] microphone {name!r} has delivered NO audio for "
            f"{gap:.1f}s (total fed this session: "
            f"{self._mic_fed_samples / rate:.1f}s). The recogniser is being "
            f"starved - this is what produces NET-0001, not a recognition "
            f"fault. Check that this is the right device and that nothing else "
            f"holds it.",
            flush=True,
        )

    #: How much microphone audio half duplex has thrown away, and when it was
    #: last reported. Reported rather than merely counted: a gate that silently
    #: discards the user's speech is indistinguishable, from the outside, from
    #: a recogniser that cannot hear.
    _muted_samples = 0
    _muted_reported_at = 0.0

    def _report_muted(self) -> None:
        if not self._muted_samples:
            return
        mic = self.ports.mic_source
        rate = getattr(mic, "rate", 0) or 48_000
        if not self._muted_reported_at:
            # First discard of the session: start the clock, do not report an
            # interval measured from the epoch.
            self._muted_reported_at = now()
            return
        elapsed = now() - self._muted_reported_at
        if elapsed < 10.0:
            return
        seconds = self._muted_samples / rate
        self._muted_samples = 0
        self._muted_reported_at = now()
        if seconds < 0.2:
            return
        print(
            f"[DEBUG OUTGOING] half duplex discarded {seconds:.1f}s of "
            f"microphone audio in the last {elapsed:.0f}s - if you are on "
            f"headphones, run with HALF_DUPLEX=0",
            flush=True,
        )

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
