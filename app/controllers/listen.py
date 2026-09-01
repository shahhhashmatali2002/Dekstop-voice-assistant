"""Listen mode: one way, any application. English in, Urdu out.

Spec reference: the incoming half of §2.1 and §8.2, the wire discipline of
§2.5, the feedback rules of §5.6, and §5.2 / D7 for what this works around.

WHAT THIS IS
------------
Meeting mode owns four endpoints and translates in both directions. This owns
two and translates in one:

    YouTube, a Teams call, a recording, anything at all
        |
        |  Windows per-application output routing
        v
    CABLE Input          a virtual cable; nothing audible is attached to it
        |
        |  WASAPI loopback capture
        v
    [ passthrough wire ] ------------------------------> headphones
        |                                                (resting state)
        +--> VAD -> English ASR -> stability + segmentation
                 -> translation -> streaming Urdu TTS --> headphones

No microphone, no second cable, no outgoing pipeline, and nothing that knows
what Teams is. The source is a device, and any application that can be pointed
at that device is supported without further work.

WHY THE VIRTUAL CABLE IS NOT OPTIONAL
-------------------------------------
"I want to hear only the Urdu" is a routing requirement, not a mixing one.
What we capture *is* the render stream of the endpoint we capture from, so
silencing that endpoint silences our own input along with it. There is no gain
setting that removes the English from your ears and leaves it in ours.

The one mechanism that separates the two is process loopback, which captures a
chosen process's audio whether or not it is being rendered anywhere. That is
D7, deferred, and PyAudioWPatch does not expose it. Until it exists the source
application must render into a cable nobody is listening to, and this module
plays the translation into the device you are actually wearing.

THE RESTING STATE
-----------------
The wire is the reason the cable is not a black hole. Before Start, after
Stop, and after any failure, the cable is pumped straight to your headphones
and you hear the original English. Start mutes that wire and the Urdu takes
over the same sink; Stop unmutes it. Every failure path lands back on English
audio rather than on silence, which for a listener is the difference between
a degraded tool and a broken one (§2.5, §14.1).

Exactly one consumer reads the capture ring at any moment - the wire while it
is unmuted, the pipeline while it is running - because a ring buffer read is
destructive and two readers would each get half the frames. That is why
mixing a little of the original underneath the Urdu is NOT offered here:
doing it properly needs a summing sink, not a second reader (see README).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass

from app.audio import convert
from app.audio.devices import (
    AudioUnavailable,
    DeviceManager,
    DeviceRef,
    looks_virtual,
)
from app.audio.formats import RATE_ASR, RATE_S2ST
from app.audio.passthrough import PassthroughWire
from app.audio.streams import CaptureSource, RenderSink
from app.controllers.drift import DriftController
from app.controllers.incoming import IncomingPipeline
from app.providers.factory import build_voice_clone
from app.controllers.segmenter import Segmenter
from app.controllers.vad import VoiceActivityGate
from app.models.config import Config
from app.models.telemetry import MetricsBus, TraceFactory
from app.models.terminology import Terminology
from app.providers.base import VoiceId

#: Status values. Deliberately not `ModeMachine`: that models two directions
#: and a direction that does not exist here would sit permanently in a state
#: nothing can leave.
IDLE = "idle"          # wire up, original English audible, not translating
LISTENING = "listening"  # translating; you hear Urdu only
PAUSED = "paused"      # provider connections warm, English audible again
DEGRADED = "degraded"  # a provider failed; English audible again


# ----------------------------------------------------------------------
# presets
# ----------------------------------------------------------------------

#: Fluency and lag are the same dial, and the right setting depends entirely
#: on what you are listening to.
#:
#: The pipeline emits a unit when the recogniser's text has been stable for
#: `stability_ms`, or when it reaches `max_words`. Short units reach your ears
#: sooner and are synthesised in isolation, so the voice restarts its prosody
#: every few words - accurate, and it sounds like someone reading a list.
#: Longer units are spoken as whole thoughts, which is what fluent sounds
#: like, and they arrive later.
#:
#: The drift thresholds are the other half. Past `drift_comfortable_ms` of
#: measured lag the audio is time-stretched to catch up, and the stretch is
#: heard as a metallic edge long before it is heard as speed. In a live call
#: that trade is worth it. Watching a recording it is not: nothing is waiting
#: on you, so let the lag sit where it lands and keep the voice clean.
PRESETS = {
    "fluent": dict(
        stability_ms=500.0, max_words=28, soft_min_words=14, vad_hangover_ms=600.0,
        playback_preroll_ms=400.0,
        drift_comfortable_ms=4000.0, drift_stretch_ms=12000.0,
        drift_aggressive_ms=25000.0, drift_max_rate=1.08,
    ),
    "balanced": dict(
        stability_ms=300.0, max_words=18, soft_min_words=8, vad_hangover_ms=400.0,
        playback_preroll_ms=300.0,
        drift_comfortable_ms=2000.0, drift_stretch_ms=6000.0,
        drift_aggressive_ms=12000.0, drift_max_rate=1.12,
    ),
    "fast": dict(
        stability_ms=150.0, max_words=10, soft_min_words=5, vad_hangover_ms=300.0,
        playback_preroll_ms=200.0,
        drift_comfortable_ms=800.0, drift_stretch_ms=2500.0,
        drift_aggressive_ms=5000.0, drift_max_rate=1.20,
    ),
}

PRESET_NOTES = {
    "fluent": "whole thoughts, natural voice, several seconds behind - for video",
    "balanced": "a compromise; start here if you are unsure",
    "fast": "short phrases, closest to real time - for a live conversation",
}


def apply_preset(config: Config, name: str) -> str:
    """Write one preset into the pipeline settings. Returns what it did."""
    values = PRESETS.get(name)
    if values is None:
        raise KeyError(f"unknown preset {name!r}; try {', '.join(PRESETS)}")
    for field_name, value in values.items():
        setattr(config.pipeline, field_name, value)
    return f"{name} - {PRESET_NOTES[name]}"


# ----------------------------------------------------------------------
# guards
# ----------------------------------------------------------------------


def feedback_problems(source: DeviceRef | None, output: DeviceRef | None) -> list[str]:
    """Refuse a binding that would feed our own output back into our input.

    Spec reference: §5.6 F2. This check happens before a stream opens, because
    the failure is not a wrong answer - it is Urdu being re-recognised as
    though it were the speaker, translated again, and played louder every
    pass.
    """
    found: list[str] = []
    name = lambda d: getattr(d, "name", None)  # noqa: E731

    if source is not None and output is not None and name(source) == name(output):
        found.append(
            f"Feedback loop: the source and the output are both "
            f"{name(source)!r}. We capture everything that device plays and "
            "we play the Urdu into it, so our own translation would be "
            "captured and translated again, louder each time. The source has "
            "to be a virtual cable you cannot hear - route the application "
            "to 'CABLE Input' and leave the output on your headphones."
        )

    if source is not None and not getattr(source, "can_render", True):
        found.append(
            f"{name(source)!r} is a capture endpoint, not a playback one. The "
            "source is the device the application plays into; we ask Windows "
            "for its loopback ourselves."
        )
    return found


# ----------------------------------------------------------------------
# ports
# ----------------------------------------------------------------------




def cable_scan(devices) -> str:
    """Is a virtual cable present on ANY host API, not just WASAPI?

    This separates two failures that produce the same complaint. `find` only
    searches WASAPI endpoints, so a cable that Windows knows about but is not
    exposing over WASAPI - disabled in the Sound control panel is the usual
    cause - looks exactly like a cable that was never installed. The fix is
    completely different in each case, so it is worth one extra scan.
    """
    hits = [d for d in devices if looks_virtual(d.name)]
    if not hits:
        return (
            "  No virtual audio device on any host API. The driver is not "
            "installed, or it installed without elevation and registered "
            "nothing."
        )

    apis = sorted({d.host_api for d in hits})
    names = sorted({d.name for d in hits})
    listing = "\n".join(f"    - {n}" for n in names)
    return (
        "  A virtual audio device IS present, on: " + ", ".join(apis) + "\n"
        + listing
        + "\n  If it is missing from the WASAPI list above, Windows has the "
        "driver\n  but is not exposing that endpoint: open mmsys.cpl, "
        "right-click in the\n  Playback tab, tick 'Show Disabled Devices', "
        "and enable it."
    )


def _playback_hint(manager: DeviceManager) -> str:
    """What playback endpoints actually exist, for when the named one does not.

    "no WASAPI endpoint matching 'CABLE Input'" is true but unhelpful: it does
    not say whether the name is wrong or the driver is missing, and WASAPI is
    not a thing the reader chose - it is simply how Windows exposes every
    endpoint, virtual cables included. Listing what is there answers both
    questions at once.
    """
    try:
        names = [d.name for d in manager.wasapi_devices() if d.can_render]
    except Exception:  # pragma: no cover - enumeration itself failed
        return ""
    if not names:
        return "Windows is reporting no playback endpoints at all."

    listing = "\n".join(f"      - {n}" for n in names)
    virtual = [n for n in names if "cable" in n.lower() or "voicemeeter" in n.lower()]
    if virtual:
        tail = (
            "\n    That list does contain a virtual cable. Pass its name "
            f"exactly as shown, e.g. --source {virtual[0]!r} (a case-insensitive "
            "fragment is enough)."
        )
    else:
        tail = (
            "\n    None of those is a virtual cable, so VB-CABLE is not "
            "installed on this machine yet. Install it from vb-audio.com/Cable "
            "and reboot - the driver only registers its endpoints on a restart. "
            "A virtual cable is not an alternative to WASAPI: it is a driver "
            "that adds a sound card Windows then exposes over WASAPI like any "
            "other."
        )
    return f"    Playback endpoints Windows can see:\n{listing}{tail}"


@dataclass
class ListenPorts:
    """The two endpoints and the one wire this mode drives."""

    source: CaptureSource          # loopback of the routed application's output
    output: RenderSink             # your headphones
    wire: PassthroughWire          # source -> output, the resting state
    source_device: DeviceRef       # the render half, for display
    output_device: DeviceRef

    def start(self) -> None:
        self.source.start()
        self.output.start()
        self.wire.start()

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.wire.stop()
        for endpoint in (self.source, self.output):
            with contextlib.suppress(Exception):
                endpoint.stop()

    def health(self) -> dict:
        return {"wire": self.wire.health()}

    def describe(self) -> str:
        return f"{self.source_device.name}  ->  {self.output_device.name}"


def build_listen_ports(
    config: Config,
    manager: DeviceManager,
) -> tuple[ListenPorts | None, list[str], list[str]]:
    """Resolve both endpoints and open them.

    Returns `(ports, problems, notes)`. Problems are collected rather than
    raised, so the caller can show everything that needs fixing in one pass
    (§14.1). `ports` is None whenever there is a problem.
    """
    problems: list[str] = []
    notes: list[str] = []
    listen = config.listen

    source_dev: DeviceRef | None = None
    if not listen.source:
        problems.append(
            "Source not set. This is the playback device the application is "
            "routed to - a virtual audio cable, normally 'CABLE Input'. There "
            "is no sensible default: anything you can hear would defeat the "
            "point, and it would howl."
        )
    else:
        try:
            source_dev = manager.find(listen.source, capture=False)
        except AudioUnavailable as exc:
            problems.append(f"Source: {exc}")
            hint = _playback_hint(manager)
            if hint:
                problems.append(hint)

    output_dev: DeviceRef | None = None
    if not listen.output:
        try:
            output_dev = manager.default_output()
            notes.append(f"Output: using system default - {output_dev.name}")
        except AudioUnavailable as exc:
            problems.append(f"Output: no system default ({exc})")
    else:
        try:
            output_dev = manager.find(listen.output, capture=False)
        except AudioUnavailable as exc:
            # A device that has since been unplugged must not stop the tool.
            with contextlib.suppress(AudioUnavailable):
                output_dev = manager.default_output()
                notes.append(
                    f"Output: {listen.output!r} not found, using system "
                    f"default - {output_dev.name}"
                )
            if output_dev is None:
                problems.append(f"Output: {exc}")

    problems.extend(feedback_problems(source_dev, output_dev))

    loopback: DeviceRef | None = None
    if source_dev is not None and not problems:
        try:
            loopback = manager.loopback_for_output(source_dev)
        except AudioUnavailable as exc:
            problems.append(f"Source loopback: {exc}")

    if problems or loopback is None or output_dev is None:
        return None, problems, notes

    source = CaptureSource(manager, loopback)
    output = RenderSink(manager, output_dev)
    # True silence, not comfort noise: nothing downstream is listening for a
    # live stream, and a noise floor in your headphones is just hiss (§5.7
    # applies to the path feeding Teams, not to this one).
    wire = PassthroughWire(
        source, output, name="source->output", comfort_noise_dbfs=None
    )
    ports = ListenPorts(source, output, wire, source_dev, output_dev)
    ports.start()
    return ports, problems, notes


# ----------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------


class ListenEngine:
    """Owns the asyncio loop and the single incoming pipeline.

    Threading follows §4.2: one engine thread running one loop, and the UI
    calls in through `submit`, which is thread-safe. Nothing here touches a Qt
    object; events go out through plain callables the caller marshals.
    """

    def __init__(
        self,
        config: Config,
        ports: ListenPorts,
        providers: dict,
        *,
        terminology: Terminology | None = None,
        on_transcript=None,
        on_notice=None,
        on_status=None,
    ) -> None:
        self.config = config
        self.ports = ports
        self.providers = providers
        self.terminology = terminology or Terminology()

        self.metrics = MetricsBus()
        self.traces = TraceFactory()

        self.on_transcript = on_transcript
        self.on_notice = on_notice
        self.on_status = on_status

        self._status = IDLE
        #: When listening began, and whether we have already explained a
        #: silence - the explanation is worth saying once, not every 10 ms.
        self._listening_since = 0.0
        self._said_why = False
        self._detail = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._pipe: IncomingPipeline | None = None
        self._pump_task: asyncio.Task | None = None
        self._closing = False

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._status

    @property
    def listening(self) -> bool:
        return self._status == LISTENING

    def _set_status(self, status: str, detail: str = "") -> None:
        if status != self._status:
            # Every transition restarts the clock: resuming from a pause
            # deserves the same explanation as a first start, and a source
            # that was fine before may not be fine now.
            self._listening_since = time.monotonic() if status == LISTENING else 0.0
            self._said_why = False
        self._status = status
        self._detail = detail
        if self.on_status:
            with contextlib.suppress(Exception):
                self.on_status({"status": status, "detail": detail})

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bring the loop up. Does NOT start translating."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="listen-engine", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)
        self._set_status(IDLE, "passthrough - you are hearing the original")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._pump_task = self._loop.create_task(self._audio_pump())
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
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

    def shutdown(self) -> None:
        """Stop everything, in order, exactly once."""
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
        self.ports.stop()

    async def _shutdown_async(self) -> None:
        await self._stop_listening()
        if self._pump_task is not None:
            self._pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pump_task
            self._pump_task = None

    def submit(self, coro):
        if self._loop is None:
            raise RuntimeError("engine is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------
    # controls - callable from any thread
    # ------------------------------------------------------------------

    def start_listening(self):
        return self.submit(self._start_listening())

    def pause(self):
        return self.submit(self._pause())

    def stop_listening(self):
        return self.submit(self._stop_listening())

    def toggle(self):
        if self._status == LISTENING:
            return self.pause()
        return self.start_listening()

    # ------------------------------------------------------------------

    def _build_pipeline(self) -> IncomingPipeline:
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
        return IncomingPipeline(
            converter=converter,
            learner=learner,
            clone_max_factor=getattr(
                self.config.providers, "clone_max_factor", 0.35
            ),
            asr=self.providers["asr"],
            translator=self.providers["translator"],
            tts=self.providers["tts"],
            voice=voice,
            sink=self.ports.output,
            metrics=self.metrics,
            traces=self.traces,
            terminology=self.terminology,
            gate=VoiceActivityGate(
                rate=RATE_ASR,
                hangover_ms=p.vad_hangover_ms,
                preroll_ms=p.vad_preroll_ms,
            ),
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
            on_degrade=self._degrade,
        )

    async def _start_listening(self) -> None:
        if self._pipe is not None and self._pipe.running:
            # Resuming from Pause. The connections are already warm, so this
            # is only a question of who owns the sink - and of throwing away
            # what accumulated in the ring while the wire owned it, which is
            # audio you have already heard in English.
            with contextlib.suppress(Exception):
                self.ports.source.ring.discard()
            self.ports.wire.mute()
            self._set_status(LISTENING, "Urdu only")
            return

        if self._pipe is None:
            self._pipe = self._build_pipeline()

        try:
            await self._pipe.start()
        except Exception as exc:
            # The wire is untouched, so the original English keeps playing.
            self._pipe = None
            self._notice("error", f"could not start translating: {exc}")
            self._set_status(IDLE, str(exc))
            return

        # Drop whatever accumulated in the capture ring while we were only
        # passing audio through: half a second of stale speech at the front of
        # the stream is half a second of latency we never get back.
        with contextlib.suppress(Exception):
            self.ports.source.ring.discard()

        # Only now does the English stop. Muting before the pipeline is up
        # would put a gap of silence in front of the first Urdu word.
        self.ports.wire.mute()
        self._set_status(LISTENING, "Urdu only")

    async def _pause(self) -> None:
        # Connections stay open and warm; resuming must cost nothing (§11.3).
        self.ports.wire.unmute()
        self._set_status(PAUSED, "original audio - press Start to resume")

    async def _stop_listening(self) -> None:
        self.ports.wire.unmute()
        if self._pipe is not None:
            with contextlib.suppress(Exception):
                await self._pipe.stop()
            self._pipe = None
        self._set_status(IDLE, "passthrough - you are hearing the original")

    # ------------------------------------------------------------------
    # the pump: capture ring -> pipeline
    # ------------------------------------------------------------------

    async def _audio_pump(self) -> None:
        """Move captured audio into the pipeline, forever.

        While nothing is translating there is nowhere to deliver to and the
        wire is doing the work, so this simply idles - it must not consume the
        ring, or the wire would get every other frame.
        """
        source = self.ports.source
        while True:
            try:
                # LISTENING, not merely "the pipeline exists": while paused
                # the wire is unmuted and owns the ring. Reading it here too
                # would give each of them every other frame.
                if (
                    self._status == LISTENING
                    and self._pipe is not None
                    and self._pipe.running
                ):
                    raw = source.ring.read_available()
                    if raw.size:
                        self._pipe.feed(
                            convert.endpoint_to_pipeline(
                                raw, source.rate, source.channels, RATE_ASR
                            )
                        )
                self._publish_health()
                self._watch_for_silence()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - must never die
                self._notice("warning", f"audio pump: {exc}")
            await asyncio.sleep(0.01)

    #: How long to listen before saying that nothing is arriving. Long enough
    #: that a pause between two sentences is never mistaken for a dead source,
    #: short enough that a user who has mis-routed the audio is not left
    #: staring at an empty window wondering whether it is working.
    SILENCE_NOTICE_S = 12.0

    def _watch_for_silence(self) -> None:
        """Say why nothing is happening, once, instead of nothing at all.

        Four completely different failures look identical from the outside -
        an empty transcript - and they have four different fixes. This tells
        them apart by where the pipeline's counters stop, and says so in the
        transcript window where the user is already looking.
        """
        if self._said_why or self._status != LISTENING or self._pipe is None:
            return
        started = self._listening_since
        if not started or time.monotonic() - started < self.SILENCE_NOTICE_S:
            return

        pipe = self._pipe
        source = getattr(self.ports.source_device, "name", "the source")

        # Anything coming back from the recogniser means the whole path works,
        # whatever the counters before it say, so there is nothing to explain.
        # Checked first because it is the only one of these that is proof
        # rather than a threshold.
        if pipe.hypotheses or pipe.units:
            return

        if pipe.fed_ms < 500.0:
            self._notice("warning", (
                f"Nothing is arriving from {source!r}. The device is open but "
                "silent, which means no application is playing to it - check "
                "the app's output in Volume mixer, and that another copy of "
                "this program is not already holding the device."
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

    def _publish_health(self) -> None:
        for name, obj in (
            ("source", self.ports.source),
            ("output", self.ports.output),
        ):
            with contextlib.suppress(Exception):
                self.metrics.set_audio_health(name, obj.health())

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------

    def _transcript(self, payload: dict) -> None:
        if self.on_transcript:
            with contextlib.suppress(Exception):
                self.on_transcript(payload)

    def _notice(self, severity: str, message: str) -> None:
        if self.on_notice:
            with contextlib.suppress(Exception):
                self.on_notice({"severity": severity, "message": message})

    def _degrade(self, kind: str, why: str) -> None:
        """A provider failed. Fall back to the original audio, never silence."""
        if kind == "impair":
            # Synthesis is gone but recognition is not: subtitles still work,
            # so this is a warning, not a fallback.
            self._notice("warning", why)
            self._set_status(LISTENING, f"text only - {why}")
            return

        self.ports.wire.unmute()

        # Retire it rather than leave it half-alive. A pipeline that has
        # degraded has a dead provider stream and a segmenter holding half a
        # sentence; restarting *that* object would carry both forward into the
        # next attempt. Its tasks are closed on the loop, out of band, because
        # this callback is running inside one of them.
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(self._retire(pipe))

        self._set_status(DEGRADED, why)
        self._notice(
            "error",
            f"Translation stopped ({why}). You are hearing the original "
            "audio again. Press Start to try again.",
        )

    async def _retire(self, pipe: IncomingPipeline) -> None:
        with contextlib.suppress(Exception):
            await pipe.stop()

    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "status": self._status,
            "detail": self._detail,
            "devices": self.ports.describe(),
            "metrics": self.metrics.snapshot(),
        }
