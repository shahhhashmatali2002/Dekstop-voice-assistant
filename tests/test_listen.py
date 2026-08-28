"""Listen mode: the wire discipline, the guards, and one end-to-end pass.

No devices, no network, no Qt. The providers are fakes, but the pipeline,
segmenter, VAD, ring buffers and format conversion are all the real ones, so
this is the first test in the project that runs English audio in at one end
and gets Urdu audio out at the other.

The invariant under test throughout is the one a listener actually notices:

    you are never left in silence.

Before Start, after Stop, while Paused, and after any provider failure, the
wire is unmuted and the original English is playing. It is muted in exactly
one state - translating - and that is the only state where the Urdu owns the
sink.
"""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np
import pytest

from app.audio.devices import DeviceRef
from app.audio.formats import RATE_ASR
from app.audio.ring import RingBuffer
from app.controllers import listen as L
from app.models.config import Config, ListenConfig
from app.providers.base import (
    AsrStream,
    PcmChunk,
    ProviderUnavailable,
    SpeechToText,
    TextToSpeech,
    TranscriptDelta,
    TranslatedUnit,
    Translator,
)

RENDER = dict(max_input_channels=0, max_output_channels=2, default_rate=48_000,
              host_api="WASAPI")
CAPTURE = dict(max_input_channels=2, max_output_channels=0, default_rate=48_000,
               host_api="WASAPI")


def dev(name: str, index: int = 1, **kw) -> DeviceRef:
    return DeviceRef(name=name, index=index, **(kw or RENDER))


# ----------------------------------------------------------------------
# the feedback guard - §5.6 F2
# ----------------------------------------------------------------------


def test_source_and_output_must_not_be_the_same_device():
    problems = L.feedback_problems(dev("Headphones"), dev("Headphones"))
    assert len(problems) == 1
    assert "Feedback loop" in problems[0]


def test_the_message_says_what_to_do_about_it():
    (msg,) = L.feedback_problems(dev("Speakers"), dev("Speakers"))
    assert "CABLE Input" in msg  # a diagnosis without a fix is half a message


def test_source_must_be_a_playback_device():
    problems = L.feedback_problems(dev("Microphone", **CAPTURE), dev("Headphones", 2))
    assert any("capture endpoint" in p for p in problems)


def test_a_cable_and_a_headset_are_accepted():
    assert L.feedback_problems(dev("CABLE Input"), dev("Headphones", 2)) == []


def test_missing_devices_are_not_a_feedback_problem():
    # They are a separate, clearer complaint from build_listen_ports.
    assert L.feedback_problems(None, None) == []


# ----------------------------------------------------------------------
# config
# ----------------------------------------------------------------------


def test_listen_settings_survive_a_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    cfg = Config()
    cfg.listen = ListenConfig(source="CABLE Input", output="Headset", autostart=True)
    cfg.save(path)

    again = Config.load(path)
    assert again.listen.source == "CABLE Input"
    assert again.listen.output == "Headset"
    assert again.listen.autostart is True


def test_a_settings_file_from_before_listen_mode_still_loads(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"devices": {"microphone": "Headset"}}', encoding="utf-8")
    cfg = Config.load(path)
    assert cfg.devices.microphone == "Headset"
    assert cfg.listen.source == ""  # defaulted, not crashed


# ----------------------------------------------------------------------
# fakes
# ----------------------------------------------------------------------


class FakeWire:
    def __init__(self) -> None:
        self._muted = False
        self.started = False
        self.mutes = 0
        self.unmutes = 0

    @property
    def muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        self._muted = True
        self.mutes += 1

    def unmute(self) -> None:
        self._muted = False
        self.unmutes += 1

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def health(self) -> dict:
        return {"muted": self._muted}


class FakeEndpoint:
    """Stands in for a CaptureSource or a RenderSink: a real ring, no device.

    A render endpoint can also be told to DRAIN, which is what makes it behave
    like the hardware: a real WASAPI callback empties the ring every period
    whether or not anything is filling it. A fake sink that never drains fills
    up and stays full, which is not a quieter version of reality - it is the
    opposite of it, and it hides exactly the bugs this file exists to catch.
    """

    def __init__(self, rate: int = 48_000, channels: int = 2) -> None:
        self.rate = rate
        self.channels = channels
        self.ring = RingBuffer(rate * channels // 2)  # 500 ms, like RenderSink
        self.written: list[np.ndarray] = []
        self.stopped = False
        self._draining: threading.Thread | None = None

    def drain_like_hardware(self) -> None:
        """Consume the ring in real time on a background thread."""
        if self._draining is not None:
            return

        def pump() -> None:
            per_tick = int(self.rate * self.channels * 0.010)
            buf = np.empty(per_tick, dtype=np.int16)
            while not self.stopped:
                self.ring.read_into(buf)
                time.sleep(0.010)

        self._draining = threading.Thread(target=pump, daemon=True)
        self._draining.start()

    def write(self, pcm: np.ndarray) -> int:
        self.written.append(pcm)
        return self.ring.write(pcm)

    def flush(self) -> int:
        return self.ring.discard()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True

    def health(self) -> dict:
        return self.ring.health()

    @property
    def played_samples(self) -> int:
        return sum(a.size for a in self.written)


class FakeAsrStream(AsrStream):
    def __init__(self, texts: list[str], *, drop_after: int | None = None) -> None:
        self.texts = texts
        self.drop_after = drop_after
        self.pushed: list[np.ndarray] = []
        self.closed = False

    async def push_audio(self, pcm: np.ndarray) -> None:
        self.pushed.append(pcm)

    async def hypotheses(self):
        for index, text in enumerate(self.texts):
            await asyncio.sleep(0.01)
            if self.drop_after is not None and index >= self.drop_after:
                raise ProviderUnavailable("socket closed by the provider")
            yield TranscriptDelta(text, is_final=True)
        while True:  # a real stream stays open after its last result
            if self.drop_after is not None:
                raise ProviderUnavailable("socket closed by the provider")
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        self.closed = True

    @property
    def pushed_samples(self) -> int:
        return sum(a.size for a in self.pushed)


class FakeAsr(SpeechToText):
    name = "fake-asr"

    def __init__(
        self,
        texts: list[str] | None = None,
        fail: bool = False,
        *,
        drop_after: int | None = None,
    ) -> None:
        self.texts = texts or []
        self.fail = fail
        self.drop_after = drop_after  # the stream dies after N hypotheses
        self.stream: FakeAsrStream | None = None
        self.streams: list = []
        self.configs: list = []

    async def open(self, cfg):
        self.configs.append(cfg)
        if self.fail:
            raise ProviderUnavailable("recognition is down")
        # Only the first stream drops; a reconnect gets a healthy one.
        drop = self.drop_after if len(self.streams) == 0 else None
        self.stream = FakeAsrStream(self.texts, drop_after=drop)
        self.streams.append(self.stream)
        return self.stream


class FakeTranslator(Translator):
    name = "fake-mt"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def translate(self, unit, ctx, terms=None) -> TranslatedUnit:
        self.seen.append(unit.text)
        assert unit.source_language == "en" and unit.target_language == "ur"
        return TranslatedUnit(text="[ur] " + unit.text, source=unit)


class FakeTts(TextToSpeech):
    name = "fake-tts"

    def __init__(self, rate: int = 24_000, chunks: int = 3, delay: float = 0.0) -> None:
        self.rate = rate
        self.chunks = chunks  # each one is 100 ms
        self.delay = delay    # stands in for a real synthesiser's think time
        self.spoken: list[str] = []
        self.on_chunk = None  # called with the chunk index before each yield

    async def synthesize(self, text: str, voice):
        self.spoken.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        for index in range(self.chunks):
            if self.on_chunk is not None:
                self.on_chunk(index)
            yield PcmChunk(np.zeros(self.rate // 10, dtype=np.int16), self.rate)
            await asyncio.sleep(0)
        yield PcmChunk(np.empty(0, dtype=np.int16), self.rate, final=True)


def build(texts=None, *, asr_fails=False, tts_chunks=3, tts_delay=0.0,
          asr_drops_after=None, **pipeline):
    """An engine wired to fakes, plus the parts a test needs to inspect."""
    cfg = Config()
    cfg.listen = ListenConfig(source="CABLE Input", output="Headphones")
    # Emit on the first hypothesis: these tests are about wiring, and the
    # stability window is tested exhaustively in test_pipeline.py.
    cfg.pipeline.stability_hypotheses = 1
    cfg.pipeline.stability_ms = 0.0
    for key, value in pipeline.items():
        setattr(cfg.pipeline, key, value)

    source = FakeEndpoint()
    output = FakeEndpoint()
    output.drain_like_hardware()
    wire = FakeWire()
    ports = L.ListenPorts(
        source=source,
        output=output,
        wire=wire,
        source_device=dev("CABLE Input"),
        output_device=dev("Headphones", 2),
    )
    providers = {
        "asr": FakeAsr(texts, fail=asr_fails, drop_after=asr_drops_after),
        "translator": FakeTranslator(),
        "tts": FakeTts(chunks=tts_chunks, delay=tts_delay),
    }
    events: list[dict] = []
    statuses: list[dict] = []
    notices: list[dict] = []
    engine = L.ListenEngine(
        cfg,
        ports,
        providers,
        on_transcript=events.append,
        on_status=statuses.append,
        on_notice=notices.append,
    )
    return engine, ports, providers, events, statuses, notices


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def loud(ms: int, rate: int = 48_000, channels: int = 2) -> np.ndarray:
    """Interleaved stereo tone well above the VAD's energy threshold."""
    n = int(rate * ms / 1000)
    t = np.arange(n) / rate
    mono = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    return np.repeat(mono, channels)


@pytest.fixture
def engine_fixture():
    made = []

    def factory(*a, **kw):
        bundle = build(*a, **kw)
        made.append(bundle[0])
        return bundle

    yield factory
    for engine in made:
        engine.shutdown()


# ----------------------------------------------------------------------
# the resting state - §2.5
# ----------------------------------------------------------------------


def test_the_wire_is_open_before_anything_translates(engine_fixture):
    engine, ports, *_ = engine_fixture()
    engine.start()
    assert engine.status == L.IDLE
    assert ports.wire.muted is False  # the original English is playing


def test_start_mutes_the_wire_so_only_urdu_is_heard(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    assert engine.status == L.LISTENING
    assert ports.wire.muted is True


def test_stop_ends_translation_not_audio(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine.stop_listening().result(timeout=5)
    assert engine.status == L.IDLE
    assert ports.wire.muted is False


def test_pause_returns_the_original_audio(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine.pause().result(timeout=5)
    assert engine.status == L.PAUSED
    assert ports.wire.muted is False


def test_resuming_from_pause_costs_no_new_connection(engine_fixture):
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine.pause().result(timeout=5)
    engine.toggle().result(timeout=5)
    assert engine.status == L.LISTENING
    assert ports.wire.muted is True
    assert len(providers["asr"].configs) == 1  # opened once, not twice


def test_a_provider_that_will_not_start_leaves_the_english_playing(engine_fixture):
    engine, ports, _p, _e, _s, notices = engine_fixture(asr_fails=True)
    engine.start()
    engine.start_listening().result(timeout=5)
    assert engine.status == L.IDLE
    assert ports.wire.muted is False
    assert any(n["severity"] == "error" for n in notices)


def test_a_provider_that_dies_mid_stream_falls_back_to_english(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine._degrade("degrade", "recognition stream closed")
    assert engine.status == L.DEGRADED
    assert ports.wire.muted is False


def test_synthesis_failure_keeps_subtitles_running(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine._degrade("impair", "synthesis unavailable")
    # Recognition still works, so this is text-only, not a fallback.
    assert engine.status == L.LISTENING
    assert ports.wire.muted is True


def test_the_asr_is_asked_for_english(engine_fixture):
    engine, _ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    assert providers["asr"].configs[0].language == "en"
    assert providers["asr"].configs[0].rate == RATE_ASR


# ----------------------------------------------------------------------
# the pump - one reader of the ring at a time
# ----------------------------------------------------------------------


def test_the_pump_leaves_the_ring_alone_while_the_wire_owns_it(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    ports.source.ring.write(loud(200))
    before = ports.source.ring.available
    time.sleep(0.2)  # several pump cycles
    assert ports.source.ring.available == before, (
        "the pump consumed audio the passthrough wire needed"
    )


def test_translating_drains_stale_audio_before_the_first_word(engine_fixture):
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    ports.source.ring.write(loud(400))
    engine.start_listening().result(timeout=5)
    assert ports.source.ring.available == 0


def test_captured_audio_reaches_the_recogniser(engine_fixture):
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))
    assert wait_for(lambda: providers["asr"].stream.pushed_samples > 0), (
        "audio written to the capture ring never reached the ASR stream"
    )


# ----------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------


def test_english_in_urdu_out(engine_fixture):
    engine, ports, providers, events, *_ = engine_fixture(
        ["The deployment failed."]
    )
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))

    assert wait_for(lambda: ports.output.played_samples > 0), (
        "no Urdu audio was ever written to the output sink"
    )
    assert providers["translator"].seen == ["The deployment failed."]
    assert providers["tts"].spoken == ["[ur] The deployment failed."]

    assert events, "nothing was published for the subtitles"
    entry = events[0]
    assert entry["direction"] == "incoming"
    assert entry["source"] == "The deployment failed."
    assert entry["translated"] == "[ur] The deployment failed."
    assert entry["dropped"] is False


def test_the_urdu_is_converted_to_the_output_device_format(engine_fixture):
    # TTS gives us 24 kHz mono; the headphones here are 48 kHz stereo.
    engine, ports, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))
    assert wait_for(lambda: ports.output.played_samples > 0)
    chunk = ports.output.written[0]
    assert chunk.dtype == np.int16
    assert chunk.size % 2 == 0, "stereo output must be an even number of samples"


def test_shutdown_closes_the_provider_stream_and_the_endpoints(engine_fixture):
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    stream = providers["asr"].stream
    engine.shutdown()
    assert stream.closed is True
    assert ports.source.stopped and ports.output.stopped
    assert ports.wire.started is False


def test_shutdown_is_idempotent(engine_fixture):
    engine, *_ = engine_fixture(["hello"])
    engine.start()
    engine.shutdown()
    engine.shutdown()  # the tray Quit and aboutToQuit both reach here


def test_pausing_hands_the_ring_back_to_the_wire(engine_fixture):
    """The bug this test exists for: two readers of one destructive ring.

    Pause leaves the provider connections open, so the pipeline is still
    "running". If the pump kept reading the capture ring on that basis, the
    wire and the pipeline would each get roughly every other frame - the
    English you are listening to would be chopped up, and the pipeline would
    keep translating half a signal into the sink you are listening to.
    """
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine.pause().result(timeout=5)

    pushed_at_pause = providers["asr"].stream.pushed_samples
    ports.source.ring.write(loud(200))
    before = ports.source.ring.available
    time.sleep(0.2)

    assert ports.source.ring.available == before, "the pump raided a paused ring"
    assert providers["asr"].stream.pushed_samples == pushed_at_pause


def test_resume_takes_the_sink_back(engine_fixture):
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine.pause().result(timeout=5)
    engine.start_listening().result(timeout=5)

    assert engine.status == L.LISTENING
    assert ports.wire.muted is True
    ports.source.ring.write(loud(400))
    assert wait_for(lambda: providers["asr"].stream.pushed_samples > 0)


def test_starting_again_after_a_failure_builds_a_fresh_pipeline(engine_fixture):
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine.submit(_degrade_on_loop(engine, "recognition stream closed")).result(5)
    assert engine.status == L.DEGRADED

    engine.start_listening().result(timeout=5)
    assert engine.status == L.LISTENING
    assert ports.wire.muted is True
    # A second connection, not the corpse of the first.
    assert len(providers["asr"].configs) == 2


async def _degrade_on_loop(engine, why: str) -> None:
    """Degrade from inside the engine loop, as a real provider failure does."""
    engine._degrade("degrade", why)
    await asyncio.sleep(0.05)  # let the retirement task run


# ----------------------------------------------------------------------
# the "no endpoint matching 'CABLE Input'" dead end
# ----------------------------------------------------------------------


class FakeManager:
    """Just enough DeviceManager to exercise the resolution failure paths."""

    def __init__(self, playback: list[str]) -> None:
        self._playback = playback

    def wasapi_devices(self) -> list[DeviceRef]:
        return [dev(n, i) for i, n in enumerate(self._playback)]

    def find(self, needle: str, *, capture=None, loopback=None) -> DeviceRef:
        from app.audio.devices import AudioUnavailable

        hits = [d for d in self.wasapi_devices() if needle.lower() in d.name.lower()]
        if not hits:
            raise AudioUnavailable(f"no WASAPI endpoint matching {needle!r}")
        return hits[0]

    def default_output(self) -> DeviceRef:
        return dev(self._playback[0], 0)


def _problems_for(source: str, playback: list[str]) -> str:
    cfg = Config()
    cfg.listen = ListenConfig(source=source)
    ports, problems, _notes = L.build_listen_ports(cfg, FakeManager(playback))
    assert ports is None
    return "\n".join(problems)


def test_a_missing_cable_says_the_driver_is_not_installed():
    text = _problems_for("CABLE Input", ["Speaker (Realtek(R) Audio)", "Headphones"])
    assert "vb-audio.com/Cable" in text
    assert "reboot" in text
    # And it lists what does exist, so the name can be checked by eye.
    assert "Speaker (Realtek(R) Audio)" in text


def test_a_present_cable_under_another_name_says_use_that_name():
    text = _problems_for(
        "CABLE Inpt",  # a typo, not a missing driver
        ["Speaker (Realtek(R) Audio)", "CABLE Input (VB-Audio Virtual Cable)"],
    )
    assert "does contain a virtual cable" in text
    assert "CABLE Input (VB-Audio Virtual Cable)" in text
    assert "vb-audio.com" not in text  # do not send them to reinstall it


def test_the_full_vb_cable_name_resolves_from_a_fragment():
    mgr = FakeManager(["CABLE Input (VB-Audio Virtual Cable)"])
    assert mgr.find("cable input").name == "CABLE Input (VB-Audio Virtual Cable)"


def test_cable_scan_reports_a_missing_driver():
    text = L.cable_scan([dev("Speaker (Realtek(R) Audio)"), dev("Headphones", 2)])
    assert "not installed" in text
    assert "elevation" in text  # the second most common cause


def test_cable_scan_finds_a_cable_windows_is_not_exposing_over_wasapi():
    devices = [
        dev("Speaker (Realtek(R) Audio)"),
        DeviceRef(name="CABLE Input (VB-Audio Virtual Cable)", index=7,
                  max_input_channels=0, max_output_channels=2,
                  default_rate=48_000, host_api="MME"),
    ]
    text = L.cable_scan(devices)
    assert "IS present" in text
    assert "MME" in text
    assert "mmsys.cpl" in text
    assert "Show Disabled Devices" in text


def test_cable_scan_recognises_voicemeeter_too():
    devices = [DeviceRef(name="VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
                         index=3, max_input_channels=0, max_output_channels=2,
                         default_rate=48_000, host_api="Windows WASAPI")]
    assert "IS present" in L.cable_scan(devices)


# ----------------------------------------------------------------------
# fluency: the cushion, the pipelining, and the drift signal
# ----------------------------------------------------------------------


def _ms(pcm: np.ndarray, rate: int = 48_000, channels: int = 2) -> float:
    return pcm.size / (rate * channels / 1000.0)


def test_playback_waits_for_a_cushion_before_the_first_sound(engine_fixture):
    """The first write must carry the preroll, not the first 100 ms chunk.

    The render callback drains the ring every period whether or not anything
    is filling it, and serves an empty ring as silence. Handing over the first
    chunk the moment it arrives therefore means any late chunk is a hole in
    the middle of a word - heard as a stutter, or as a robotic clip.
    """
    engine, ports, providers, *_ = engine_fixture(
        ["hello"], tts_chunks=8, playback_preroll_ms=400.0
    )
    # Each chunk is 100 ms, so nothing may be written until the fifth is asked
    # for - by which point four have been produced and the cushion is full.
    seen: list[tuple[int, int]] = []
    providers["tts"].on_chunk = lambda i: seen.append((i, len(ports.output.written)))

    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))

    assert wait_for(lambda: ports.output.played_samples > 0)
    early = [(i, w) for i, w in seen if i <= 3]
    assert early and all(w == 0 for _, w in early), (
        f"audio was written before the cushion filled: {early}"
    )
    # And it did not hold the whole utterance back either.
    assert sum(1 for _, w in seen if w > 0) > 0


def test_an_utterance_shorter_than_the_cushion_is_still_spoken(engine_fixture):
    # "kya?" is a whole turn. A cushion that swallows it is a bug, not a buffer.
    engine, ports, *_ = engine_fixture(
        ["yes"], tts_chunks=1, playback_preroll_ms=400.0
    )
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))

    assert wait_for(lambda: ports.output.played_samples > 0), (
        "a 100 ms utterance never reached the sink"
    )
    assert _ms(ports.output.written[0]) > 0


def test_speaking_does_not_block_on_the_tail_of_the_utterance(engine_fixture):
    """Two units must not be serialised behind each other's playback.

    If synthesis waits for the previous utterance to finish, then recognising,
    translating and synthesising the next one - one to two seconds of it -
    happens in the silence after it. That pause between every phrase is what
    "not fluent" sounds like.
    """
    engine, ports, providers, *_ = engine_fixture(
        ["First sentence.", "Second sentence."], tts_chunks=8
    )
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))

    # Both units reach synthesis well inside the time it would take to play
    # the first one out (800 ms) twice over.
    assert wait_for(lambda: len(providers["tts"].spoken) == 2, timeout=2.0), (
        f"only {providers['tts'].spoken} was synthesised; the second unit was "
        "waiting for the first to finish playing"
    )


def test_a_full_ring_is_not_mistaken_for_falling_behind():
    """The bug this encodes: a constant metallic edge on every utterance.

    The sink ring is a fixed half-second buffer the player deliberately keeps
    full. Reading it as "queue depth" reports a permanent 500 ms of lag, and a
    drift controller fed that number time-stretches every utterance forever -
    audible as a metallic edge long before it is audible as speed.
    """
    from app.controllers.drift import DriftController

    for name, values in L.PRESETS.items():
        drift = DriftController(
            comfortable_ms=values["drift_comfortable_ms"],
            stretch_ms=values["drift_stretch_ms"],
            aggressive_ms=values["drift_aggressive_ms"],
            max_rate=values["drift_max_rate"],
        )
        assert drift.decide(500.0).rate == 1.0, (
            f"preset {name!r} time-stretches a merely-full ring"
        )


def test_the_fluent_preset_leaves_the_voice_alone_for_seconds(engine_fixture):
    cfg = Config()
    note = L.apply_preset(cfg, "fluent")
    assert "video" in note
    assert cfg.pipeline.drift_comfortable_ms >= 4000.0
    assert cfg.pipeline.max_words >= 24     # whole thoughts, not phrases
    assert cfg.pipeline.drift_max_rate <= 1.10  # no audible stretch


def test_the_fast_preset_really_is_tighter_than_the_fluent_one():
    fluent, fast = Config(), Config()
    L.apply_preset(fluent, "fluent")
    L.apply_preset(fast, "fast")
    assert fast.pipeline.stability_ms < fluent.pipeline.stability_ms
    assert fast.pipeline.max_words < fluent.pipeline.max_words
    assert fast.pipeline.drift_comfortable_ms < fluent.pipeline.drift_comfortable_ms


def test_an_unknown_preset_is_refused_by_name():
    with pytest.raises(KeyError):
        L.apply_preset(Config(), "smooth")


# ----------------------------------------------------------------------
# two regressions that produced "no audio at all"
# ----------------------------------------------------------------------


def test_synthesis_never_starves_the_recogniser(engine_fixture):
    """The Deepgram 1011 bug, encoded.

    Translating and synthesising a unit takes seconds. Both callers that emit
    units sit on the recognition path, and one of them is the task that feeds
    audio to the recogniser. Awaiting synthesis inside them stops that feed,
    and a streaming recogniser does not wait politely: Deepgram closes the
    socket with 1011 after roughly ten seconds without audio, which the user
    sees as translation dying part-way through a video.
    """
    engine, ports, providers, *_ = engine_fixture(["hello"], tts_delay=1.0)
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(300))

    # Wait until synthesis is under way and deliberately slow.
    assert wait_for(lambda: providers["tts"].spoken, timeout=3.0)
    pushed = providers["asr"].stream.pushed_samples

    # More speech arrives while the synthesiser is still busy.
    ports.source.ring.write(loud(300))
    assert wait_for(
        lambda: providers["asr"].stream.pushed_samples > pushed, timeout=2.0
    ), "the recogniser was starved of audio while synthesis was running"


def test_a_cascade_at_its_normal_lag_is_not_thrown_away():
    """The "no audio at all" bug, encoded.

    Queue depth used to be read from the sink ring - a fixed half-second
    buffer - so it could never report more than 500 ms no matter how far
    behind the pipeline really was, and the 3 s drop threshold was simply
    unreachable. Once the measurement became the real lag, those same numbers
    dropped every single unit and played nothing.

    Recognition plus translation plus synthesis genuinely runs three to six
    seconds behind. That is the normal state of this pipeline, not a fault,
    and it must not be treated as one.
    """
    from app.controllers.drift import DriftController

    p = Config().pipeline
    drift = DriftController(
        comfortable_ms=p.drift_comfortable_ms,
        stretch_ms=p.drift_stretch_ms,
        aggressive_ms=p.drift_aggressive_ms,
        max_rate=p.drift_max_rate,
    )
    for lag_ms in (3000.0, 4500.0, 6000.0):
        assert not drift.decide(lag_ms).drop, (
            f"{lag_ms / 1000:.1f} s of lag - normal for a cascade - is dropped "
            "by the default settings, so nothing would ever be spoken"
        )
    # It must still give up eventually rather than grow without limit.
    assert drift.decide(30_000.0).drop


def test_no_preset_ever_drops_a_normal_cascade_lag():
    from app.controllers.drift import DriftController

    for name, values in L.PRESETS.items():
        drift = DriftController(
            comfortable_ms=values["drift_comfortable_ms"],
            stretch_ms=values["drift_stretch_ms"],
            aggressive_ms=values["drift_aggressive_ms"],
            max_rate=values["drift_max_rate"],
        )
        assert not drift.decide(4000.0).drop, f"preset {name!r} drops at 4 s"


def test_units_keep_being_spoken_over_a_run(engine_fixture):
    """Several units in a row must all reach synthesis, not just the first."""
    engine, ports, providers, *_ = engine_fixture(
        ["One.", "Two.", "Three.", "Four."]
    )
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(400))

    assert wait_for(lambda: len(providers["tts"].spoken) == 4, timeout=5.0), (
        f"only {len(providers['tts'].spoken)} of 4 units were spoken"
    )
    assert ports.output.played_samples > 0


def test_every_utterance_of_a_session_is_spoken_not_just_the_first():
    """The bug that made it fall silent after one sentence.

    `_emitted` stops the SAME utterance being spoken twice while the
    recogniser is still revising it. Carrying it across utterances is a
    different rule and a fatal one: `_remainder` compares brand-new speech
    against a sentence with nothing in common, finds no shared prefix, and
    returns nothing. One sentence is translated and the rest of the session
    is silence.
    """
    from app.controllers.segmenter import Hypothesis, Segmenter

    seg = Segmenter(min_hypotheses=1, stable_ms=0.0)
    spoken = []
    for text in ("Example your grammar.", "Now we begin.", "The third one."):
        spoken += [u.text for u in seg.push(Hypothesis(text, True, 0.0))]
    assert spoken == ["Example your grammar.", "Now we begin.", "The third one."]


def test_the_same_thing_via_the_vad_flush_path():
    from app.controllers.segmenter import Hypothesis, Segmenter

    seg = Segmenter(min_hypotheses=1, stable_ms=0.0)
    spoken = []
    for utterance in (["Example your", "Example your grammar"],
                      ["Now we will", "Now we will begin"]):
        for text in utterance:
            seg.push(Hypothesis(text, False, 0.0))
        spoken += [u.text for u in seg.flush()]
    assert spoken == ["Example your grammar", "Now we will begin"]


def test_a_repeated_final_is_still_not_spoken_twice():
    # What the accumulating `_emitted` was there to prevent, kept.
    from app.controllers.segmenter import Hypothesis, Segmenter

    seg = Segmenter(min_hypotheses=1, stable_ms=0.0)
    assert [u.text for u in seg.push(Hypothesis("Hello there.", True, 0.0))]
    assert seg.push(Hypothesis("Hello there.", True, 0.1)) == []
    assert seg.push(Hypothesis("Hello there", True, 0.2)) == []  # punctuation


# ----------------------------------------------------------------------
# half-repeated sentences: replayed from a real video
# ----------------------------------------------------------------------


def _replay(script):
    """Run a recogniser script through a segmenter; return what gets spoken."""
    from app.controllers.segmenter import Hypothesis, Segmenter

    seg = Segmenter(min_hypotheses=1, stable_ms=0.0)
    spoken = []
    for text, final in script:
        units = seg.flush() if text is None else seg.push(
            Hypothesis(text, final, 0.0)
        )
        spoken += [u.text for u in units]
    return spoken


def test_a_breath_does_not_make_the_sentence_start_again():
    """Taken verbatim from a session on a real YouTube video.

    The gate reports end-of-speech at every 400 ms breath. The recogniser
    does not: it keeps extending one hypothesis straight across it. If the
    segmenter forgets what it has already spoken at each of those breaths,
    the next hypothesis - which still contains all of it - is spoken again
    from the top with a few new words on the end. Heard as the same half
    sentence over and over.
    """
    spoken = _replay([
        ("or an advanced student?", False),
        (None, None),                       # VAD says the speaker paused
        ("or an advanced student? Here are the rules.", False),
        (None, None),
        ("No pausing the", False),
        (None, None),
        ("No pausing the video. No translating.", False),
    ])
    assert spoken == [
        "or an advanced student?",
        "Here are the rules.",
        "No pausing the",
        "video. No translating.",
    ], f"a phrase was spoken twice: {spoken}"


def test_punctuation_the_recogniser_moves_around_is_not_new_speech():
    # "student" -> "student," -> "student?" is the same word three times.
    assert _replay([
        ("are you an intermediate", False),
        (None, None),
        ("Are you an intermediate,", False),
        (None, None),
        ("are you an intermediate?", False),
    ]) == ["are you an intermediate"]


def test_genuinely_new_speech_still_gets_through():
    # The failure the repeat-guard must not cause: swallowing everything.
    assert _replay([
        ("What is your level of English?", False),
        (None, None),
        ("Completely different sentence now.", False),
    ]) == ["What is your level of English?", "Completely different sentence now."]


def test_a_retraction_is_not_spoken_backwards():
    # The recogniser shortening its guess cannot unsay what we already said.
    assert _replay([
        ("I think that is right", False),
        (None, None),
        ("I think that is", False),
    ]) == ["I think that is right"]


# ----------------------------------------------------------------------
# time to the first word after Start
# ----------------------------------------------------------------------


def test_the_first_unit_is_cut_short_then_normal_length_resumes():
    """Pressing Start mid-sentence must not feel broken.

    Waiting for a whole clause to be confirmed is correct and feels like
    nothing is happening - the listener cannot tell "working, nothing to say
    yet" from "not working". The first unit is cut short deliberately;
    everything after it is worth the wait it costs.
    """
    from app.controllers.segmenter import Hypothesis, Segmenter

    line = "so today we are going to"  # six words, no clause boundary

    patient = Segmenter(min_hypotheses=1, stable_ms=0.0, max_words=18)
    assert patient.push(Hypothesis(line, False, 0.0)) == [], (
        "normally this waits for a clause - that is the point of the setting"
    )

    hurried = Segmenter(min_hypotheses=1, stable_ms=0.0, max_words=18)
    hurried.hurry(stable_ms=0.0, max_words=5)
    assert hurried.push(Hypothesis(line, False, 0.0)), (
        "the first unit after Start should not wait for a whole clause"
    )
    # And the hurry is over: the configured length applies again.
    assert hurried.max_words == 18


def test_hurrying_twice_does_not_lose_the_real_settings():
    from app.controllers.segmenter import Segmenter

    seg = Segmenter(min_hypotheses=1, stable_ms=350.0, max_words=18)
    seg.hurry()
    seg.hurry()          # a second Start before anything was spoken
    seg._settle()
    assert (seg.stability.stable_ms, seg.max_words) == (350.0, 18)


def test_synthesis_is_audible_before_the_sentence_is_finished():
    """Edge TTS streams MP3 while it is still generating.

    Collecting every chunk and decoding once at the end throws that away: a
    five-second sentence becomes five seconds of silence followed by five
    seconds of speech, and time-to-first-sound becomes the length of the
    sentence instead of the length of a network round trip.
    """
    import asyncio as aio

    from app.providers.base import VoiceId
    from app.providers.tts import EdgeTextToSpeech

    class FakeEdge(EdgeTextToSpeech):
        """Real streaming logic, fake network and fake decoder."""

        def __init__(self):
            super().__init__(rate=24_000)
            self.parts = [b"x" * 3000] * 6   # six bursts of "mp3"

        async def _stream_mp3(self, text, voice_name):
            for part in self.parts:
                yield part

        def _decode(self, mp3: bytes):
            # One byte of "mp3" stands for one sample.
            return np.zeros(len(mp3), dtype=np.int16)

    async def run():
        got = []
        async for chunk in FakeEdge().synthesize("a whole sentence", VoiceId("edge", "ur")):
            got.append(chunk.pcm.size)
        return got

    sizes = aio.run(run())
    audible = [n for n in sizes if n > 0]
    assert len(audible) > 1, (
        f"the whole utterance arrived in one block ({sizes}); nothing would be "
        "heard until synthesis finished"
    )
    # Nothing is emitted twice: the total is the full stream, once.
    assert sum(audible) == 18_000


# ----------------------------------------------------------------------
# first run should reach the bubble, not a form
# ----------------------------------------------------------------------


class FakeManagerWithDefaults(FakeManager):
    def __init__(self, playback, *, default_in="Microphone Array",
                 default_out=None):
        super().__init__(playback)
        self._default_in = default_in
        self._default_out = default_out or playback[0]

    def default_input(self) -> DeviceRef:
        return dev(self._default_in, 99, **CAPTURE)

    def default_output(self) -> DeviceRef:
        return dev(self._default_out, 0)


def test_two_cables_are_chosen_and_saved_without_asking(tmp_path):
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE-B Input (VB-Audio Cable B)",
    ])
    notes, problems = autoconfigure(cfg, mgr, save=False)

    assert problems == []
    assert cfg.devices.microphone == "Microphone Array"
    assert cfg.devices.headphones == "Speaker (Realtek(R) Audio)"
    assert cfg.devices.virtual_mic != cfg.devices.teams_speaker
    assert "CABLE" in cfg.devices.virtual_mic
    assert "CABLE" in cfg.devices.teams_speaker
    assert len(notes) == 4


def test_one_cable_says_the_one_thing_that_is_missing():
    """Not three feedback-loop complaints, which are consequences of it."""
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)

    assert len(problems) == 1
    assert "TWO virtual cables" in problems[0]
    assert "listen.py" in problems[0], "the one-way way out must be offered"


def test_a_cable_you_can_hear_is_never_picked():
    # Feedback loop §5.6: the endpoint we render into must not be audible.
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.devices.headphones = "CABLE Input (VB-Audio Virtual Cable)"
    mgr = FakeManagerWithDefaults(["CABLE Input (VB-Audio Virtual Cable)"])
    autoconfigure(cfg, mgr, save=False)
    assert cfg.devices.virtual_mic == ""


def test_choices_already_made_are_never_overwritten():
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.devices.microphone = "My Good Headset"
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE-B Input (VB-Audio Cable B)",
    ])
    autoconfigure(cfg, mgr, save=False)
    assert cfg.devices.microphone == "My Good Headset"


def test_a_complete_autoconfigure_marks_setup_done(tmp_path):
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.save(tmp_path / "settings.json")  # so save() has somewhere to go
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE-B Input (VB-Audio Cable B)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)
    assert problems == []
    # save=False here so the test never writes to the real profile; the flag
    # is what the application acts on.
    assert cfg.devices.virtual_mic and cfg.devices.teams_speaker


# ----------------------------------------------------------------------
# the bubble, and being able to photograph it
# ----------------------------------------------------------------------


def test_startup_never_puts_a_form_in_front_of_the_bubble():
    """A modal setup window is a door where a sign would do.

    Someone who has already configured this - or who only wants to look at it
    - should get the interface. When audio cannot start the bubble goes red
    and the panel says why, with Settings one click away inside it.
    """
    import ast
    from pathlib import Path as _Path

    # Parsed from source, not imported: this must hold on a machine without
    # PySide6, which is where the tests run.
    source = _Path("app/controllers/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "Application"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    called = {
        node.func.attr
        for node in ast.walk(run)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_show_ui" in called
    assert "_run_setup" not in called, (
        "startup opens the setup dialog again; the bubble should come up "
        "either way"
    )


def test_capture_exclusion_is_one_switch_for_every_window(monkeypatch):
    from app.views import capture_guard

    applied = []
    monkeypatch.setattr(
        capture_guard, "exclude_from_capture", lambda w: applied.append(w) or True
    )
    monkeypatch.setattr(capture_guard, "no_activate", lambda w: False)
    monkeypatch.setattr(capture_guard, "supported", lambda: True)

    try:
        capture_guard.set_enabled(False)
        capture_guard.harden(object(), unfocusable=False)
        assert applied == [], "a window hid itself while the switch was off"

        capture_guard.set_enabled(True)
        capture_guard.harden(object(), unfocusable=False)
        assert len(applied) == 1, "the switch did not turn back on"
    finally:
        capture_guard.set_enabled(True)


def test_the_waiver_is_never_written_to_the_profile():
    """§2.4 is not a preference. A saved "off" waits for the wrong moment."""
    from app.models.config import Config

    assert Config().ui.exclude_from_capture is True


def test_a_speaker_saved_as_a_cable_is_repaired_not_respected():
    """The state a first run leaves behind when every dropdown defaults alike.

    "Never overwrite a choice already made" is right for a microphone. For
    the two cable roles it is wrong: they MUST be virtual cables, so a saved
    speaker there is not a preference anybody expressed - it is the leftover
    of a setup window where every dropdown defaulted to the same endpoint.
    Respecting it means explaining the same feedback-loop refusal for ever
    while a working cable sits unused in the list.
    """
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.devices.headphones = "Speaker (Realtek(R) Audio)"
    cfg.devices.virtual_mic = "Speaker (Realtek(R) Audio)"     # the leftover
    cfg.devices.teams_speaker = "Speaker (Realtek(R) Audio)"

    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE-B Input (VB-Audio Cable B)",
    ])
    notes, problems = autoconfigure(cfg, mgr, save=False)

    assert problems == []
    assert cfg.devices.virtual_mic != cfg.devices.headphones
    assert "CABLE" in cfg.devices.virtual_mic
    assert "CABLE" in cfg.devices.teams_speaker
    assert any("cleared" in n for n in notes), "the repair was silent"


def test_a_real_cable_choice_is_still_left_alone():
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.devices.virtual_mic = "CABLE-B Input (VB-Audio Cable B)"
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE-B Input (VB-Audio Cable B)",
    ])
    autoconfigure(cfg, mgr, save=False)
    assert cfg.devices.virtual_mic == "CABLE-B Input (VB-Audio Cable B)"
    assert cfg.devices.teams_speaker == "CABLE Input (VB-Audio Virtual Cable)"


def test_one_cable_is_never_handed_out_for_both_roles():
    """A collision is the same failure as a speaker: our English feeds back in.

    Repairing a bad `virtual_mic` on a machine with a single cable must not
    quietly assign that cable to both roles and report success.
    """
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.devices.headphones = "Speaker (Realtek(R) Audio)"
    cfg.devices.virtual_mic = "Speaker (Realtek(R) Audio)"        # leftover
    cfg.devices.teams_speaker = "CABLE Input (VB-Audio Virtual Cable)"

    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)

    assert cfg.devices.virtual_mic != cfg.devices.teams_speaker or not problems
    assert problems, "a one-cable machine was told two-way mode was ready"
    assert "TWO virtual cables" in problems[0]


# ----------------------------------------------------------------------
# the OpenAI realtime wire shape
# ----------------------------------------------------------------------


def test_the_retired_beta_header_is_never_sent():
    """`beta_api_shape_disabled`, encoded.

    `OpenAI-Beta: realtime=v1` selected the beta wire shape. That shape is
    retired: a session opened with the header is closed immediately with
    4000 / invalid_request_error.beta_api_shape_disabled, which reaches the
    user as "outgoing translation unavailable" with nothing pointing at a
    header as the cause.
    """
    from app.providers.openai_s2st import OpenAiSpeechTranslator

    headers = OpenAiSpeechTranslator("sk-test").headers()
    assert "OpenAI-Beta" not in headers
    assert headers["Authorization"] == "Bearer sk-test"


def test_the_session_declares_its_type():
    from app.providers.base import S2stConfig
    from app.providers.openai_s2st import OpenAiSpeechTranslator

    cfg = S2stConfig(source_language="ur", target_language="en", rate=24_000)
    payload = OpenAiSpeechTranslator("sk-test").session_update(cfg)

    assert payload["type"] == "session.update"
    # The GA discriminator. The beta shape had no such field and inferred it
    # from the header we no longer send.
    assert payload["session"]["type"] == "realtime"
    assert payload["session"]["audio"]["output"]["language"] == "en"
    assert payload["session"]["audio"]["input"]["format"]["rate"] == 24_000


def test_the_panel_refits_whenever_its_content_changes():
    """Two rows of buttons, one a ghost of the previous layout.

    The panel's reason text is word-wrapped and can go from nothing to four
    lines. `adjustSize` used to run only when the panel opened, so a panel
    that was ALREADY open kept its old geometry, the card was drawn outside
    it, and on a frameless translucent window nothing repainted the
    abandoned pixels.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path("app/views/overlay.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    panel = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Panel"
    )
    methods = {
        node.name: node for node in panel.body if isinstance(node, ast.FunctionDef)
    }
    assert "_refit" in methods, "the panel never resizes to its content"

    for name in ("apply_state", "apply_metrics", "set_quote"):
        called = {
            node.func.attr
            for node in ast.walk(methods[name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "_refit" in called, f"Panel.{name} changes content without refitting"


# ----------------------------------------------------------------------
# staying alive: an idle source, and a dropped socket
# ----------------------------------------------------------------------


def test_an_idle_source_does_not_look_like_a_dead_client(engine_fixture):
    """The Deepgram 1011 that ends a session when a video goes quiet.

    An idle capture endpoint delivers no callbacks at all - WASAPI loopback
    on a device nothing is playing into is not a stream of silence, it is no
    stream. So nothing reaches the recogniser, and it closes the socket after
    about ten seconds: exactly what happens when a video is paused, buffers,
    or its audio device fails.
    """
    engine, ports, providers, *_ = engine_fixture(["hello"])
    engine.start()
    engine.start_listening().result(timeout=5)
    engine._pipe.KEEPALIVE_S = 0.2  # the real one is 4 s

    # Nothing at all is written to the capture ring: the source is idle.
    assert wait_for(
        lambda: providers["asr"].stream.pushed_samples > 0, timeout=3.0
    ), "no keepalive audio was sent; the recogniser would close the session"


def test_a_dropped_socket_reconnects_instead_of_ending_the_session(engine_fixture):
    """A dropped websocket is ordinary - a sleeping laptop, a wifi handover.

    Ending the session for one, and falling back to the original audio, is a
    worse answer than waiting a moment and dialling again.
    """
    engine, ports, providers, *_ = engine_fixture(
        ["one", "two", "three"], asr_drops_after=1
    )
    engine.start()
    engine.start_listening().result(timeout=5)
    ports.source.ring.write(loud(300))

    assert wait_for(lambda: len(providers["asr"].streams) >= 2, timeout=5.0), (
        "the stream died and was never reopened"
    )
    assert engine.status == L.LISTENING, (
        f"status fell to {engine.status!r} instead of recovering"
    )
    assert ports.wire.muted is True, "the wire unmuted over a recoverable drop"


def test_it_still_gives_up_when_the_recogniser_will_not_answer(engine_fixture):
    engine, ports, providers, *_ = engine_fixture(["one"], asr_drops_after=0)
    engine.start()
    engine.start_listening().result(timeout=5)
    # Every reopen fails from here on.
    providers["asr"].fail = True

    assert wait_for(lambda: engine.status == L.DEGRADED, timeout=8.0), (
        "a recogniser that never answers should end in a fallback, not a "
        "reconnect loop"
    )
    assert ports.wire.muted is False, "degrading must return the original audio"


# ----------------------------------------------------------------------
# video mode: one cable, no microphone
# ----------------------------------------------------------------------


def test_video_mode_needs_one_cable_and_no_microphone():
    """Nothing is being sent to anybody, so no microphone is opened.

    Not merely unused: opening it would put a live capture of the room into
    a session that has no direction to send it in.
    """
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.mode = "video"
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)

    assert problems == [], problems
    assert cfg.devices.microphone == "", "video mode opened a microphone"
    assert cfg.devices.teams_speaker == "CABLE Input (VB-Audio Virtual Cable)"
    assert cfg.devices.headphones == "Speaker (Realtek(R) Audio)"


def test_video_mode_still_needs_something_to_listen_to():
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.mode = "video"
    mgr = FakeManagerWithDefaults(["Speaker (Realtek(R) Audio)"])
    _notes, problems = autoconfigure(cfg, mgr, save=False)
    assert len(problems) == 1
    assert "one virtual cable" in problems[0]


def test_a_machine_set_up_for_meetings_keeps_working_in_video_mode():
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.mode = "video"
    cfg.devices.headphones = "Speaker (Realtek(R) Audio)"
    cfg.devices.virtual_mic = "CABLE In 16ch (VB-Audio Virtual Cable)"
    cfg.devices.teams_speaker = "CABLE Input (VB-Audio Virtual Cable)"
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE In 16ch (VB-Audio Virtual Cable)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)
    assert problems == []
    assert cfg.devices.teams_speaker == "CABLE Input (VB-Audio Virtual Cable)"


def test_meeting_mode_still_wants_both_cables_and_a_microphone():
    from app.controllers.autoconfig import autoconfigure

    cfg = Config()
    cfg.mode = "meeting"
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)
    assert cfg.devices.microphone == "Microphone Array"
    assert problems and "TWO virtual cables" in problems[0]


def test_the_mode_survives_a_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    cfg = Config()
    cfg.mode = "video"
    cfg.save(path)
    assert Config.load(path).mode == "video"
    # And a file written before modes existed still loads.
    path.write_text('{"devices": {}}', encoding="utf-8")
    assert Config.load(path).mode == ""


def test_a_pair_with_one_wire_behaves():
    """Video mode has no outgoing wire; the pair must not assume two."""
    from app.audio.passthrough import PassthroughPair

    class W:
        def __init__(self):
            self.muted = False
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def mute(self):
            self.muted = True

        def unmute(self):
            self.muted = False

        def health(self):
            return {"muted": self.muted}

    incoming = W()
    pair = PassthroughPair(outgoing=None, incoming=incoming)
    pair.start()
    assert incoming.started
    pair.mute_all()
    assert incoming.muted
    pair.unmute_all()
    assert not incoming.muted
    assert pair.health()["outgoing"] is None
    pair.stop()
    assert not incoming.started


def test_settings_offers_the_source_picker_and_saves_it():
    """The mode has to be changeable without the command line.

    It decides whether a microphone is opened, so a dialog that cannot reach
    it leaves the CLI as the only way to correct the most consequential
    choice in the application.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path("app/views/settings.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    dialog = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SettingsDialog"
    )
    methods = {
        node.name: node for node in dialog.body if isinstance(node, ast.FunctionDef)
    }
    assert "_apply_mode" in methods, "the dialog cannot show mode-specific fields"

    devices_tab = ast.dump(methods["_devices_tab"])
    assert "cb_mode" in devices_tab, "no source picker on the Devices tab"
    for value in ("video", "meeting"):
        assert f"'{value}'" in devices_tab, f"the {value!r} option is missing"

    saved = ast.dump(methods["_save"])
    assert "cb_mode" in saved, "the dialog never writes the mode back"
    assert "mode" in saved


def test_video_mode_is_configured_with_one_cable_by_the_dialogs_rule():
    """`setup_complete` must not demand two cables in a one-cable topology."""
    import ast
    from pathlib import Path as _Path

    source = _Path("app/views/settings.py").read_text(encoding="utf-8")
    save = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_save"
    )
    body = ast.dump(save)
    # The rule is branched on the mode rather than always requiring both.
    assert "setup_complete" in body
    assert body.count("virtual_mic") >= 1 and "'video'" in body


def test_the_settings_dialog_cannot_grow_past_the_screen():
    """Save must stay clickable on a short display.

    The Devices tab is the tallest and grew again with the source picker. A
    tab as tall as its contents makes the dialog as tall as its contents,
    which on a 768-pixel-high laptop screen pushes the Save button off the
    bottom of the display - the one control the window exists for.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path("app/views/settings.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    dialog = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SettingsDialog"
    )
    methods = {
        node.name: node for node in dialog.body if isinstance(node, ast.FunctionDef)
    }

    assert "_scrollable" in methods, "the tabs cannot scroll"
    assert "_fit_to_screen" in methods, "the dialog never measures the screen"

    init = ast.dump(methods["__init__"])
    assert init.count("_scrollable") == 5, "not every tab was made scrollable"
    assert "setMinimumHeight" not in init, (
        "a minimum height defeats fitting to the screen"
    )

    shown = ast.dump(methods["showEvent"])
    assert "_fit_to_screen" in shown

    # The buttons must be added to the dialog, not inside a scroll area.
    fit = ast.dump(methods["_fit_to_screen"])
    assert "availableGeometry" in fit
