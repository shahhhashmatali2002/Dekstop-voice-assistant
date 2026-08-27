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
    """Stands in for a CaptureSource or a RenderSink: a real ring, no device."""

    def __init__(self, rate: int = 48_000, channels: int = 2) -> None:
        self.rate = rate
        self.channels = channels
        self.ring = RingBuffer(rate * channels)  # one second
        self.written: list[np.ndarray] = []
        self.stopped = False

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
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.pushed: list[np.ndarray] = []
        self.closed = False

    async def push_audio(self, pcm: np.ndarray) -> None:
        self.pushed.append(pcm)

    async def hypotheses(self):
        for text in self.texts:
            await asyncio.sleep(0.01)
            yield TranscriptDelta(text, is_final=True)
        while True:  # a real stream stays open after its last result
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        self.closed = True

    @property
    def pushed_samples(self) -> int:
        return sum(a.size for a in self.pushed)


class FakeAsr(SpeechToText):
    name = "fake-asr"

    def __init__(self, texts: list[str] | None = None, fail: bool = False) -> None:
        self.texts = texts or []
        self.fail = fail
        self.stream: FakeAsrStream | None = None
        self.configs: list = []

    async def open(self, cfg):
        self.configs.append(cfg)
        if self.fail:
            raise ProviderUnavailable("recognition is down")
        self.stream = FakeAsrStream(self.texts)
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

    def __init__(self, rate: int = 24_000) -> None:
        self.rate = rate
        self.spoken: list[str] = []

    async def synthesize(self, text: str, voice):
        self.spoken.append(text)
        for _ in range(3):
            yield PcmChunk(np.zeros(self.rate // 10, dtype=np.int16), self.rate)
        yield PcmChunk(np.empty(0, dtype=np.int16), self.rate, final=True)


def build(texts=None, *, asr_fails=False):
    """An engine wired to fakes, plus the parts a test needs to inspect."""
    cfg = Config()
    cfg.listen = ListenConfig(source="CABLE Input", output="Headphones")
    # Emit on the first hypothesis: these tests are about wiring, and the
    # stability window is tested exhaustively in test_pipeline.py.
    cfg.pipeline.stability_hypotheses = 1
    cfg.pipeline.stability_ms = 0.0

    source = FakeEndpoint()
    output = FakeEndpoint()
    wire = FakeWire()
    ports = L.ListenPorts(
        source=source,
        output=output,
        wire=wire,
        source_device=dev("CABLE Input"),
        output_device=dev("Headphones", 2),
    )
    providers = {
        "asr": FakeAsr(texts, fail=asr_fails),
        "translator": FakeTranslator(),
        "tts": FakeTts(),
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
