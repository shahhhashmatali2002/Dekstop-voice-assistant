"""Wiring checks: does every method a class calls on itself actually exist?

WHY THIS FILE EXISTS
--------------------
An edit once removed two methods from the Application class while leaving the
call sites in place. Nothing caught it: `py_compile` is happy with a missing
method, and the unit tests never went down that path, so the first sign of
trouble was an AttributeError in the user's terminal on startup.

This is a static check - it parses the source and compares what each class
calls on `self` against what it defines. It needs no Qt, no audio hardware and
no network, so it runs everywhere and it is fast.

Classes that inherit from something outside this project (every Qt widget) are
skipped, because their real surface includes everything the base class brings
and we cannot see that statically.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parent.parent / "app"
SOURCES = sorted(APP.rglob("*.py"))

#: Set on instances from outside the class body, legitimately.
EXTERNALLY_ASSIGNED = {
    "on_transcript",
    "on_notice",
    "on_degrade",
    "on_speech",
    "draggable",
    "frameless",
    "unfocusable",
}


def classes_with_local_bases(tree: ast.Module):
    """Only classes whose entire base set is inside this project (or none).

    A Qt subclass calls hundreds of inherited methods on self; checking those
    statically would be noise.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if all(
                isinstance(b, ast.Name) and b.id in ("object", "ABC", "Enum", "str")
                for b in node.bases
            ) or not node.bases:
                yield node


def defined_on(cls: ast.ClassDef) -> set[str]:
    names: set[str] = set(EXTERNALLY_ASSIGNED)
    for node in ast.walk(cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            if isinstance(node.value, ast.Name) and node.value.id == "self":
                names.add(node.attr)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Attribute):
            t = node.target
            if isinstance(t.value, ast.Name) and t.value.id == "self":
                names.add(t.attr)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):  # class-level attribute
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def used_on(cls: ast.ClassDef) -> dict[str, int]:
    used: dict[str, int] = {}
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            used.setdefault(node.attr, node.lineno)
    return used


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(APP)))
def test_no_class_calls_a_method_it_does_not_have(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems = []
    for cls in classes_with_local_bases(tree):
        defined = defined_on(cls)
        for attr, line in used_on(cls).items():
            if attr.startswith("__"):
                continue
            if attr not in defined:
                problems.append(f"{path.name}:{line}  {cls.name}.{attr}")
    assert not problems, "self.<attr> with nothing defining it:\n  " + "\n  ".join(
        problems
    )


def test_application_defines_everything_its_startup_path_calls():
    """The specific regression: run() calling a method an edit had removed."""
    main = APP / "controllers" / "main.py"
    tree = ast.parse(main.read_text(encoding="utf-8"))
    app_cls = next(
        c for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "Application"
    )
    defined = defined_on(app_cls)
    for required in (
        "run",
        "_install_signal_handlers",
        "_show_ui",
        "_run_setup",
        "_setup_needed",
        "_start_engine",
        "build_audio",
        "build_providers",
        "quit",
        "_cleanup",
    ):
        assert required in defined, f"Application.{required} is missing"


def test_entry_points_resolve():
    """run.py must be able to reach the functions it imports."""
    root = pathlib.Path(__file__).resolve().parent.parent
    source = (root / "run.py").read_text(encoding="utf-8")
    assert "from app.controllers.main import main" in source
    assert "from app.views.demo import main" in source


# ----------------------------------------------------------------------
# live startup - the dynamic counterpart to the static checks above
# ----------------------------------------------------------------------


def test_application_starts_headless_without_audio(monkeypatch, tmp_path):
    """Construct the real Application and walk its startup path.

    Runs offscreen with no devices configured, which is exactly the state a
    new user is in. It does not enter the Qt event loop - it checks that
    everything run() touches exists and behaves before exec() is reached.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication

    from app.controllers.main import Application
    from app.models.config import Config
    from app.models.state import Colour, Mode

    if QApplication.instance() is None:
        QApplication([])

    cfg = Config()
    cfg.save(tmp_path / "settings.json")
    app = Application(cfg)

    # No devices are configured, so audio must decline politely, not explode.
    ports, problems = app.build_audio()
    assert ports is None
    assert problems, "an unconfigured install must say what is missing"

    app._install_signal_handlers()
    assert app._sigint_timer.isActive(), "Ctrl+C would be swallowed"

    app._show_ui()
    assert app.bubble.isVisible()

    app._setup_prompted = True  # do not open a modal dialog inside a test
    app._setup_needed(problems)
    assert app.bubble._colour is Colour.RED

    # The transcript follows the mode (§ user flow: it opens on Start).
    app._on_state({"mode": Mode.TRANSLATING.value, "colour": Colour.GREEN.value,
                   "reason": ""})
    assert app.transcript.isVisible()
    app._on_state({"mode": Mode.PASSTHROUGH.value, "colour": Colour.GREY.value,
                   "reason": ""})
    assert not app.transcript.isVisible()

    app._cleanup()


# ----------------------------------------------------------------------
# build_audio's contract
# ----------------------------------------------------------------------


def test_build_audio_always_returns_a_pair_statically():
    """Every return in build_audio must be (ports, problems).

    The success path is the one that matters and the one no test on a
    non-Windows machine can ever reach, because DeviceManager refuses first.
    A returned bare AudioPorts crashes the caller unpacking it.
    """
    main = APP / "controllers" / "main.py"
    tree = ast.parse(main.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "build_audio"
    )
    bad = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        # Skip the returns belonging to the nested resolve() helper.
        inner = [
            f for f in ast.walk(fn)
            if isinstance(f, ast.FunctionDef) and f is not fn
            and f.lineno <= node.lineno <= (f.end_lineno or f.lineno)
        ]
        if inner:
            continue
        if not (isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2):
            bad.append(node.lineno)
    assert not bad, f"build_audio returns a non-pair at line(s) {bad}"


def test_build_audio_success_path_runs(monkeypatch, tmp_path):
    """Exercise the success path with a faked audio layer.

    Windows-only code cannot run here, but its SHAPE can: substituting the
    device and stream classes lets the whole of build_audio execute, which is
    how the bare-AudioPorts return should have been caught.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app.controllers import main as main_mod
    from app.models.config import Config

    if QApplication.instance() is None:
        QApplication([])

    class FakeDevice:
        def __init__(self, name):
            self.name = name
            self.can_capture = True
            self.can_render = True
            self.default_rate = 48_000
            self.max_input_channels = 2
            self.max_output_channels = 2

    class FakeDeviceManager:
        def __init__(self):
            self.closed = False

        def find(self, needle, capture=None):
            return FakeDevice(needle)

        def loopback_for_output(self, dev):
            return FakeDevice(f"{dev.name} [Loopback]")

        def default_input(self):
            return FakeDevice("System default microphone")

        def default_output(self):
            return FakeDevice("System default speakers")

        def close(self):
            self.closed = True

    class FakeStream:
        def __init__(self, dm, device, **kw):
            from app.audio.ring import RingBuffer

            self.device = device
            self.rate = 48_000
            self.channels = 2
            self.ring = RingBuffer(48_000)
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

        def write(self, pcm):
            return self.ring.write(pcm)

        def flush(self):
            return self.ring.discard()

        def health(self):
            return self.ring.health()

    class FakeWire:
        def __init__(self, src, sink, **kw):
            self.source, self.sink = src, sink
            self.muted = False

        def start(self): ...
        def stop(self): ...
        def mute(self):
            self.muted = True

        def unmute(self):
            self.muted = False

        def health(self):
            return {}

    class FakePair:
        def __init__(self, outgoing, incoming):
            self.outgoing, self.incoming = outgoing, incoming

        def start(self): ...
        def stop(self): ...
        def mute_all(self): ...
        def unmute_all(self): ...

    monkeypatch.setattr(main_mod, "DeviceManager", FakeDeviceManager)
    monkeypatch.setattr(main_mod, "CaptureSource", FakeStream)
    monkeypatch.setattr(main_mod, "RenderSink", FakeStream)
    monkeypatch.setattr(main_mod, "PassthroughWire", FakeWire)
    monkeypatch.setattr(main_mod, "PassthroughPair", FakePair)

    cfg = Config()
    cfg.devices.microphone = "Headset Mic"
    cfg.devices.headphones = "Headset Earphone"
    cfg.devices.virtual_mic = "CABLE Input"
    cfg.devices.teams_speaker = "CABLE-B Input"

    app = main_mod.Application(cfg)
    result = app.build_audio()

    assert isinstance(result, tuple) and len(result) == 2, (
        "build_audio must return (ports, problems); the caller unpacks it"
    )
    ports, problems = result
    assert ports is not None, problems
    assert not problems
    assert ports.mic_source.started and ports.vmic_sink.started

    # And the next path a real machine reaches: bringing the engine up. It
    # must start in PASSTHROUGH with audio already moving (spec 2.5), long
    # before anything translates.
    from app.models.state import Mode

    app._start_engine(ports)
    assert app.engine is not None
    assert app.engine.modes.mode is Mode.PASSTHROUGH
    assert app._metrics_timer.isActive()

    snap = app.engine.snapshot()
    assert snap["state"]["mode"] == "passthrough"
    assert "incoming" in snap["metrics"] and "outgoing" in snap["metrics"]

    app.engine.shutdown()
    app._cleanup()


def test_unset_mic_and_speakers_fall_back_to_the_system_default(monkeypatch):
    """Only the two virtual cables are genuinely required.

    Windows already knows which microphone and speakers you use; making
    somebody choose them before the application will run is busywork.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app.controllers import main as main_mod
    from app.models.config import Config

    if QApplication.instance() is None:
        QApplication([])

    class Dev:
        def __init__(self, name):
            self.name = name
            self.can_capture = self.can_render = True
            self.default_rate = 48_000
            self.max_input_channels = self.max_output_channels = 2

    class DM:
        def find(self, needle, capture=None):
            if "missing" in needle:
                from app.audio.devices import AudioUnavailable

                raise AudioUnavailable(f"no endpoint matching {needle!r}")
            return Dev(needle)

        def loopback_for_output(self, dev):
            return Dev(dev.name + " [LB]")

        def default_input(self):
            return Dev("System default microphone")

        def default_output(self):
            return Dev("System default speakers")

        def close(self): ...

    monkeypatch.setattr(main_mod, "DeviceManager", DM)

    # nothing configured at all
    app = main_mod.Application(Config())
    ports, problems = app.build_audio()
    assert ports is None
    joined = " ".join(problems)
    assert "Translator microphone" in joined and "Teams speaker" in joined
    assert "Microphone:" not in joined, "the mic has a default; it is not a problem"
    assert "Headphones:" not in joined, "speakers have a default too"

    # a remembered device that has since been unplugged must not be fatal
    cfg = Config()
    cfg.devices.microphone = "missing headset"
    app2 = main_mod.Application(cfg)
    ports, problems = app2.build_audio()
    assert not any(p.startswith("Microphone:") for p in problems), (
        "an unplugged microphone should fall back, not block startup"
    )
    app._cleanup()
    app2._cleanup()


# ----------------------------------------------------------------------
# feedback loops (§5.6 F1/F2) - the check that must run before a stream opens
# ----------------------------------------------------------------------


class _Dev:
    def __init__(self, name):
        self.name = name


def test_virtual_mic_on_an_audible_device_is_refused():
    """F1: your voice rendered into something you can hear, heard back."""
    speakers = _Dev("Speaker (Realtek(R) Audio)")
    from app.audio.devices import feedback_loop_problems

    found = feedback_loop_problems(
        _Dev("Microphone Array"), speakers, speakers, _Dev("CABLE-B Input")
    )
    assert found and "Feedback loop" in found[0]


def test_capturing_the_device_we_render_into_is_refused():
    """F2: our own output re-enters its own capture and compounds."""
    speakers = _Dev("Speaker (Realtek(R) Audio)")
    from app.audio.devices import feedback_loop_problems

    found = feedback_loop_problems(
        _Dev("Microphone Array"), speakers, _Dev("CABLE Input"), speakers
    )
    assert found and "re-enter" in found[0]


def test_one_cable_used_for_both_roles_is_warned_about_not_refused():
    """A deliberate decision, recorded here because it reverses an earlier one.

    It IS a loop - what we render into the cable is what we capture out of it -
    but refusing it outright meant no two-way mode at all on a machine with one
    cable. So it opens, and says what the user is about to hear."""
    cable = _Dev("CABLE Input")
    from app.audio.devices import feedback_loop_problems, feedback_loop_warnings

    assert not feedback_loop_problems(
        _Dev("Mic"), _Dev("Headphones"), cable, cable
    ), "a single-cable setup must still be allowed to open"

    warnings = feedback_loop_warnings(
        _Dev("Mic"), _Dev("Headphones"), cable, cable
    )
    assert warnings and "come back as the client speaking" in warnings[0]
    assert "VB-CABLE A+B" in warnings[0], "the way out must be named"


def test_two_different_cables_warn_about_nothing():
    from app.audio.devices import feedback_loop_warnings

    assert not feedback_loop_warnings(
        _Dev("Mic"), _Dev("Headphones"), _Dev("CABLE-A Input"),
        _Dev("CABLE-B Input"),
    )


def test_a_sane_binding_raises_nothing():
    from app.audio.devices import feedback_loop_problems

    assert not feedback_loop_problems(
        _Dev("Headset Mic"),
        _Dev("Headset Earphone"),
        _Dev("CABLE Input"),
        _Dev("CABLE-B Input"),
    )


def test_a_looping_configuration_never_opens_a_stream(monkeypatch):
    """The whole point: refuse before any audio device is touched.

    This is the configuration the user's machine actually produces - one real
    output device and no virtual cable - and it must not reach the point of
    making a sound.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from app.controllers import main as main_mod
    from app.models.config import Config

    if QApplication.instance() is None:
        QApplication([])

    opened: list[str] = []

    class Dev:
        def __init__(self, name):
            self.name = name
            self.can_capture = self.can_render = True
            self.default_rate = 48_000
            self.max_input_channels = self.max_output_channels = 2

    class DM:
        def find(self, needle, capture=None):
            return Dev(needle)

        def loopback_for_output(self, dev):
            return Dev(dev.name + " [Loopback]")

        def default_input(self):
            return Dev("Microphone Array")

        def default_output(self):
            return Dev("Speaker (Realtek(R) Audio)")

        def close(self): ...

    class Stream:
        def __init__(self, dm, device, **kw):
            opened.append(device.name)

        def start(self):
            opened.append("STARTED")

    monkeypatch.setattr(main_mod, "DeviceManager", DM)
    monkeypatch.setattr(main_mod, "CaptureSource", Stream)
    monkeypatch.setattr(main_mod, "RenderSink", Stream)

    cfg = Config()
    # Exactly what you get with no virtual cable installed: everything points
    # at the one real output device.
    cfg.devices.virtual_mic = "Speaker (Realtek(R) Audio)"
    cfg.devices.teams_speaker = "Speaker (Realtek(R) Audio)"

    app = main_mod.Application(cfg)
    ports, problems = app.build_audio()

    assert ports is None
    assert any("Feedback loop" in p or "cables" in p for p in problems), problems
    assert "STARTED" not in opened, (
        "a looping configuration must be refused before any stream opens"
    )
    app._cleanup()


# ----------------------------------------------------------------------
# routing, done by the application instead of by hand
# ----------------------------------------------------------------------


def test_a_call_routes_the_microphone_as_well_as_the_speaker():
    """Teams records from the OUTPUT half of the cable we render into, so both
    flows have to move for Teams to be left on "Default" for both."""
    from app.controllers.engine import Engine

    moved: list[tuple[str, bool]] = []

    class FakeRouter:
        def route(self, name, capture=False):
            moved.append((name, capture))
            return "previous-id"

        def restore(self):
            pass

    engine = Engine.__new__(Engine)
    engine.router = FakeRouter()
    engine.config = _config_with_route("CABLE-B Input", auto=True)
    engine.config.mode = "meeting"
    engine.config.devices.virtual_mic = "CABLE-A Input (VB-Audio Cable A)"
    engine._notice = lambda *a, **k: None

    # The recording half is found by WIRE - the driver in brackets - not by
    # replacing "Input" with "Output", which does nothing at all for names
    # like "CABLE In 16 Ch (2- VB-Audio Virtual Cable)".
    import app.audio.routing as routing

    original = routing.endpoints
    routing.endpoints = lambda capture=False: (
        [("CABLE-A Output (VB-Audio Cable A)", "a-out-id")] if capture else []
    )
    try:
        engine._route_audio()
    finally:
        routing.endpoints = original

    assert moved == [
        ("CABLE-B Input", False),
        ("CABLE-A Output (VB-Audio Cable A)", True),
    ], moved


def test_one_way_listening_leaves_the_microphone_alone():
    """Nothing is being sent anywhere, so taking over the default recording
    device would affect every other application for no reason."""
    from app.controllers.engine import Engine

    moved: list[tuple[str, bool]] = []

    class FakeRouter:
        def route(self, name, capture=False):
            moved.append((name, capture))
            return "previous-id"

        def restore(self):
            pass

    engine = Engine.__new__(Engine)
    engine.router = FakeRouter()
    engine.config = _config_with_route("CABLE Input", auto=True)
    engine.config.mode = "video"
    engine.config.devices.virtual_mic = "CABLE-A Input"
    engine._notice = lambda *a, **k: None

    engine._route_audio()
    assert moved == [("CABLE Input", False)], moved


def test_a_single_cable_setup_is_warned_about_at_start():
    from app.controllers.engine import Engine

    said: list[tuple[str, str]] = []

    class FakeRouter:
        def route(self, name, capture=False):
            return ""

        def restore(self):
            pass

    engine = Engine.__new__(Engine)
    engine.router = FakeRouter()
    engine.config = _config_with_route("CABLE Input", auto=False)
    engine.config.mode = "meeting"
    engine.config.devices.virtual_mic = "CABLE Input"
    engine.config.devices.teams_speaker = "CABLE Input"
    engine._notice = lambda severity, message, *a: said.append((severity, message))

    engine._route_audio()
    assert said and "come back as the client speaking" in said[0][1]


def test_the_engine_routes_on_start_and_restores_on_stop():
    """The user's ask: do not make me change Windows settings by hand, and put
    them back when you close."""
    from app.controllers.engine import Engine

    moved: list[str] = []

    class FakeRouter:
        routed = False

        def route(self, name, capture=False):
            moved.append(f"route:{name}")
            self.routed = True
            return "speakers-id"

        def restore(self):
            if self.routed:
                moved.append("restore")
                self.routed = False

    engine = Engine.__new__(Engine)          # no audio devices required
    engine.router = FakeRouter()
    engine.config = _config_with_route("CABLE Input", auto=True)
    engine._notice = lambda *a, **k: None

    engine._route_audio()
    engine.router.restore()
    assert moved == ["route:CABLE Input", "restore"]


def test_nothing_is_touched_when_the_user_turned_routing_off():
    from app.controllers.engine import Engine

    moved: list[str] = []

    class FakeRouter:
        def route(self, name, capture=False):
            moved.append(name)
            return ""

        def restore(self):
            pass

    engine = Engine.__new__(Engine)
    engine.router = FakeRouter()
    engine.config = _config_with_route("CABLE Input", auto=False)
    engine._notice = lambda *a, **k: None

    engine._route_audio()
    assert moved == []


def test_a_routing_failure_is_a_notice_not_a_refusal_to_translate():
    """Failing to route leaves the user exactly where they were before this
    feature existed - doing it by hand. Refusing to translate over it would be
    strictly worse."""
    from app.controllers.engine import Engine
    from app.audio.routing import RoutingUnavailable

    said: list[str] = []

    class BrokenRouter:
        def route(self, name, capture=False):
            raise RoutingUnavailable("no such device")

        def restore(self):
            pass

    engine = Engine.__new__(Engine)
    engine.router = BrokenRouter()
    engine.config = _config_with_route("CABLE Input", auto=True)
    engine._notice = lambda severity, message, *a: said.append((severity, message))

    engine._route_audio()   # must not raise
    assert said and said[0][0] == "warning"
    assert "Volume mixer" in said[0][1], "the manual fallback was not explained"


def _config_with_route(device: str, *, auto: bool):
    from app.models.config import Config

    config = Config()
    config.devices.teams_speaker = device
    config.devices.auto_route = auto
    return config


# ----------------------------------------------------------------------
# half duplex, so speakers do not send our own Urdu back as English
# ----------------------------------------------------------------------


def test_the_microphone_is_ignored_while_the_urdu_is_playing():
    """The user said speakers must be usable. On speakers the microphone hears
    the Urdu, so without this it is translated back to English and sent to the
    client - who hears their own words returned, louder each pass."""
    from app.controllers.engine import Engine
    from app.models.config import Config

    engine = Engine.__new__(Engine)
    engine.config = Config()

    class Speaking:
        is_speaking = True

    class Quiet:
        is_speaking = False

    engine._incoming = Speaking()
    assert engine._muted_by_playback() is True

    engine._incoming = Quiet()
    assert engine._muted_by_playback() is False


def test_half_duplex_can_be_turned_off():
    """On headphones it never triggers anyway, and somebody with echo
    cancellation in hardware should be able to talk over the translation."""
    from app.controllers.engine import Engine
    from app.models.config import Config

    engine = Engine.__new__(Engine)
    engine.config = Config()
    engine.config.devices.half_duplex = False

    class Speaking:
        is_speaking = True

    engine._incoming = Speaking()
    assert engine._muted_by_playback() is False


def test_nothing_is_muted_when_there_is_no_incoming_pipeline():
    from app.controllers.engine import Engine
    from app.models.config import Config

    engine = Engine.__new__(Engine)
    engine.config = Config()
    engine._incoming = None
    assert engine._muted_by_playback() is False


# ----------------------------------------------------------------------
# two endpoints of one cable are not two cables
# ----------------------------------------------------------------------


def test_the_free_vb_cable_counts_as_one_cable_not_two():
    """Straight off the user's screen. VB-Audio's free driver installs ONE
    cable and exposes it twice:

        CABLE Input (VB-Audio Virtual Cable)
        CABLE In 16ch (VB-Audio Virtual Cable)

    Counting endpoints said "this machine has 2" and offered both to the two
    roles - which is a loop with extra steps, and left somebody certain they
    had installed a second cable when they had not.
    """
    from app.audio.devices import cable_family, distinct_cables

    endpoints = [
        "CABLE In 16ch (VB-Audio Virtual Cable)",
        "CABLE Input (VB-Audio Virtual Cable)",
    ]
    assert len({cable_family(name) for name in endpoints}) == 1
    assert distinct_cables(endpoints) == ["CABLE Input (VB-Audio Virtual Cable)"], (
        "the plain endpoint should be preferred over the 16-channel one"
    )


def test_the_a_plus_b_pack_really_is_two_more():
    from app.audio.devices import distinct_cables

    got = distinct_cables([
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE In 16ch (VB-Audio Virtual Cable)",
        "CABLE-A Input (VB-Audio Cable A)",
        "CABLE-B Input (VB-Audio Cable B)",
    ])
    assert len(got) == 3


def test_voicemeeter_cables_are_distinct():
    from app.audio.devices import distinct_cables

    got = distinct_cables([
        "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
        "VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)",
    ])
    assert len(got) == 2


def test_a_name_with_no_brackets_is_its_own_family():
    from app.audio.devices import cable_family, distinct_cables

    assert cable_family("Some Cable") == "some cable"
    assert len(distinct_cables(["Some Cable", "Another Cable"])) == 2


def test_the_message_says_cables_not_endpoints():
    """The count in the refusal has to be cables, and the extra endpoint has
    to be named as what it is."""
    from app.controllers.autoconfig import autoconfigure
    from app.models.config import Config
    from tests.test_listen import FakeManagerWithDefaults

    cfg = Config()
    mgr = FakeManagerWithDefaults([
        "Speaker (Realtek(R) Audio)",
        "CABLE Input (VB-Audio Virtual Cable)",
        "CABLE In 16ch (VB-Audio Virtual Cable)",
    ])
    _notes, problems = autoconfigure(cfg, mgr, save=False)

    assert len(problems) == 1
    text = problems[0]
    assert "has 1 (" in text, text
    assert "same wire, not a second one" in text, text
    assert "VB-CABLE A+B" in text


# ----------------------------------------------------------------------
# the free route: VoiceMeeter instead of a paid second cable
# ----------------------------------------------------------------------


class _Endpoint:
    def __init__(self, name, ins=0, outs=2, loopback=False):
        self.name = name
        self._ins = ins
        self._outs = outs
        self.is_loopback = loopback

    @property
    def can_capture(self):
        return self._ins > 0

    @property
    def can_render(self):
        return self._outs > 0


class _Manager:
    def __init__(self, endpoints):
        self._endpoints = endpoints

    def wasapi_devices(self):
        return self._endpoints

    def default_input(self):
        return self._endpoints[-1]

    def default_output(self):
        return self._endpoints[0]


def _realistic_machine():
    """A Windows machine as PortAudio actually reports it, loopbacks and all."""
    return [
        _Endpoint("Speaker (Realtek(R) Audio)", outs=2),
        _Endpoint("Speaker (Realtek(R) Audio)", ins=2, outs=0, loopback=True),
        _Endpoint("CABLE Input (VB-Audio Virtual Cable)", outs=2),
        _Endpoint("CABLE In 16ch (VB-Audio Virtual Cable)", outs=16),
        _Endpoint("Microphone Array (Intel Smart Sound)", ins=2, outs=0),
    ]


def test_the_microphone_is_never_a_speaker_loopback():
    """WASAPI exposes a loopback capture endpoint for every playback device,
    named after it - so "the first device that can capture" is frequently the
    loopback of the speakers. Chosen as the microphone, the outgoing direction
    sends the client whatever the machine is playing instead of the user's
    voice."""
    from app.controllers.autoconfig import autoconfigure
    from app.models.config import Config

    config = Config()
    autoconfigure(config, _Manager(_realistic_machine()), save=False)
    assert config.devices.microphone == "Microphone Array (Intel Smart Sound)"


def test_voicemeeter_provides_the_second_cable_for_free():
    """The paid A+B pack is not the only way. VoiceMeeter is donationware and
    exposes its virtual inputs as distinct cables, which is all two-way mode
    needs."""
    from app.controllers.autoconfig import autoconfigure
    from app.models.config import Config

    endpoints = _realistic_machine() + [
        _Endpoint("VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", outs=2),
        _Endpoint("VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)", outs=2),
    ]
    config = Config()
    _notes, problems = autoconfigure(config, _Manager(endpoints), save=False)

    assert problems == [], problems
    assert config.devices.virtual_mic
    assert config.devices.teams_speaker
    assert config.devices.virtual_mic != config.devices.teams_speaker


def test_one_cable_alone_still_blocks_with_the_honest_count():
    from app.controllers.autoconfig import autoconfigure
    from app.models.config import Config

    config = Config()
    _notes, problems = autoconfigure(
        config, _Manager(_realistic_machine()), save=False
    )
    assert len(problems) == 1
    assert "has 1 (" in problems[0]


def test_the_machine_this_was_debugged_on():
    """Every endpoint name here is verbatim from the user's --call-check output
    after installing VoiceMeeter beside the free VB-CABLE. Three bugs at once:

      * Windows had renamed "CABLE Input (VB-Audio Virtual Cable)" to
        "Speakers (2- VB-Audio Virtual Cable)", so the saved name resolved to
        nothing and the report said the cable was missing.
      * "Speakers (2- ...)" and "CABLE In 16 Ch (2- ...)" are one wire under
        two names, and a distinctness check that compared strings put one in
        each role.
      * Every VoiceMeeter cable reports the same driver in brackets, so
        grouping by driver collapsed nine usable cables into one.
    """
    from app.audio.devices import cable_family
    from app.controllers.autoconfig import autoconfigure
    from app.models.config import Config

    endpoints = [
        _Endpoint("CABLE In 16 Ch (2- VB-Audio Virtual Cable)", outs=16),
        _Endpoint("Voicemeeter In 3 (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Voicemeeter VAIO3 Input (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Voicemeeter In 4 (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Voicemeeter In 2 (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Speaker (Realtek(R) Audio)", outs=2),
        _Endpoint("Speaker (Realtek(R) Audio)", ins=2, outs=0, loopback=True),
        _Endpoint("Voicemeeter In 1 (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Speakers (2- VB-Audio Virtual Cable)", outs=2),
        _Endpoint("Voicemeeter In 5 (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Voicemeeter Input (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Microphone Array (Intel Smart Sound)", ins=2, outs=0),
    ]

    config = Config()
    config.devices.microphone = "Microphone Array (Intel Smart Sound)"
    config.devices.headphones = "Speaker (Realtek(R) Audio)"
    config.devices.teams_speaker = "CABLE Input (VB-Audio Virtual Cable)"  # gone
    config.devices.virtual_mic = "CABLE In 16 Ch (2- VB-Audio Virtual Cable)"

    notes, problems = autoconfigure(config, _Manager(endpoints), save=False)

    assert problems == [], problems
    assert any("no endpoint of that name exists" in note for note in notes)
    assert config.devices.virtual_mic and config.devices.teams_speaker
    assert cable_family(config.devices.virtual_mic) != cable_family(
        config.devices.teams_speaker
    ), "both roles ended up on the same wire"


def test_voicemeeter_cables_are_not_collapsed_into_one():
    from app.audio.devices import distinct_cables

    got = distinct_cables([
        "Voicemeeter Input (VB-Audio Voicemeeter VAIO)",
        "Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)",
        "Voicemeeter VAIO3 Input (VB-Audio Voicemeeter VAIO)",
    ])
    assert len(got) == 3


def test_a_renamed_cable_is_still_the_same_wire():
    from app.audio.devices import cable_family

    assert cable_family("Speakers (2- VB-Audio Virtual Cable)") == cable_family(
        "CABLE In 16 Ch (2- VB-Audio Virtual Cable)"
    )


def test_a_16_channel_endpoint_with_a_space_is_recognised():
    """"16ch" and "16 Ch" are the same thing, and a substring test only caught
    the first."""
    from app.audio.devices import distinct_cables

    got = distinct_cables([
        "CABLE In 16 Ch (2- VB-Audio Virtual Cable)",
        "Speakers (2- VB-Audio Virtual Cable)",
    ])
    assert got == ["Speakers (2- VB-Audio Virtual Cable)"]


def test_the_recording_half_of_a_cable_is_found_by_wire():
    """The exact name from the user's machine, which the string-surgery version
    left untouched: "CABLE In 16 Ch (2- VB-Audio Virtual Cable)" contains no
    "Input" to replace, so the default microphone was pointed at a playback
    device's name and routing reported itself unavailable."""
    import app.audio.routing as routing

    original = routing.endpoints
    routing.endpoints = lambda capture=False: (
        [
            ("Microphone Array (Intel Smart Sound)", "mic-id"),
            ("CABLE Output (2- VB-Audio Virtual Cable)", "cable-out-id"),
        ]
        if capture
        else []
    )
    try:
        got = routing.recording_partner("CABLE In 16 Ch (2- VB-Audio Virtual Cable)")
    finally:
        routing.endpoints = original
    assert got == "CABLE Output (2- VB-Audio Virtual Cable)"


def test_a_cable_with_no_recording_half_is_reported():
    import app.audio.routing as routing

    original = routing.endpoints
    routing.endpoints = lambda capture=False: [("Microphone Array", "mic")]
    try:
        with pytest.raises(routing.RoutingUnavailable):
            routing.recording_partner("CABLE In 16 Ch (2- VB-Audio Virtual Cable)")
    finally:
        routing.endpoints = original


def test_the_sending_cable_must_have_a_recording_half():
    """The asymmetry that broke a live call. What we capture FROM Teams is
    taken by WASAPI loopback off a render endpoint, so any playback cable
    works. What we SEND to Teams has to appear as a recording device Teams can
    select - and VoiceMeeter's recording endpoints are mixer buses, not fixed
    partners of its inputs. Given the VoiceMeeter cable for that role, the
    client hears nothing from us at all."""
    from app.audio.devices import cable_family
    from app.controllers.autoconfig import autoconfigure
    from app.models.config import Config

    endpoints = [
        _Endpoint("Speaker (Realtek(R) Audio)", outs=2),
        _Endpoint("Speakers (2- VB-Audio Virtual Cable)", outs=2),
        _Endpoint("CABLE In 16 Ch (2- VB-Audio Virtual Cable)", outs=16),
        _Endpoint("Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)", outs=2),
        _Endpoint("Microphone Array (Intel Smart Sound)", ins=2, outs=0),
        _Endpoint("CABLE Output (2- VB-Audio Virtual Cable)", ins=2, outs=0),
        _Endpoint("Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)", ins=2, outs=0),
    ]

    config = Config()
    # The broken state from the live attempt: VoiceMeeter in the sending role.
    config.devices.virtual_mic = "Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)"
    config.devices.teams_speaker = "Speakers (2- VB-Audio Virtual Cable)"

    notes, problems = autoconfigure(config, _Manager(endpoints), save=False)

    assert problems == [], problems
    assert any("no recording half" in note for note in notes)
    assert cable_family(config.devices.virtual_mic) == cable_family(
        "CABLE Output (2- VB-Audio Virtual Cable)"
    ), "the sending role must be the cable that has a capture half"
    assert cable_family(config.devices.teams_speaker) != cable_family(
        config.devices.virtual_mic
    )
