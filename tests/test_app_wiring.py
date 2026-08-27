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


def test_one_cable_used_for_both_roles_is_refused():
    cable = _Dev("CABLE Input")
    from app.audio.devices import feedback_loop_problems

    found = feedback_loop_problems(
        _Dev("Mic"), _Dev("Headphones"), cable, cable
    )
    assert found and "Two separate cables" in found[0]


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
