"""Application entry point.

Spec reference: §2.5, §2.6, §11.5, §14.

STARTUP ORDER, AND WHY IT IS THIS ORDER
---------------------------------------
    1. open the four audio endpoints
    2. start the passthrough wires        <- audio is now flowing
    3. start the engine loop
    4. show the tray icon and the bubble

Audio comes up before the UI, and long before anything translates. Teams is
bound to our virtual microphone permanently: if that endpoint is not being
fed, the client hears silence and does not know why. The wire starts with the
application and stops with it, never with the Start button (§2.5).

The application is meant to launch at login so the wire is up before Teams is.
"""

from __future__ import annotations

import contextlib
import signal
import sys

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from app.audio.devices import (
    AudioUnavailable,
    DeviceManager,
    feedback_loop_problems,
)
from app.audio.passthrough import PassthroughPair, PassthroughWire
from app.audio.streams import CaptureSource, RenderSink
from app.providers.base import ProviderMisconfigured
from app.models.config import Config, get_secret
from app.controllers.engine import AudioPorts, Engine
from app.models.state import Colour, Mode
from app.models.terminology import Terminology
from app.views import capture_guard, errors
from app.views.hotkeys import HotkeyManager
from app.views.overlay import RING, Bubble, Panel
from app.views.transcript import TranscriptWindow


class Bridge(QObject):
    """Marshals engine-thread events onto the Qt thread.

    Spec reference: §11.5. Engine to UI is Qt signals only, delivered by
    queued connection. Payloads are immutable value objects. The UI never sees
    per-frame or per-hypothesis events - only aggregated updates.
    """

    state_changed = Signal(dict)
    transcript = Signal(dict)
    notice = Signal(dict)


def make_icon(colour: QColor) -> QIcon:
    pm = QPixmap(32, 32)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(28, 30, 34))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, 28, 28)
    pen = p.pen()
    pen.setColor(colour)
    pen.setWidth(3)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    p.drawEllipse(5, 5, 22, 22)
    p.end()
    return QIcon(pm)


class Application:
    def __init__(self, config: Config) -> None:
        self.config = config
        # Reuse an existing instance rather than constructing a second one:
        # Qt allows exactly one per process, and blindly creating it makes the
        # class impossible to test and impossible to embed.
        self.qt = QApplication.instance() or QApplication(sys.argv)
        self.qt.setQuitOnLastWindowClosed(False)  # the tray owns the lifetime

        self.bridge = Bridge()
        self.bubble = Bubble()
        self.panel = Panel()
        self.transcript = TranscriptWindow()
        self.hotkeys = HotkeyManager()

        self._dm: DeviceManager | None = None
        self._streams: list = []
        self.engine: Engine | None = None
        self._setup_prompted = False

        errors.install(self.qt)
        self._wire_ui()

    # ------------------------------------------------------------------
    # audio
    # ------------------------------------------------------------------

    def build_audio(self) -> tuple[AudioPorts | None, list[str]]:
        """Open the four endpoints, collecting problems instead of raising.

        Every role is resolved independently and every failure is reported, so
        the user sees the complete list of what to fix in one pass rather than
        one error, one fix, one error. If any role is missing we return None
        and the application still starts - in a visibly broken state, which is
        far more useful than exiting with a stack trace (§14.1).
        """
        problems: list[str] = []
        notes: list[str] = []
        d = self.config.devices

        try:
            self._dm = DeviceManager()
        except AudioUnavailable as exc:
            return None, [str(exc)]
        dm = self._dm

        def resolve(name: str, role: str, capture: bool, fallback=None):
            """Resolve one endpoint, falling back to the Windows default.

            Your own microphone and headphones have a sensible default - the
            one Windows is already using - so there is no reason to make
            anyone pick them before the application will run. The two virtual
            cables have no default: nothing else on the machine can stand in
            for them, so those are the only genuinely required choices.
            """
            if not name:
                if fallback is None:
                    problems.append(
                        f"{role}: not set. There is no default for this - it "
                        "must be a virtual audio cable."
                    )
                    return None
                try:
                    dev = fallback()
                    notes.append(f"{role}: using system default — {dev.name}")
                    return dev
                except AudioUnavailable as exc:
                    problems.append(f"{role}: no system default ({exc})")
                    return None
            try:
                return dm.find(name, capture=capture)
            except AudioUnavailable as exc:
                # A remembered device that has since been unplugged should not
                # stop the application - fall back and say so.
                if fallback is not None:
                    with contextlib.suppress(AudioUnavailable):
                        dev = fallback()
                        notes.append(
                            f"{role}: {name!r} not found, using system "
                            f"default — {dev.name}"
                        )
                        return dev
                problems.append(f"{role}: {exc}")
                return None

        mic_dev = resolve(d.microphone, "Microphone", True, dm.default_input)
        phones_dev = resolve(d.headphones, "Headphones", False, dm.default_output)
        vmic_dev = resolve(d.virtual_mic, "Translator microphone", False)
        teams_dev = resolve(d.teams_speaker, "Teams speaker", False)

        teams_lb = None
        if teams_dev is not None:
            try:
                teams_lb = dm.loopback_for_output(teams_dev)
            except AudioUnavailable as exc:
                problems.append(f"Teams speaker loopback: {exc}")

        problems.extend(
            feedback_loop_problems(mic_dev, phones_dev, vmic_dev, teams_dev)
        )

        for note in notes:
            self._on_notice({"severity": "info", "message": note})

        if None in (mic_dev, phones_dev, vmic_dev) or teams_lb is None or problems:
            return None, problems

        mic = CaptureSource(dm, mic_dev)
        phones = RenderSink(dm, phones_dev)
        vmic = RenderSink(dm, vmic_dev)
        teams = CaptureSource(dm, teams_lb)

        self._streams = [mic, phones, vmic, teams]
        for s in self._streams:
            s.start()

        wires = PassthroughPair(
            outgoing=PassthroughWire(mic, vmic, name="mic->vmic"),
            # The headphone wire may emit true silence; only the path feeding
            # Teams needs a comfort-noise floor (§5.7).
            incoming=PassthroughWire(
                teams, phones, name="teams->phones", comfort_noise_dbfs=None
            ),
        )
        # Every return from this method is a (ports, problems) pair - the
        # success path included. Returning a bare AudioPorts here is a crash
        # at the call site, and one that only Windows ever reaches.
        return AudioPorts(mic, vmic, teams, phones, wires), problems

    def build_providers(self) -> dict:
        from app.providers.deepgram_asr import DeepgramSpeechToText
        from app.providers.llm_translator import LlmTranslator
        from app.providers.openai_s2st import OpenAiSpeechTranslator
        from app.providers.tts import (
            AzureTextToSpeech,
            ElevenLabsTextToSpeech,
            NullTextToSpeech,
        )

        p = self.config.providers
        if p.tts.startswith("azure"):
            tts = AzureTextToSpeech(
                get_secret("AZURE_SPEECH_KEY"), get_secret("AZURE_SPEECH_REGION")
            )
        elif p.tts.startswith("eleven"):
            tts = ElevenLabsTextToSpeech(get_secret("ELEVENLABS_API_KEY"))
        else:
            tts = NullTextToSpeech()

        return {
            "s2st": OpenAiSpeechTranslator(get_secret("OPENAI_API_KEY")),
            "asr": DeepgramSpeechToText(get_secret("DEEPGRAM_API_KEY")),
            "translator": LlmTranslator(
                get_secret("OPENAI_API_KEY"), model=p.translator_model
            ),
            "tts": tts,
        }

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------

    def _wire_ui(self) -> None:
        self.bubble.clicked.connect(self._toggle_panel)
        self.bubble.moved.connect(self._save_position)

        self.panel.start_clicked.connect(lambda: self._safe(self._start))
        self.panel.pause_clicked.connect(lambda: self._safe(self._pause))
        self.panel.stop_clicked.connect(lambda: self._safe(self._stop))
        self.panel.transcript_clicked.connect(self._toggle_transcript)
        self.panel.settings_clicked.connect(self._open_settings)

        self.bridge.state_changed.connect(self._on_state)
        self.bridge.transcript.connect(self._on_transcript)
        self.bridge.notice.connect(self._on_notice)

        self.tray = QSystemTrayIcon(make_icon(RING[Colour.GREY]))
        menu = QMenu()
        self.act_start = QAction("Start translating")
        self.act_pause = QAction("Pause")
        self.act_stop = QAction("Stop")
        act_show = QAction("Show / hide bubble")
        act_tx = QAction("Transcript")
        act_quit = QAction("Quit")
        for a in (self.act_start, self.act_pause, self.act_stop):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(act_show)
        menu.addAction(act_tx)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Urdu ↔ English translator")

        self.act_start.triggered.connect(lambda: self._safe(self._start))
        self.act_pause.triggered.connect(lambda: self._safe(self._pause))
        self.act_stop.triggered.connect(lambda: self._safe(self._stop))
        act_show.triggered.connect(self._toggle_bubble)
        act_tx.triggered.connect(self._toggle_transcript)
        act_quit.triggered.connect(self.quit)

        self._metrics_timer = QTimer()
        self._metrics_timer.setInterval(250)  # 4 Hz (§11.5)
        self._metrics_timer.timeout.connect(self._tick_metrics)

    def _bind_hotkeys(self) -> None:
        u = self.config.ui
        self.hotkeys.bind(u.hotkey_pause, lambda: self._post(self._toggle_pause))
        self.hotkeys.bind(u.hotkey_start_stop, lambda: self._post(self._start_stop))
        self.hotkeys.bind(u.hotkey_hide, lambda: self._post(self._toggle_bubble))
        self.hotkeys.bind(u.hotkey_repeat, lambda: self._post(self._repeat))
        self.hotkeys.start()

        # A combination someone else owns should cost a different shortcut, not
        # the feature. Say which one it ended up as - a silently moved hotkey
        # is as useless as a missing one.
        moved = self.hotkeys.reassigned()
        if moved:
            self._on_notice(
                {
                    "severity": "info",
                    "message": "Hotkey already in use, moved: "
                    + ", ".join(f"{k} -> {v}" for k, v in moved.items()),
                }
            )
        if self.hotkeys.failed:
            self._on_notice(
                {
                    "severity": "warning",
                    "message": "No shortcut available for: "
                    + ", ".join(self.hotkeys.failed)
                    + ". Change them in Settings > Hotkeys.",
                }
            )

    def _post(self, fn) -> None:
        """Hop from the hotkey thread onto the Qt thread."""
        QTimer.singleShot(0, fn)

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------

    def _safe(self, fn) -> None:
        try:
            fn()
        except Exception as exc:  # a UI slot must never reach the engine
            self._on_notice({"severity": "error", "message": str(exc)})
            errors.show_error(str(exc), exc)

    def _start(self) -> None:
        if self.engine:
            self.engine.start_translating()

    def _pause(self) -> None:
        if self.engine:
            self.engine.pause()

    def _stop(self) -> None:
        if self.engine:
            self.engine.stop_translating()

    def _toggle_pause(self) -> None:
        if self.engine:
            self.engine.toggle_pause()

    def _start_stop(self) -> None:
        if not self.engine:
            return
        if self.engine.modes.mode is Mode.PASSTHROUGH:
            self.engine.start_translating()
        else:
            self.engine.stop_translating()

    def _repeat(self) -> None:
        self._on_notice(
            {"severity": "info", "message": "Repeat is not implemented yet."}
        )

    def _toggle_panel(self) -> None:
        if self.panel.isVisible():
            self.panel.hide()
        else:
            self.panel.popup_near(self.bubble)

    def _toggle_bubble(self) -> None:
        self.bubble.setVisible(not self.bubble.isVisible())
        if not self.bubble.isVisible():
            self.panel.hide()

    def _toggle_transcript(self) -> None:
        if self.transcript.isVisible():
            self.transcript.hide()
        else:
            self.transcript.show()
        self.config.ui.transcript_open = self.transcript.isVisible()

    def _open_settings(self) -> None:
        from app.views.settings import SettingsDialog

        dlg = SettingsDialog(self.config, None)
        if dlg.exec():  # Accepted == 1
            self._on_notice(
                {"severity": "info", "message": "Settings saved. Restarting audio…"}
            )
            self._restart_audio()

    def _restart_audio(self) -> None:
        """Re-open the endpoints after a settings change, without a restart.

        The user has just told us which devices to use; making them quit and
        relaunch to act on that would be absurd.
        """
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
        for s in self._streams:
            try:
                s.stop()
            except Exception:
                pass
        self._streams = []
        if self._dm is not None:
            try:
                self._dm.close()
            except Exception:
                pass
            self._dm = None

        ports, problems = self.build_audio()
        if ports is None:
            self._setup_needed(problems)
            return
        self._start_engine(ports)
        self._on_notice({"severity": "info", "message": "Audio running."})

    def _save_position(self, pos) -> None:
        self.config.ui.bubble_x = pos.x()
        self.config.ui.bubble_y = pos.y()
        self.config.save()

    # ------------------------------------------------------------------
    # engine events
    # ------------------------------------------------------------------

    def _on_state(self, state: dict) -> None:
        colour = Colour(state["colour"])
        self.bubble.set_state(colour)
        self.panel.apply_state(state)
        self.tray.setIcon(make_icon(RING[colour]))
        mode = Mode(state["mode"])
        self.act_start.setEnabled(mode is not Mode.TRANSLATING)
        self.act_pause.setEnabled(mode is Mode.TRANSLATING)
        self.act_stop.setEnabled(mode is not Mode.PASSTHROUGH)

        # The transcript is what the user actually watches while translating,
        # so it follows the mode rather than making them go and find it.
        if mode in (Mode.TRANSLATING, Mode.DEGRADED):
            if not self.transcript.isVisible():
                self.transcript.show()
        elif mode is Mode.PASSTHROUGH and self.transcript.isVisible():
            self.transcript.hide()

    def _on_transcript(self, payload: dict) -> None:
        self.transcript.add_entry(payload)
        text = payload.get("translated") or payload.get("source") or ""
        if payload.get("direction") == "incoming" and text:
            self.panel.set_quote(text[:120])

    def _on_notice(self, payload: dict) -> None:
        severity = payload.get("severity", "info")
        message = payload.get("message", "")
        self.transcript.add_notice(severity, message)
        if severity in ("error", "warning"):
            self.tray.showMessage("Translator", message, QSystemTrayIcon.Warning, 4000)

    def _tick_metrics(self) -> None:
        if not self.engine:
            return
        self.panel.apply_metrics(self.engine.metrics.snapshot())

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Setup first, then a two-button application.

        The flow is deliberately linear:

            not configured  ->  ONLY the setup window. Nothing else is shown,
                                because nothing else can do anything yet.
            configured      ->  the bubble and its panel: Start and Stop.
            translating     ->  the transcript opens by itself.

        Showing a dead overlay next to an unconfigured engine just gives the
        user buttons that do nothing.
        """
        self._install_signal_handlers()

        ports, problems = self.build_audio()
        if ports is None:
            if not self._run_setup(problems):
                return 0  # user cancelled setup; there is nothing to run
            ports, problems = self.build_audio()
            if ports is None:
                self._show_ui()
                self._setup_needed(problems)
                return self.qt.exec()

        self._show_ui()
        self._start_engine(ports)
        return self.qt.exec()

    def _install_signal_handlers(self) -> None:
        """Make Ctrl+C in the console actually quit.

        Qt's event loop blocks inside C, so Python never gets a chance to run
        its signal handlers and SIGINT is swallowed - the window stays up and
        the terminal appears frozen. A timer that fires periodically gives the
        interpreter the opening it needs.
        """
        signal.signal(signal.SIGINT, lambda *_: self.quit())
        self._sigint_timer = QTimer()
        self._sigint_timer.setInterval(200)
        self._sigint_timer.timeout.connect(lambda: None)
        self._sigint_timer.start()
        self.qt.aboutToQuit.connect(self._cleanup)

    def _show_ui(self) -> None:
        """Place and show the overlay, the tray icon and the hotkeys."""
        u = self.config.ui
        if u.bubble_x >= 0:
            self.bubble.move(u.bubble_x, u.bubble_y)
        else:
            area = self.qt.primaryScreen().availableGeometry()
            self.bubble.move(
                area.right() - self.bubble.width() - 40,
                area.bottom() - self.bubble.height() - 90,
            )
        self.bubble.show()
        self.bubble.clamp_to_screen()
        self.tray.show()
        self._bind_hotkeys()

        if u.exclude_from_capture and not self.bubble.capture_excluded:
            # Do not let this fail quietly - it is the §13.3 disclosure risk.
            self._on_notice(
                {
                    "severity": "warning",
                    "message": (
                        "The overlay could NOT be hidden from screen capture. "
                        "It will be visible if you share your screen. "
                        "Use the hide hotkey before sharing."
                    ),
                }
            )

    def _run_setup(self, problems: list[str]) -> bool:
        """Show only the settings window. True if the user saved."""
        from app.views.settings import SettingsDialog

        print("  Devices are not configured. Opening setup.\n")
        for p in problems:
            print(f"    - {p}")
        print()

        dlg = SettingsDialog(self.config, None)
        dlg.setWindowTitle("Set up the translator")
        return bool(dlg.exec())

    def _setup_needed(self, problems: list[str]) -> None:
        """Audio could not start. Say exactly what is missing, and stay open."""
        first_time = not self._setup_prompted
        self._setup_prompted = True
        self.bubble.set_state(Colour.RED, "!")
        self.panel.apply_state(
            {
                "mode": Mode.PASSTHROUGH.value,
                "colour": Colour.RED.value,
                "status": "NOT CONFIGURED",
                "reason": "Choose your devices in Settings to start",
            }
        )
        self.panel.set_devices(False, False, False)
        self.act_start.setEnabled(False)
        self.act_pause.setEnabled(False)
        self.act_stop.setEnabled(False)

        self.transcript.add_notice("error", "Audio could not start:")
        for p in problems:
            self.transcript.add_notice("error", f"   • {p}")
        self.transcript.add_notice(
            "info", "Open Settings from the panel to choose your devices."
        )
        self.transcript.show()
        self.panel.popup_near(self.bubble)
        self.panel._idle.stop()
        # Open settings once, unprompted. After that the user drives - looping
        # the dialog back at them on every failed save would be hostile.
        if first_time:
            self._open_settings()

    def _start_engine(self, ports: AudioPorts) -> None:
        self.engine = Engine(
            self.config,
            ports,
            self.build_providers(),
            terminology=Terminology.load(self.config.terminology_path()),
        )
        # Engine callbacks fire on the engine thread; the signals hop to Qt.
        self.engine.on_transcript = self.bridge.transcript.emit
        self.engine.on_notice = self.bridge.notice.emit
        self.engine.modes.subscribe(
            lambda snap: self.bridge.state_changed.emit(snap.as_dict())
        )

        self.engine.start()  # audio is now flowing, in PASSTHROUGH
        self._metrics_timer.start()
        self.bridge.state_changed.emit(self.engine.modes.snapshot().as_dict())

    def quit(self) -> None:
        self._cleanup()
        self.qt.quit()

    def _cleanup(self) -> None:
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True
        self._metrics_timer.stop()
        self.hotkeys.stop()
        if self.engine:
            self.engine.shutdown()
        for s in self._streams:
            try:
                s.stop()
            except Exception:
                pass
        if self._dm:
            self._dm.close()
        self.config.save()


def _report_startup_failure(message: str, exc: BaseException) -> None:
    import traceback

    print(f"\n  {message}\n")
    traceback.print_exception(type(exc), exc, exc.__traceback__)
    try:
        from PySide6.QtWidgets import QApplication

        from app.views import errors

        if QApplication.instance() is None:
            QApplication([])
        errors.show_error(message, exc, title="Could not start")
    except Exception:
        pass


def main() -> int:
    config = Config.load()
    print(
        "\n  Urdu <-> English translator\n"
        + ("  Devices are not configured yet - Settings will open.\n"
           if not config.setup_complete else "")
        + "  The bubble is in the corner of your screen; the tray icon has a menu.\n"
        "  Ctrl+C here, or Quit from the tray, closes it.\n"
    )
    try:
        return Application(config).run()
    except ProviderMisconfigured as exc:
        _report_startup_failure(f"Configuration error: {exc}", exc)
        return 3
    except Exception as exc:
        # Anything that escapes startup still has to be visible to someone who
        # launched the application by double-clicking it.
        _report_startup_failure(f"The translator could not start: {exc}", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
