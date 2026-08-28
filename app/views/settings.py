"""Settings — device binding, voices, credentials, tuning.

Spec reference: §11.7, §12.2, §9.3.

The user should never edit a JSON file. Every endpoint is a dropdown populated
from the live device list, every credential is written to Windows Credential
Manager rather than to disk, and the three endpointing knobs that dominate
latency are exposed where they can be changed and measured.

The device tab is deliberately the first one: until it is filled in, nothing
else in the application can run.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.audio.devices import AudioUnavailable, DeviceManager
from app.models.config import SECRET_NAMES, Config, get_secret, set_secret
from app.providers.tts import AZURE_URDU_VOICES
from app.views import capture_guard

STYLE = """
QDialog, QWidget { background: #16181C; color: #E8EAED; }
QLabel { color: #C7CBD1; font-size: 12px; }
QLabel#hint { color: #8A9099; font-size: 11px; }
QLabel#bad  { color: #D0453B; font-size: 11px; }
QLabel#good { color: #2FA84F; font-size: 11px; }
QGroupBox { border: 1px solid rgba(255,255,255,24); border-radius: 8px;
            margin-top: 14px; padding-top: 10px; font-size: 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #9AA0A6; }
QComboBox, QLineEdit, QSpinBox {
    background: #0E1013; border: 1px solid rgba(255,255,255,26);
    border-radius: 6px; padding: 6px 8px; color: #E8EAED; font-size: 12px; }
QPushButton { background: rgba(255,255,255,20); border: none; border-radius: 6px;
              padding: 7px 14px; font-size: 12px; }
QPushButton:hover { background: rgba(255,255,255,36); }
QPushButton#primary { background: #2FA84F; color: white; font-weight: 600; }
QTabBar::tab { background: transparent; padding: 8px 14px; color: #9AA0A6; }
QTabBar::tab:selected { color: #E8EAED; border-bottom: 2px solid #2FA84F; }
QTabWidget::pane { border: none; }
"""


class SettingsDialog(QDialog):
    """Returns Accepted when the user saved. Read `config` afterwards."""

    applied = Signal()

    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Translator settings")
        self.setMinimumWidth(600)
        # No minimum height. A dialog taller than the screen puts Save below
        # the bottom edge, where it cannot be clicked - the one control the
        # window exists for. The tabs scroll instead; see `_scrollable`.
        self.setStyleSheet(STYLE)

        self._render: list[str] = []
        self._capture: list[str] = []
        self._device_error = ""
        self._load_devices()

        tabs = QTabWidget(self)
        tabs.addTab(self._scrollable(self._devices_tab()), "Devices")
        tabs.addTab(self._scrollable(self._voices_tab()), "Language and voice")
        tabs.addTab(self._scrollable(self._keys_tab()), "API keys")
        tabs.addTab(self._scrollable(self._hotkeys_tab()), "Hotkeys")
        tabs.addTab(self._scrollable(self._tuning_tab()), "Tuning")

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel, self
        )
        buttons.button(QDialogButtonBox.Save).setObjectName("primary")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 14)
        v.addWidget(tabs)
        v.addWidget(buttons)  # outside the scroll area: always reachable

    @staticmethod
    def _scrollable(content: QWidget) -> QScrollArea:
        """Put a tab's contents in a scroll area.

        The Devices tab is the tallest and grew again with the source picker.
        A tab that is simply as tall as its contents makes the dialog as tall
        as its contents, which on a 768-pixel-high laptop screen pushes Save
        off the bottom of the display. Scrolling the contents keeps the window
        inside the screen and keeps the buttons where they can be clicked.
        """
        area = QScrollArea()
        area.setWidget(content)
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area

    def _fit_to_screen(self) -> None:
        """Open at the content's size, but never taller than the display."""
        screen = QApplication.screenAt(self.pos()) or QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        # Leave room for the title bar and a margin; a window sized to the
        # exact available area still ends up with its edge under the taskbar
        # on some setups.
        height = min(self.sizeHint().height(), int(area.height() * 0.88))
        width = min(max(self.width(), self.sizeHint().width()), area.width() - 40)
        self.resize(width, height)
        self.move(
            area.left() + (area.width() - width) // 2,
            area.top() + max(0, (area.height() - height) // 2),
        )

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._fit_to_screen()
        # §2.4 - settings is a window like any other and must not appear in a
        # screen share; it shows device names and provider choices.
        capture_guard.harden(self, unfocusable=False)

    # ------------------------------------------------------------------

    def _load_devices(self) -> None:
        try:
            with DeviceManager() as dm:
                for d in dm.wasapi_devices():
                    if d.can_render:
                        self._render.append(d.name)
                    if d.can_capture:
                        self._capture.append(d.name)
        except AudioUnavailable as exc:
            self._device_error = str(exc)

    def _combo(self, options: list[str], current: str,
               empty: str = "— not selected —") -> QComboBox:
        box = QComboBox()
        box.addItem(empty, "")
        for name in options:
            box.addItem(name, name)
        if current:
            idx = box.findData(current)
            if idx < 0:  # remembered device is not present right now
                box.addItem(f"{current}  (not found)", current)
                idx = box.count() - 1
            box.setCurrentIndex(idx)
        return box

    # ------------------------------------------------------------------

    def _devices_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        d = self.config.devices

        if self._device_error:
            bad = QLabel(self._device_error)
            bad.setObjectName("bad")
            bad.setWordWrap(True)
            v.addWidget(bad)

        what = QGroupBox("What are you translating?")
        f0 = QFormLayout(what)
        self.cb_mode = QComboBox()
        self.cb_mode.addItem(
            "Something playing on this computer  (a video, a podcast)", "video"
        )
        self.cb_mode.addItem("A Teams call  (both directions)", "meeting")
        index = self.cb_mode.findData(self.config.mode or "video")
        self.cb_mode.setCurrentIndex(max(0, index))
        self.cb_mode.currentIndexChanged.connect(self._apply_mode)
        f0.addRow("Source", self.cb_mode)
        self.lb_mode_hint = QLabel("")
        self.lb_mode_hint.setObjectName("hint")
        self.lb_mode_hint.setWordWrap(True)
        f0.addRow("", self.lb_mode_hint)
        v.addWidget(what)

        yours = QGroupBox("Your own hardware")
        f1 = QFormLayout(yours)
        self.cb_mic = self._combo(self._capture, d.microphone, "— system default —")
        self.cb_phones = self._combo(self._render, d.headphones, "— system default —")
        self.lb_mic = QLabel("Microphone")
        f1.addRow(self.lb_mic, self.cb_mic)
        f1.addRow("Headphones", self.cb_phones)
        hint = QLabel(
            "Leave either on system default and Windows decides. Headphones "
            "are strongly recommended: on speakers the microphone picks up the "
            "translated audio and the system transcribes its own output."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        f1.addRow("", hint)
        v.addWidget(yours)

        virt = QGroupBox("Virtual audio cables")
        f2 = QFormLayout(virt)
        self.cb_vmic = self._combo(self._render, d.virtual_mic)
        self.cb_teams = self._combo(self._render, d.teams_speaker)
        self.lb_vmic = QLabel("Write English into")
        self.lb_teams = QLabel("Teams outputs to")
        f2.addRow(self.lb_vmic, self.cb_vmic)
        f2.addRow(self.lb_teams, self.cb_teams)
        self.lb_cables_note = QLabel("")
        self.lb_cables_note.setObjectName("hint")
        self.lb_cables_note.setWordWrap(True)
        f2.addRow("", self.lb_cables_note)
        v.addWidget(virt)

        row = QHBoxLayout()
        refresh = QPushButton("Refresh device list")
        refresh.clicked.connect(self._refresh)
        row.addWidget(refresh)
        row.addStretch(1)
        v.addLayout(row)
        v.addStretch(1)
        self._apply_mode()
        return w

    def _apply_mode(self) -> None:
        """Show only the fields the chosen topology actually uses.

        The two modes need different hardware, and leaving the unused rows
        active invites exactly the misconfiguration this dialog exists to
        prevent: a microphone chosen for a session that has no outgoing
        direction, or a second cable chosen where one endpoint is all that
        will ever be opened.
        """
        video = self.cb_mode.currentData() == "video"

        for widget in (self.cb_mic, self.lb_mic, self.cb_vmic, self.lb_vmic):
            widget.setEnabled(not video)

        self.lb_teams.setText("Listen to" if video else "Teams outputs to")

        if video:
            self.lb_mode_hint.setText(
                "One way: English in, Urdu out of your speakers. Point the "
                "application you are watching at the cable below - Windows "
                "Settings › System › Sound › Volume mixer › (the app) › Output "
                "device - and everything else on the machine is untouched. "
                "Your microphone is not opened at all."
            )
            self.lb_cables_note.setText(
                "Only this one is used. It has to be a virtual cable: we "
                "capture whatever it plays, so anything you can hear would be "
                "captured along with the Urdu we are speaking into it."
            )
        else:
            self.lb_mode_hint.setText(
                "Both directions. You speak Urdu and the client hears English; "
                "the client speaks English and you hear Urdu. Needs two cables "
                "and your microphone."
            )
            self.lb_cables_note.setText(
                "These two are required - there is no sensible default for "
                "them. Set the Teams microphone to the capture half of the "
                "first; set the Teams speaker to the second, which we capture "
                "so you never hear the original. Turn Teams noise suppression "
                "off, or it will gate the synthesised speech."
            )

    def _refresh(self) -> None:
        self._render.clear()
        self._capture.clear()
        self._load_devices()
        for box, options, current in (
            (self.cb_mic, self._capture, self.cb_mic.currentData()),
            (self.cb_phones, self._render, self.cb_phones.currentData()),
            (self.cb_vmic, self._render, self.cb_vmic.currentData()),
            (self.cb_teams, self._render, self.cb_teams.currentData()),
        ):
            box.clear()
            box.addItem("— not selected —", "")
            for name in options:
                box.addItem(name, name)
            idx = box.findData(current) if current else 0
            box.setCurrentIndex(max(0, idx))

    # ------------------------------------------------------------------

    def _voices_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        p = self.config.providers

        self.cb_voice = QComboBox()
        for name in AZURE_URDU_VOICES:
            self.cb_voice.addItem(name, name)
        idx = self.cb_voice.findData(p.urdu_voice)
        self.cb_voice.setCurrentIndex(max(0, idx))
        f.addRow("Urdu voice", self.cb_voice)

        self.cb_tts = QComboBox()
        for label, value in (
            ("Azure Neural TTS", "azure-neural-tts"),
            ("ElevenLabs Flash v2.5", "elevenlabs"),
            ("Silent (testing only)", "null-tts"),
        ):
            self.cb_tts.addItem(label, value)
        idx = self.cb_tts.findData(p.tts)
        self.cb_tts.setCurrentIndex(max(0, idx))
        f.addRow("Voice engine", self.cb_tts)

        self.ed_model = QLineEdit(p.translator_model)
        f.addRow("Translation model", self.ed_model)

        self.ed_region = QLineEdit(p.region)
        self.ed_region.setPlaceholderText("e.g. southeastasia")
        f.addRow("Provider region", self.ed_region)

        hint = QLabel(
            "Urdu → English uses one speech-to-speech model. English → Urdu "
            "cannot: Urdu is not an available output language, so that "
            "direction runs recognition → translation → voice."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        f.addRow("", hint)
        return w

    # ------------------------------------------------------------------

    def _keys_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.key_fields: dict[str, QLineEdit] = {}

        for name in SECRET_NAMES:
            row = QHBoxLayout()
            field = QLineEdit()
            if name.endswith("REGION"):
                field.setPlaceholderText("e.g. southeastasia")
                field.setText(get_secret(name))
            else:
                field.setEchoMode(QLineEdit.Password)
                field.setPlaceholderText(
                    "•••• stored" if get_secret(name) else "not set"
                )
            self.key_fields[name] = field
            row.addWidget(field)
            state = QLabel("stored" if get_secret(name) else "missing")
            state.setObjectName("good" if get_secret(name) else "bad")
            row.addWidget(state)
            holder = QWidget()
            holder.setLayout(row)
            f.addRow(name.replace("_", " ").title(), holder)

        v.addLayout(f)
        hint = QLabel(
            "Keys are written to Windows Credential Manager, never to a file "
            "in this folder. Leave a field blank to keep the stored value."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------------

    def _hotkeys_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        u = self.config.ui

        self.hk_fields: dict[str, QLineEdit] = {}
        for attr, label in (
            ("hotkey_pause", "Pause / Resume"),
            ("hotkey_start_stop", "Start / Stop"),
            ("hotkey_hide", "Hide the bubble"),
            ("hotkey_repeat", "Repeat last utterance"),
        ):
            field = QLineEdit(getattr(u, attr))
            field.setPlaceholderText("e.g. Ctrl+Shift+Space")
            self.hk_fields[attr] = field
            f.addRow(label, field)

        hint = QLabel(
            "These work while Teams has focus, which is the whole point - but "
            "it also means another application can already own one. Ctrl+Alt+"
            "Space in particular is claimed by Teams' own push-to-talk and by "
            "most input-method editors.\n\n"
            "If a combination is taken the application tries alternatives and "
            "tells you which one it settled on. Set one explicitly here to "
            "stop it moving."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        f.addRow("", hint)
        return w

    def _tuning_tab(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        p = self.config.pipeline

        self.sl_hangover = QSlider(Qt.Horizontal)
        self.sl_hangover.setRange(150, 900)
        self.sl_hangover.setValue(int(p.vad_hangover_ms))
        self.lb_hangover = QLabel(f"{int(p.vad_hangover_ms)} ms")
        self.sl_hangover.valueChanged.connect(
            lambda v: self.lb_hangover.setText(f"{v} ms")
        )
        row = QHBoxLayout()
        row.addWidget(self.sl_hangover)
        row.addWidget(self.lb_hangover)
        holder = QWidget()
        holder.setLayout(row)
        f.addRow("Silence before end of speech", holder)

        self.sp_stability = QSpinBox()
        self.sp_stability.setRange(50, 1200)
        self.sp_stability.setSingleStep(50)
        self.sp_stability.setSuffix(" ms")
        self.sp_stability.setValue(int(p.stability_ms))
        f.addRow("Text must hold steady for", self.sp_stability)

        self.sp_words = QSpinBox()
        self.sp_words.setRange(4, 40)
        self.sp_words.setValue(p.max_words)
        f.addRow("Translate after at most", self.sp_words)

        hint = QLabel(
            "These three dominate the delay before you hear anything — far "
            "more than the network or the model. Lower is faster but risks "
            "translating half a sentence. Tune them against a recording with "
            "tools/replay.py --sweep, not in a live meeting."
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        f.addRow("", hint)
        return w

    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.config.mode = self.cb_mode.currentData() or "video"
        d = self.config.devices
        d.microphone = self.cb_mic.currentData() or ""
        d.headphones = self.cb_phones.currentData() or ""
        d.virtual_mic = self.cb_vmic.currentData() or ""
        d.teams_speaker = self.cb_teams.currentData() or ""

        p = self.config.providers
        p.urdu_voice = self.cb_voice.currentData()
        p.tts = self.cb_tts.currentData()
        p.translator_model = self.ed_model.text().strip() or p.translator_model
        p.region = self.ed_region.text().strip()

        pl = self.config.pipeline
        pl.vad_hangover_ms = float(self.sl_hangover.value())
        pl.stability_ms = float(self.sp_stability.value())
        pl.max_words = self.sp_words.value()

        for attr, field in self.hk_fields.items():
            text = field.text().strip()
            if text:
                setattr(self.config.ui, attr, text)

        for name, field in self.key_fields.items():
            value = field.text().strip()
            if value:
                set_secret(name, value)

        # Only the cables are required; microphone and speakers fall back to
        # whatever Windows is already using. Video mode uses one cable, so
        # demanding two would leave it permanently "not configured".
        if self.config.mode == "video":
            self.config.setup_complete = bool(d.teams_speaker)
        else:
            self.config.setup_complete = bool(d.virtual_mic and d.teams_speaker)
        self.config.save()
        self.applied.emit()
        self.accept()
