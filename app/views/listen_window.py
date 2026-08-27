"""The listen-mode window.

Spec reference: §2.4 (never in a screen share), §11.4 (targets big enough to
hit mid-sentence).

WHY THIS IS NOT THE BUBBLE AND PANEL
------------------------------------
The bubble exists because meeting mode has to be operated while a client is
watching your screen, so its controls have to be tiny, frameless and unable to
steal focus. Listening to a video has none of those constraints: nobody is
waiting on you, and a window with a title bar you can drag and a button you
can actually hit is simply better. What is worth keeping is the capture
exclusion - a Teams call is one of the two things this tool is for, and an
always-on-top panel with a live transcript of the other person's words is not
something to share back to them by accident.

Subtitles are the existing `TranscriptWindow`, unchanged: it already lays Urdu
out right-to-left, marks dropped units instead of hiding them, and is excluded
from capture.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from app.views.overlay import OverlayWindow

#: status -> (dot colour, headline)
LOOK = {
    "idle": ("#8A9099", "Not translating"),
    "listening": ("#2FA84F", "Listening — Urdu only"),
    "paused": ("#E0A526", "Paused"),
    "degraded": ("#D0453B", "Translation stopped"),
}

STYLE = """
QWidget { background: #16181C; }
QLabel { color: #E8EAED; font-size: 12px; }
QLabel#detail { color: #9AA0A6; font-size: 11px; }
QLabel#devices { color: #6C7481; font-size: 11px; }
QLabel#hint { color: #6C7481; font-size: 11px; }
QLabel#headline { font-size: 15px; font-weight: 600; }
QPushButton {
    background: #23262C; color: #E8EAED; border: 1px solid rgba(255,255,255,25);
    border-radius: 8px; padding: 9px 14px; font-size: 12px;
}
QPushButton:hover { background: #2C3038; }
QPushButton:disabled { color: #5A6068; border-color: rgba(255,255,255,12); }
QPushButton#primary {
    background: #2FA84F; border: none; font-weight: 600; padding: 11px 14px;
}
QPushButton#primary:hover { background: #35BC58; }
QPushButton#primary[stop="true"] { background: #8A3F39; }
QPushButton#primary[stop="true"]:hover { background: #9C4740; }
/* A disabled primary must not keep the go-colour. A green Start that does
   nothing when clicked reads as a broken button rather than as a blocked
   one, and the blocked reason is right above it. */
QPushButton#primary:disabled { background: #2A2D33; color: #5A6068; }
"""


class ListenWindow(OverlayWindow):
    """Start, stop, and the one line of state that matters."""

    frameless = False   # a title bar to drag and a close button, for free
    unfocusable = False  # and Escape reaches us

    start_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    subtitles_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Listen in Urdu")
        self.setStyleSheet(STYLE)
        self.setMinimumWidth(340)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(9)
        self.dot = QLabel()
        self.dot.setFixedSize(11, 11)
        head.addWidget(self.dot, 0, Qt.AlignVCenter)
        self.headline = QLabel("Not translating")
        self.headline.setObjectName("headline")
        head.addWidget(self.headline, 1)
        v.addLayout(head)

        self.detail = QLabel("")
        self.detail.setObjectName("detail")
        self.detail.setWordWrap(True)
        v.addWidget(self.detail)

        self.primary = QPushButton("Start")
        self.primary.setObjectName("primary")
        self.primary.clicked.connect(self._primary)
        v.addWidget(self.primary)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_clicked)
        self.btn_pause.setEnabled(False)
        row.addWidget(self.btn_pause)
        self.btn_subs = QPushButton("Subtitles")
        self.btn_subs.clicked.connect(self.subtitles_clicked)
        row.addWidget(self.btn_subs)
        v.addLayout(row)

        self.devices = QLabel("")
        self.devices.setObjectName("devices")
        self.devices.setWordWrap(True)
        v.addWidget(self.devices)

        self.hint = QLabel(
            "Route the app you are watching to the cable: Windows Settings › "
            "System › Sound › Volume mixer › (the app) › Output device."
        )
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        v.addWidget(self.hint)

        self._status = "idle"
        self.set_status("idle", "passthrough — you are hearing the original")

    # ------------------------------------------------------------------

    def _primary(self) -> None:
        # One button, because there is only ever one sensible next action.
        if self._status == "listening":
            self.stop_clicked.emit()
        else:
            self.start_clicked.emit()

    def set_status(self, status: str, detail: str = "") -> None:
        colour, headline = LOOK.get(status, LOOK["idle"])
        self._status = status
        self.dot.setStyleSheet(
            f"background: {colour}; border-radius: 5px; min-width: 11px;"
        )
        self.headline.setText(headline)
        self.detail.setText(detail)

        listening = status == "listening"
        self.primary.setText("Stop" if listening else "Start")
        self.primary.setProperty("stop", "true" if listening else "false")
        # A dynamic property only repaints if the stylesheet is re-applied.
        self.primary.style().unpolish(self.primary)
        self.primary.style().polish(self.primary)
        self.btn_pause.setEnabled(listening or status == "paused")
        self.btn_pause.setText("Resume" if status == "paused" else "Pause")

    def set_devices(self, text: str) -> None:
        self.devices.setText(text)

    def set_blocked(self, problems: list[str]) -> None:
        """No audio. Say what to fix and refuse to pretend Start would work."""
        self.set_status("degraded", "\n".join(problems))
        self.headline.setText("Audio not configured")
        self.primary.setEnabled(False)
        self.btn_pause.setEnabled(False)

    # No Escape-to-hide here, deliberately. The transcript can afford it - it
    # has the control window to be reopened from. This window has nothing
    # above it and no tray icon, so hiding it would leave the process running
    # and unreachable. Escape belongs to the subtitle window only.
