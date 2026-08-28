"""The floating overlay: bubble and panel.

Spec reference: §2.3, §11.1, §11.2, §11.3.

During a meeting the consultant is looking at Teams, not at us. A conventional
window would compete for screen space and for the taskbar, and would have to
be hunted for when something needs to change quickly. So: a small always-on-top
bubble, collapsed to a single control most of the time, whose ring colour is
the entire status display.

    grey    idle, not translating (audio still passing through)
    green   translating, both directions healthy
    amber   paused, or running in one direction only
    red     failed - needs attention

Three rules govern this file, and all three are load-bearing:

  1. It must never steal focus (§11.2 / capture_guard.no_activate).
  2. It must be invisible to screen sharing (§2.4 / capture_guard).
  3. Every control is also a global hotkey - a target the size of a coin is
     not something to aim at mid-sentence in front of a client (§11.4).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.models.state import Colour, Mode
from app.views import capture_guard

RING = {
    Colour.GREY: QColor("#8A9099"),
    Colour.GREEN: QColor("#2FA84F"),
    Colour.AMBER: QColor("#E0A526"),
    Colour.RED: QColor("#D0453B"),
}

BUBBLE_SIZE = 56
PANEL_W = 300


class OverlayWindow(QWidget):
    """Base for the floating windows: on top, uncaptured, optionally frameless.

    Two of the three properties are shared by everything; the third is not, and
    conflating them was a mistake worth spelling out.

    The BUBBLE and PANEL must be frameless and must never take focus - they sit
    on top of Teams and a stolen focus loses the user their place. The cost is
    that Windows gives them no title bar, so they have to implement their own
    dragging, and they can never receive a keystroke.

    A utility window like the transcript wants the opposite: a real frame, with
    the close button, drag, resize and keyboard focus that Windows provides for
    free. Making it frameless bought nothing and broke all four. Subclasses set
    `frameless` and `unfocusable` to say which they are.
    """

    #: Frameless windows must supply their own dragging (below) and get no
    #: title bar, no close button and no resize handles from the OS.
    frameless = True
    #: Never take focus from Teams. Also means no keyboard events, ever.
    unfocusable = True

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        flags = Qt.WindowStaysOnTopHint | Qt.Tool  # no taskbar entry
        if self.frameless:
            flags |= Qt.FramelessWindowHint
        if self.unfocusable:
            flags |= Qt.WindowDoesNotAcceptFocus
        self.setWindowFlags(flags)
        if self.frameless:
            self.setAttribute(Qt.WA_TranslucentBackground)
        if self.unfocusable:
            self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._hardened: dict = {}
        self._drag_from: QPoint | None = None
        self._dragged = False

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().showEvent(event)
        if not self._hardened:
            # Must run after the native handle exists. Capture exclusion always;
            # the no-activate style only where focus really must not move.
            self._hardened = capture_guard.harden(
                self, unfocusable=self.unfocusable
            )

    @property
    def capture_excluded(self) -> bool:
        return bool(self._hardened.get("excluded"))

    # ------------------------------------------------------------------
    # dragging
    # ------------------------------------------------------------------
    # These windows are frameless, so there is no title bar to grab. Without
    # this they are nailed to wherever they first appeared - which is exactly
    # wrong for something that floats over someone else's application and will
    # inevitably end up covering the thing they need to see.

    draggable = True
    moved = Signal(QPoint)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.draggable and self.frameless and event.button() == Qt.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.pos()
            self._dragged = False
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is not None and event.buttons() & Qt.LeftButton:
            new = event.globalPosition().toPoint() - self._drag_from
            if (new - self.pos()).manhattanLength() > 3:
                self._dragged = True
            self.move(new)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._drag_from is not None:
            self._drag_from = None
            if self._dragged:
                self.moved.emit(self.pos())
                self._dragged = False
            else:
                self.on_click()
            event.accept()

    def on_click(self) -> None:
        """A press-and-release that did not move. Overridden where it matters."""


class Bubble(OverlayWindow):
    """The collapsed control. One circle, one colour, no chrome."""

    clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(BUBBLE_SIZE + 16, BUBBLE_SIZE + 16)
        self._colour = Colour.GREY
        self._label = "UR"
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setToolTip("Urdu ↔ English translator")

    # ------------------------------------------------------------------

    def set_state(self, colour: Colour, label: str = "UR") -> None:
        if (colour, label) != (self._colour, self._label):
            self._colour = colour
            self._label = label
            self.update()

    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRect(8, 8, BUBBLE_SIZE, BUBBLE_SIZE)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(28, 30, 34, 235))
        p.drawEllipse(r)

        pen = QPen(RING[self._colour], 3.5)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(r.adjusted(2, 2, -2, -2))

        p.setPen(QColor("#F2F4F7"))
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        p.setFont(f)
        p.drawText(r, Qt.AlignCenter, self._label)

    def on_click(self) -> None:
        self.clicked.emit()

    # ------------------------------------------------------------------

    def clamp_to_screen(self) -> None:
        """Rescue a bubble stranded off-screen after undocking (§11.2)."""
        screens = QApplication.screens()
        if not screens:
            return
        rect = self.frameGeometry()
        if any(s.availableGeometry().intersects(rect) for s in screens):
            return
        area = QApplication.primaryScreen().availableGeometry()
        self.move(area.right() - self.width() - 40, area.bottom() - self.height() - 80)


class Panel(OverlayWindow):
    """The expanded controls. Opens on click, closes on click-away or idle."""

    start_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    transcript_clicked = Signal()
    settings_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(PANEL_W)
        self._build()
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(6000)
        self._idle.timeout.connect(self.hide)
        # Once the user has placed the panel themselves, stop yanking it back
        # to the bubble every time it reopens.
        self._user_positioned = False
        self.moved.connect(lambda _: setattr(self, "_user_positioned", True))

    # ------------------------------------------------------------------

    def _build(self) -> None:
        card = QFrame(self)
        card.setObjectName("card")
        card.setStyleSheet(
            """
            #card { background: rgba(24,26,30,242); border-radius: 12px;
                    border: 1px solid rgba(255,255,255,26); }
            QLabel { color: #E8EAED; font-size: 12px; }
            QLabel#dim { color: #9AA0A6; font-size: 11px; }
            QLabel#head { color: #F2F4F7; font-size: 13px; font-weight: 600; }
            QLabel#quote { color: #C7CBD1; font-size: 11px; }
            QPushButton { background: rgba(255,255,255,20); color: #E8EAED;
                          border: none; border-radius: 6px; padding: 7px 10px;
                          font-size: 12px; }
            QPushButton:hover { background: rgba(255,255,255,36); }
            QPushButton#primary { background: #2FA84F; color: white;
                                  font-weight: 600; }
            QPushButton#primary:hover { background: #35B958; }
            QPushButton#link { background: transparent; color: #9AA0A6;
                               font-size: 11px; padding: 3px; }
            """
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(card)

        v = QVBoxLayout(card)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(9)

        top = QHBoxLayout()
        self.title = QLabel("Urdu ↔ English")
        self.title.setObjectName("head")
        self.status = QLabel("PASSTHROUGH")
        self.status.setObjectName("dim")
        self.status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(self.title)
        top.addWidget(self.status)
        v.addLayout(top)

        self.lat_in = QLabel("incoming  EN → UR      —")
        self.lat_out = QLabel("outgoing  UR → EN      —")
        for w in (self.lat_in, self.lat_out):
            w.setObjectName("dim")
            v.addWidget(w)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_primary = QPushButton("Start")
        self.btn_primary.setObjectName("primary")
        self.btn_stop = QPushButton("Stop")
        row.addWidget(self.btn_primary, 2)
        row.addWidget(self.btn_stop, 1)
        v.addLayout(row)

        self.devices = QLabel("mic —   phones —   teams —")
        self.devices.setObjectName("dim")
        v.addWidget(self.devices)

        self.quote = QLabel("")
        self.quote.setObjectName("quote")
        self.quote.setWordWrap(True)
        self.quote.setMinimumHeight(30)
        v.addWidget(self.quote)

        links = QHBoxLayout()
        self.btn_transcript = QPushButton("transcript")
        self.btn_transcript.setObjectName("link")
        self.btn_settings = QPushButton("settings")
        self.btn_settings.setObjectName("link")
        links.addWidget(self.btn_transcript)
        links.addStretch(1)
        links.addWidget(self.btn_settings)
        v.addLayout(links)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 170))
        card.setGraphicsEffect(shadow)

        self.btn_primary.clicked.connect(self._primary)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_transcript.clicked.connect(self.transcript_clicked.emit)
        self.btn_settings.clicked.connect(self.settings_clicked.emit)
        self._mode = Mode.PASSTHROUGH

    def _primary(self) -> None:
        if self._mode is Mode.TRANSLATING:
            self.pause_clicked.emit()
        else:
            self.start_clicked.emit()

    # ------------------------------------------------------------------

    def apply_state(self, state: dict) -> None:
        mode = Mode(state["mode"])
        self._mode = mode
        # `status` lets a caller show something truer than the raw mode - e.g.
        # "NOT CONFIGURED" while the engine has never started.
        self.status.setText(state.get("status") or mode.value.upper())

        if mode in (Mode.TRANSLATING, Mode.DEGRADED):
            # DEGRADED is still a live session - one direction is translating.
            # Offering "Start" there would imply nothing is running.
            self.btn_primary.setText("Pause")
            self.btn_primary.setObjectName("")
        elif mode is Mode.PAUSED:
            self.btn_primary.setText("Resume")
            self.btn_primary.setObjectName("primary")
        else:
            self.btn_primary.setText("Start")
            self.btn_primary.setObjectName("primary")
        self.btn_primary.style().unpolish(self.btn_primary)
        self.btn_primary.style().polish(self.btn_primary)

        if state.get("reason"):
            self.quote.setText(state["reason"])
        self._refit()

    def apply_metrics(self, metrics: dict) -> None:
        def fmt(d):
            p50 = d["ttfta"]["p50"]
            return "—" if p50 is None else f"{p50:.2f} s"

        self.lat_in.setText(f"incoming  EN → UR      {fmt(metrics['incoming'])}")
        self.lat_out.setText(f"outgoing  UR → EN      {fmt(metrics['outgoing'])}")
        self._refit()

    def set_devices(self, mic: bool, phones: bool, teams: bool) -> None:
        tick = lambda ok: "OK" if ok else "—"  # noqa: E731
        self.devices.setText(
            f"mic {tick(mic)}   phones {tick(phones)}   teams {tick(teams)}"
        )

    def set_quote(self, text: str) -> None:
        self.quote.setText(f"“{text}”" if text else "")
        self._refit()

    def _refit(self) -> None:
        """Resize to the content while the panel is open, growing upwards.

        Content changes height: an error reason is word-wrapped and can be
        four lines where the last was none. `adjustSize` otherwise only runs
        when the panel opens, so a panel that is ALREADY open keeps its old
        geometry and the card is drawn outside it. On a frameless translucent
        window nothing repaints the abandoned pixels, so the previous layout
        stays on screen underneath the new one - two rows of buttons, one of
        them a ghost.

        It grows upwards because the panel is anchored above the bubble:
        keeping the top fixed would slide the controls down over it.
        """
        if not self.isVisible():
            return
        before = self.geometry()
        self.adjustSize()
        left, bottom = before.left(), before.bottom()
        top = bottom - self.height() + 1

        screen = QApplication.screenAt(before.center()) or QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            top = max(area.top(), min(top, area.bottom() - self.height()))
        self.move(left, top)
        self.update()

    # ------------------------------------------------------------------

    def popup_near(self, bubble: Bubble) -> None:
        """Anchor above-left of the bubble, kept inside the screen."""
        self.adjustSize()
        if self._user_positioned:
            self.show()
            self.raise_()
            self._idle.start()
            return
        area = QApplication.screenAt(bubble.pos()) or QApplication.primaryScreen()
        g = area.availableGeometry()
        x = bubble.x() + bubble.width() - self.width()
        y = bubble.y() - self.height() - 8
        if y < g.top():
            y = bubble.y() + bubble.height() + 8
        x = max(g.left() + 8, min(x, g.right() - self.width() - 8))
        self.move(x, y)
        self.show()
        self.raise_()
        self._idle.start()

    def enterEvent(self, event) -> None:  # noqa: N802
        self._idle.stop()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._idle.start()
