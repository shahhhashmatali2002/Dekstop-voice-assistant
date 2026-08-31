"""The transcript window.

Spec reference: §7.5, §7.7, §13.3, §14.2.

Two jobs beyond simply showing text:

  1. SPEED. Even at a perfect 800 ms, the consultant is always responding to
     something the client finished saying nearly a second ago. Reading is
     faster than listening, so the transcript updates ahead of the Urdu audio
     and the eyes get ahead of the ears (§7.5).

  2. HONESTY. Every drop, gap and degradation gets a visible marker. A
     translated sentence that never arrives, with no indication, is worse than
     an error message - the consultant must always be able to tell the
     difference between "the client said nothing" and "we lost that sentence"
     (§14.2).

Numbers and dates are highlighted, because the highest-cost errors in a
business meeting are a wrong figure rather than awkward phrasing (§7.7).

DEFAULT CLOSED (§13.3): this panel shows the client's words in plain text on
an always-on-top window. It is excluded from capture like everything else, but
shoulder-surfing is not.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QTextBlockFormat, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtWidgets import QApplication

from app.views.overlay import OverlayWindow

#: Arabic-script ranges. Urdu uses the Arabic script, so any of these means
#: the paragraph should be laid out right-to-left.
RTL_RANGES = (
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
)


def is_rtl(text: str) -> bool:
    """True when the line is predominantly Arabic-script.

    Counts letters only: a sentence of Urdu carrying two English nouns is
    still an Urdu sentence and must be laid out right-to-left.
    """
    rtl = ltr = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in RTL_RANGES):
            rtl += 1
        else:
            ltr += 1
    return rtl > ltr


NUMBER_RE = re.compile(
    r"\b("
    r"\d[\d,.]*\s?(?:%|k|m|bn|million|billion)?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}"
    r"|mon|tue|wed|thu|fri|sat|sun"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r")\b",
    re.IGNORECASE,
)


class TranscriptWindow(OverlayWindow):
    """Live transcript.

    A NORMAL WINDOW, DELIBERATELY
    -----------------------------
    Unlike the bubble and the panel, this one keeps its native frame. That
    single decision hands over, for free and correctly on every Windows
    version, the four things a frameless version had to reimplement and got
    wrong: a title bar to drag by, a close button, resize handles, and
    keyboard focus so Escape can work.

    It is still always-on-top and still excluded from screen capture (§2.4) -
    those are the properties that actually matter here. Not taking focus is
    not one of them: reading the transcript is a deliberate act, and the user
    may well want to select and copy from it.
    """

    frameless = False   # give me a title bar, a close button and resize grips
    unfocusable = False  # and let me receive Escape

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Live transcript")
        self.resize(480, 360)
        self.setMinimumSize(320, 200)
        self.setStyleSheet(
            """
            QWidget { background: #16181C; }
            QLabel { color: #9AA0A6; font-size: 11px; }
            QCheckBox { color: #9AA0A6; font-size: 11px; }
            QTextEdit { background: #101215; color: #E8EAED;
                        border: 1px solid rgba(255,255,255,20);
                        border-radius: 8px; font-size: 12px; padding: 8px; }
            """
        )

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)
        v.setSpacing(8)

        head = QHBoxLayout()
        head.addWidget(QLabel("Press Esc to close"))
        head.addStretch(1)
        self.show_source = QCheckBox("show original")
        self.show_source.setChecked(True)
        head.addWidget(self.show_source)
        v.addLayout(head)

        self.view = QTextEdit(self)
        self.view.setReadOnly(True)
        v.addWidget(self.view)

        self.note = QLabel("Excluded from screen capture · not recorded to disk")
        v.addWidget(self.note)

        self._last_direction = ""
        self._placed = False

    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.hide()
            return
        super().keyPressEvent(event)

    def place_near(self, other=None) -> None:
        """Put the window somewhere deliberate the first time it is shown.

        Qt centres an unpositioned window, which for an always-on-top panel
        means parking it squarely over whatever the user is reading.
        """
        if self._placed:
            return
        self._placed = True
        screen = QApplication.screenAt(self.pos()) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        self.move(area.left() + 48, area.top() + 72)

    def showEvent(self, event) -> None:  # noqa: N802
        self.place_near()
        super().showEvent(event)
        self.activateWindow()  # so Escape reaches us

    # ------------------------------------------------------------------

    def _fmt(self, colour: str, *, bold=False, italic=False) -> QTextCharFormat:
        f = QTextCharFormat()
        f.setForeground(QColor(colour))
        f.setFontWeight(700 if bold else 400)
        f.setFontItalic(italic)
        return f

    def _append(self, text: str, fmt: QTextCharFormat) -> None:
        cur = self.view.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertText(text, fmt)
        self.view.setTextCursor(cur)
        self.view.ensureCursorVisible()

    def _new_block(self, rtl: bool, indent: int = 0) -> None:
        """Start a paragraph with the correct base direction.

        This is not cosmetic. A line of Urdu carrying English technical nouns -
        which is every line this product handles - is laid out by the Unicode
        bidi algorithm relative to the PARAGRAPH's base direction. Leave the
        paragraph left-to-right and the Urdu runs get reordered around the
        English ones, so the sentence reads scrambled to the person it is for.
        """
        cur = self.view.textCursor()
        cur.movePosition(QTextCursor.End)
        cur.insertBlock()
        bf = QTextBlockFormat()
        bf.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)
        bf.setAlignment(Qt.AlignRight if rtl else Qt.AlignLeft)
        bf.setLeftMargin(indent)
        bf.setRightMargin(indent)
        bf.setTopMargin(1)
        cur.setBlockFormat(bf)
        self.view.setTextCursor(cur)

    def _append_highlighted(self, text: str, base: QTextCharFormat) -> None:
        """Insert text with numbers and dates picked out (§7.7)."""
        strong = QTextCharFormat(base)
        strong.setForeground(QColor("#F5C542"))
        strong.setFontWeight(700)

        pos = 0
        for m in NUMBER_RE.finditer(text):
            if m.start() > pos:
                self._append(text[pos : m.start()], base)
            self._append(m.group(0), strong)
            pos = m.end()
        if pos < len(text):
            self._append(text[pos:], base)

    # ------------------------------------------------------------------

    def add_entry(self, payload: dict) -> None:
        direction = payload.get("direction", "")
        source = (payload.get("source") or "").strip()
        translated = (payload.get("translated") or "").strip()
        dropped = payload.get("dropped", False)

        who = "client" if direction == "incoming" else "you"
        self._new_block(False)
        self._append(f"[{who}]", self._fmt("#6C7481", bold=True))

        if dropped:
            # Never fail silently (§14.2).
            if source:
                self._new_block(is_rtl(source), indent=14)
                self._append_highlighted(source, self._fmt("#6C7481", italic=True))
            self._new_block(False, indent=14)
            self._append(
                "\u26a0 dropped \u2014 this sentence was not translated",
                self._fmt("#D0453B", italic=True),
            )
            self._last_direction = direction
            return

        # Labelled, not merely greyed. Two lines of different languages with
        # no labels leave the reader guessing which one the app produced -
        # and when the recogniser mishears, an unlabelled original reads as a
        # translation into the wrong language.
        if source and self.show_source.isChecked():
            self._new_block(False, indent=14)
            self._append("Original: ", self._fmt("#6C7481", bold=True))
            self._new_block(is_rtl(source), indent=28)
            self._append_highlighted(source, self._fmt("#9AA0A6"))
        if translated:
            if source and self.show_source.isChecked():
                self._new_block(False, indent=14)
                self._append("Translation: ", self._fmt("#6C7481", bold=True))
            self._new_block(is_rtl(translated), indent=28 if source else 14)
            self._append_highlighted(translated, self._fmt("#E8EAED"))
        self._last_direction = direction

    def add_notice(self, severity: str, message: str) -> None:
        colour = {"error": "#D0453B", "warning": "#E0A526"}.get(severity, "#9AA0A6")
        self._new_block(False)
        self._append(f"\u2014 {message}", self._fmt(colour, italic=True))

    def clear(self) -> None:
        self.view.clear()
        self._last_direction = ""
