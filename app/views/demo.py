"""UI preview - the overlay with no audio, no providers, no Windows.

    python tools/ui_preview.py

Runs the real Bubble, Panel and TranscriptWindow against a scripted fake
engine, so the interface can be seen, clicked and iterated on before any of
the audio plumbing exists. Every widget here is the production widget; only
the data behind it is invented.

Use it to answer the questions a static mockup cannot: does the bubble read at
a glance, is the panel reachable without hunting, does the transcript stay
legible while it scrolls.

Two properties it CANNOT show you, both of which need a real Windows session:
  * that the overlay never steals focus from Teams (§11.2)
  * that it is absent from a screen share (§2.4)
Neither is visible on a canvas. Test them in a real call.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.models.state import Colour, Mode
from app.views import capture_guard
from app.views.overlay import Bubble, Panel
from app.views.transcript import TranscriptWindow

# A scripted exchange, in the register the product is actually for.
SCRIPT = [
    ("incoming", "Can you finish the API integration by Friday?",
     "کیا آپ جمعہ تک API integration مکمل کر سکتے ہیں؟"),
    ("outgoing", "جی، میں جمعہ تک مکمل کر دوں گا۔",
     "Yes, I will have it done by Friday."),
    ("incoming", "We already tested the endpoint but not in production.",
     "ہم نے endpoint ٹیسٹ کر لیا ہے لیکن production میں نہیں۔"),
    ("outgoing", "میں staging پر deploy کر کے 3 بجے تک confirm کر دوں گا۔",
     "I will deploy to staging and confirm by 3 pm."),
    ("incoming", "Let us move the deployment to the next sprint.",
     "آئیے deployment کو اگلے sprint میں منتقل کر دیتے ہیں۔"),
]

STATES = [
    (Mode.PASSTHROUGH, Colour.GREY, "audio passing through, not translating"),
    (Mode.TRANSLATING, Colour.GREEN, "both directions healthy"),
    (Mode.PAUSED, Colour.AMBER, "paused - your own voice is going through"),
    (Mode.DEGRADED, Colour.RED, "incoming translation lost - reconnecting"),
]


class DemoController:
    """Drives the real widgets from a script instead of from an engine."""

    def __init__(self, auto: bool = True) -> None:
        self.bubble = Bubble()
        self.panel = Panel()
        self.transcript = TranscriptWindow()
        self.index = 0
        self.line = 0
        self.auto = auto

        self.bubble.clicked.connect(self._toggle_panel)
        self.panel.start_clicked.connect(lambda: self._set(1))
        self.panel.pause_clicked.connect(lambda: self._set(2))
        self.panel.stop_clicked.connect(lambda: self._set(0))
        self.panel.transcript_clicked.connect(self._toggle_transcript)
        self.panel.settings_clicked.connect(
            lambda: self.transcript.add_notice("info", "Settings UI is not built yet.")
        )

    # ------------------------------------------------------------------

    def _toggle_panel(self) -> None:
        if self.panel.isVisible():
            self.panel.hide()
        else:
            self.panel.popup_near(self.bubble)

    def _toggle_transcript(self) -> None:
        self.transcript.setVisible(not self.transcript.isVisible())

    def _set(self, index: int) -> None:
        self.index = index % len(STATES)
        mode, colour, reason = STATES[self.index]
        self.bubble.set_state(colour)
        self.panel.apply_state(
            {"mode": mode.value, "colour": colour.value, "reason": reason}
        )
        self.panel.apply_metrics(self.fake_metrics(mode))
        self.panel.set_devices(True, True, mode is not Mode.DEGRADED)
        if mode is Mode.DEGRADED:
            self.transcript.add_notice(
                "error",
                "Incoming translation stopped — you are hearing the original "
                "audio. You can still be heard.",
            )

    @staticmethod
    def fake_metrics(mode: Mode) -> dict:
        live = mode in (Mode.TRANSLATING, Mode.DEGRADED)

        def d(p50, p90):
            return {
                "ttfta": {"n": 12 if live else 0,
                          "p50": p50 if live else None,
                          "p90": p90 if live else None,
                          "max": p90},
                "stages": {}, "units": 12, "dropped": 0, "reconnects": 0,
                "queue_depth_ms": 180.0, "drift_rate": 1.03,
                "provider_rtt_ms": 138.0,
            }

        return {"incoming": d(1.16, 1.94), "outgoing": d(0.83, 1.35)}

    # ------------------------------------------------------------------

    def next_line(self) -> None:
        direction, source, translated = SCRIPT[self.line % len(SCRIPT)]
        self.line += 1
        self.transcript.add_entry(
            {"direction": direction, "source": source, "translated": translated}
        )
        if direction == "incoming":
            self.panel.set_quote(translated[:110])

    def cycle_state(self) -> None:
        self._set(self.index + 1)

    # ------------------------------------------------------------------

    def show(self) -> None:
        screen = QApplication.primaryScreen().availableGeometry()
        self.bubble.move(
            screen.right() - self.bubble.width() - 60,
            screen.bottom() - self.bubble.height() - 110,
        )
        self.bubble.show()
        self.transcript.move(screen.left() + 60, screen.top() + 80)
        self.transcript.show()
        self._set(1)
        self.panel.popup_near(self.bubble)
        self.panel._idle.stop()  # keep it open while previewing

        status = capture_guard.harden(self.bubble)
        self.transcript.add_notice(
            "info",
            "capture exclusion "
            + ("active" if status["excluded"] else "NOT active on this platform"),
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    demo = DemoController()
    demo.show()

    lines = QTimer()
    lines.timeout.connect(demo.next_line)
    lines.start(2600)

    states = QTimer()
    states.timeout.connect(demo.cycle_state)
    states.start(7000)

    demo.next_line()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
