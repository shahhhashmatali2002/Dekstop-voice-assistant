"""Show failures in the interface, not only in a terminal nobody is watching.

An unhandled exception in a desktop application should not vanish into a
console window. The user is looking at the overlay; that is where a problem
has to appear, with enough detail to act on and enough detail to report.

Two entry points:

    install(app)        - route unhandled exceptions to a dialog
    show_error(...)     - report one failure deliberately

Both are safe to call from any thread: Qt insists that widgets are only
touched from the GUI thread, so anything arriving from the engine thread is
marshalled across first.
"""

from __future__ import annotations

import sys
import traceback

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QApplication, QMessageBox


class _Reporter(QObject):
    """Carries an error from whatever thread raised it onto the GUI thread."""

    failed = Signal(str, str, str)

    def __init__(self) -> None:
        super().__init__()
        self.failed.connect(self._show, Qt.QueuedConnection)
        self._seen: set[str] = set()

    def _show(self, title: str, message: str, detail: str) -> None:
        # The same fault can fire every frame; showing it once is a report,
        # showing it sixty times a second is a denial of service.
        key = f"{title}|{message}"
        if key in self._seen:
            return
        self._seen.add(key)

        box = QMessageBox()
        box.setWindowTitle(title)
        box.setIcon(QMessageBox.Critical)
        box.setText(message)
        if detail:
            box.setDetailedText(detail)
        box.setStandardButtons(QMessageBox.Ok)
        try:
            from app.views import capture_guard

            box.show()
            capture_guard.harden(box, unfocusable=False)
        except Exception:
            pass
        box.exec()

    def reset(self) -> None:
        self._seen.clear()


_reporter: _Reporter | None = None


def _get() -> _Reporter | None:
    global _reporter
    if _reporter is None and QApplication.instance() is not None:
        _reporter = _Reporter()
    return _reporter


def show_error(message: str, exc: BaseException | None = None,
               title: str = "Translator problem") -> None:
    """Report a failure to the user. Falls back to stderr with no GUI."""
    detail = ""
    if exc is not None:
        detail = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    rep = _get()
    if rep is None:
        print(f"{title}: {message}\n{detail}", file=sys.stderr)
        return
    rep.failed.emit(title, message, detail)


def install(app: QApplication) -> None:
    """Route unhandled exceptions - on any thread - into a dialog."""
    _get()

    previous = sys.excepthook

    def hook(kind, value, tb):
        if issubclass(kind, KeyboardInterrupt):
            previous(kind, value, tb)
            return
        show_error(
            f"{kind.__name__}: {value}",
            value,
            title="Unexpected error",
        )
        previous(kind, value, tb)

    sys.excepthook = hook

    # Exceptions raised on a non-main thread bypass sys.excepthook in 3.8+.
    try:
        import threading

        def thread_hook(args):
            if issubclass(args.exc_type, KeyboardInterrupt):
                return
            show_error(
                f"{args.exc_type.__name__}: {args.exc_value}",
                args.exc_value,
                title=f"Error on {args.thread.name if args.thread else 'a thread'}",
            )

        threading.excepthook = thread_hook
    except Exception:
        pass
