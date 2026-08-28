"""Exclude every window from screen capture.

Spec reference: §2.4, §11.1, §13.3.

THE PROBLEM
-----------
The consultant shares their screen. The overlay is on top of everything, so it
is now on the client's screen too - a badge reading "Urdu <-> English ·
RUNNING", with a live transcript of the client's own words next to it.
Whatever the consultant intended to disclose, and however they intended to
phrase it, that decision has now been made for them by a UI element.

THE FIX
-------
SetWindowDisplayAffinity with WDA_EXCLUDEFROMCAPTURE (0x11) makes the desktop
compositor render the window to the physical monitor but omit it from any
screen capture - screenshots, recorders, and the screen-sharing path Teams
uses. Available from Windows 10 version 2004, which is why that is the minimum
supported version.

APPLY IT TO EVERY TOP-LEVEL WINDOW WITHOUT EXCEPTION - the bubble, the panel,
the transcript, settings dialogs, the debug view, tooltips and the tray menu
popup. A single window that misses it defeats the entire measure the first
time the consultant shares their screen.

TWO CAVEATS TO RECORD
---------------------
1. It requires the Desktop Window Manager to be composing. That is the normal
   state but is not guaranteed.
2. Microsoft is explicit that this is NOT a security feature. It will not
   survive a phone camera pointed at the screen, and it must never be relied
   on as if it were a confidentiality control.

Verify per build, from the far side of a real Teams call (§15.4). This
behaviour depends on the capture path in use and is exactly the kind of thing
that silently regresses.
"""

from __future__ import annotations

import sys

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001
WDA_EXCLUDEFROMCAPTURE = 0x00000011

_IS_WINDOWS = sys.platform == "win32"


def supported() -> bool:
    """True when the running OS can honour WDA_EXCLUDEFROMCAPTURE.

    On Windows earlier than 10.0.19041 (version 2004) the flag silently falls
    back to WDA_MONITOR, which blanks the window in captures rather than
    removing it - visible as a black rectangle, which is arguably worse than
    showing the overlay because it looks like a fault.
    """
    if not _IS_WINDOWS:
        return False
    try:
        version = sys.getwindowsversion()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return False
    return (version.major, version.minor, version.build) >= (10, 0, 19041)


def exclude_from_capture(window) -> bool:
    """Apply the affinity to a Qt window. Returns True on success.

    Call AFTER the window has a native handle - i.e. after show() or
    winId() - because the handle does not exist before then.
    """
    if not _IS_WINDOWS:
        return False
    try:
        import ctypes

        hwnd = int(window.winId())
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetWindowDisplayAffinity.restype = ctypes.c_bool
        return bool(
            user32.SetWindowDisplayAffinity(
                ctypes.c_void_p(hwnd), ctypes.c_uint(WDA_EXCLUDEFROMCAPTURE)
            )
        )
    except Exception:
        return False


def allow_capture(window) -> bool:
    """Undo the exclusion. Only for the setup wizard's verification step."""
    if not _IS_WINDOWS:
        return False
    try:
        import ctypes

        hwnd = int(window.winId())
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.SetWindowDisplayAffinity.restype = ctypes.c_bool
        return bool(
            user32.SetWindowDisplayAffinity(
                ctypes.c_void_p(hwnd), ctypes.c_uint(WDA_NONE)
            )
        )
    except Exception:
        return False


def no_activate(window) -> bool:
    """Stop the window taking focus when clicked.

    Spec reference: §2.3 rule 1, §11.2.

    Clicking the bubble while Teams is focused must not take focus away from
    Teams - otherwise the consultant loses their place and keyboard shortcuts
    stop reaching the meeting. Qt's WindowDoesNotAcceptFocus covers most of
    it; WS_EX_NOACTIVATE is what makes it reliable on Windows.
    """
    if not _IS_WINDOWS:
        return False
    try:
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_NOACTIVATE = 0x08000000
        WS_EX_TOOLWINDOW = 0x00000080  # also keeps it out of alt-tab

        hwnd = int(window.winId())
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        get_l = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_l = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        style = get_l(ctypes.c_void_p(hwnd), GWL_EXSTYLE)
        set_l(
            ctypes.c_void_p(hwnd),
            GWL_EXSTYLE,
            style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW,
        )
        return True
    except Exception:
        return False


#: Process-wide switch for §2.4. True is the shipping default and the only
#: value that is safe in front of a client: an always-on-top overlay carrying
#: a live transcript of the other person's words appears in a screen share
#: unless it is excluded. It is turned off deliberately - to photograph the
#: interface, or to demonstrate it over a call - and never quietly.
_ENABLED = True


def set_enabled(on: bool) -> None:
    """Turn capture exclusion on or off for every window in this process."""
    global _ENABLED
    _ENABLED = bool(on)


def enabled() -> bool:
    return _ENABLED


def harden(window, *, exclude: bool = True, unfocusable: bool = True) -> dict:
    """Apply every protection a top-level window needs. Call once, after show().

    Returns what actually took effect, so the caller can warn the user rather
    than assume - an overlay that silently failed to exclude itself is the
    §13.3 disclosure risk realised.
    """
    exclude = exclude and _ENABLED
    return {
        "supported": supported(),
        "excluded": exclude_from_capture(window) if exclude else False,
        "no_activate": no_activate(window) if unfocusable else False,
    }
