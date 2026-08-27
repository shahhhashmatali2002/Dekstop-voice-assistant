"""Global hotkeys.

Spec reference: §2.3 rule 3, §11.4.

A target the size of a coin is not something to aim at mid-sentence in front
of a client. Every control has a keyboard equivalent that works while Teams
has focus.

    Ctrl+Alt+Space   toggle Pause / Resume   the most time-critical control
    Ctrl+Alt+T       Start / Stop            starting mid-meeting
    Ctrl+Alt+H       show / hide the bubble  belt-and-braces before sharing
    Ctrl+Alt+R       repeat last utterance   recovering a dropped sentence

Registered through the Win32 RegisterHotKey API on a dedicated thread, because
Qt cannot see keystrokes destined for another application's window - and the
whole point is that these work while Teams is focused.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312

_MODS = {
    "ctrl": MOD_CONTROL,
    "control": MOD_CONTROL,
    "alt": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN,
}

_KEYS = {
    "space": 0x20,
    "escape": 0x1B,
    "esc": 0x1B,
    "tab": 0x09,
    "enter": 0x0D,
    "return": 0x0D,
}


def parse(spec: str) -> tuple[int, int]:
    """"Ctrl+Alt+Space" -> (modifiers, virtual key code)."""
    parts = [p.strip().lower() for p in spec.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"empty hotkey: {spec!r}")
    mods = 0
    key = None
    for part in parts:
        if part in _MODS:
            mods |= _MODS[part]
        elif part in _KEYS:
            key = _KEYS[part]
        elif len(part) == 1:
            key = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit():
            key = 0x70 + int(part[1:]) - 1  # VK_F1 = 0x70
        else:
            raise ValueError(f"unrecognised key {part!r} in {spec!r}")
    if key is None:
        raise ValueError(f"no non-modifier key in {spec!r}")
    return mods | MOD_NOREPEAT, key


#: Tried in order when the preferred combination is already taken. Ctrl+Alt+Space
#: in particular is claimed by Teams' own push-to-talk and by most IMEs, so a
#: conflict there is the common case rather than the exception.
FALLBACKS = {
    "Ctrl+Alt+Space": ["Ctrl+Shift+Space", "Ctrl+Alt+P", "Ctrl+Shift+F9", "Alt+F9"],
    "Ctrl+Alt+T": ["Ctrl+Shift+T", "Ctrl+Alt+F10", "Alt+F10"],
    "Ctrl+Alt+H": ["Ctrl+Shift+H", "Ctrl+Alt+F11", "Alt+F11"],
    "Ctrl+Alt+R": ["Ctrl+Shift+R", "Ctrl+Alt+F12", "Alt+F12"],
}


class HotkeyManager:
    """Owns a message-loop thread and the registered hotkeys.

    Callbacks fire on the hotkey thread. They must not touch Qt objects
    directly - post to the Qt thread with a queued signal (§11.5).
    """

    def __init__(self) -> None:
        self._bindings: dict[int, tuple[str, object]] = {}
        self._thread: threading.Thread | None = None
        self._tid = 0
        self._next_id = 1
        self._failed: list[str] = []
        self._bound: dict[str, str] = {}
        self._ready = threading.Event()

    @property
    def failed(self) -> list[str]:
        """Combinations no alternative could rescue. Surface these in settings."""
        return list(self._failed)

    @property
    def bound(self) -> dict[str, str]:
        """What each requested combination ended up as: requested -> actual."""
        return dict(self._bound)

    def reassigned(self) -> dict[str, str]:
        """Only the ones that had to move, for reporting to the user."""
        return {k: v for k, v in self._bound.items() if k != v}

    def bind(self, spec: str, callback) -> None:
        self._bindings[self._next_id] = (spec, callback)
        self._next_id += 1

    # ------------------------------------------------------------------

    def start(self) -> bool:
        if sys.platform != "win32" or not self._bindings:
            return False
        self._thread = threading.Thread(
            target=self._run, name="hotkeys", daemon=True
        )
        self._thread.start()
        return self._ready.wait(timeout=3.0)

    def stop(self) -> None:
        if self._tid:
            ctypes.windll.user32.PostThreadMessageW(  # type: ignore[attr-defined]
                self._tid, 0x0012, 0, 0  # WM_QUIT
            )
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _run(self) -> None:  # pragma: no cover - Windows message loop
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        self._tid = ctypes.windll.kernel32.GetCurrentThreadId()  # type: ignore

        for hid, (spec, _) in self._bindings.items():
            # Try the preferred combination, then the fallbacks, and record
            # which one actually took. A hotkey somebody else owns should cost
            # the user a different shortcut, not the feature.
            for candidate in [spec, *FALLBACKS.get(spec, [])]:
                try:
                    mods, key = parse(candidate)
                except ValueError:
                    continue
                if user32.RegisterHotKey(None, hid, mods, key):
                    self._bound[spec] = candidate
                    break
            else:
                self._failed.append(spec)

        self._ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY:
                    entry = self._bindings.get(int(msg.wParam))
                    if entry:
                        try:
                            entry[1]()
                        except Exception:
                            pass  # a handler must never kill the loop
        finally:
            for hid in self._bindings:
                user32.UnregisterHotKey(None, hid)
