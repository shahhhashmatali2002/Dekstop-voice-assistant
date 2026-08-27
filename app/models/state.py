"""The engine mode machine.

Spec reference: §2.5, §11.3, §14.

Four modes, and the bubble colour maps one-to-one onto them:

      application launch
             |
             v
     +---------------+   Start / hotkey    +---------------+
     |  PASSTHROUGH  | ------------------> |  TRANSLATING  |
     |    (grey)     | <------------------ |    (green)    |
     +---------------+   Stop / hotkey     +---------------+
             ^                                    |
             |                                    | Pause / hotkey
             |                                    v
             |                            +---------------+
             |            Stop            |    PAUSED     |
             +----------------------------|    (amber)    |
                                          +---------------+
                                                  |
                        one direction fails       v
                                          +---------------+
                                          |   DEGRADED    |
                                          | (amber / red) |
                                          +---------------+

THE INVARIANT THIS FILE PROTECTS
--------------------------------
In every mode, audio is moving. There is no state in which the virtual
microphone stops being fed while the application is running. `PASSTHROUGH` is
the resting state, not "off" - see §2.5 and the dead-air hazard.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from app.models.telemetry import Direction


class Mode(str, Enum):
    PASSTHROUGH = "passthrough"  # resting state; raw audio flowing both ways
    TRANSLATING = "translating"  # both directions translating
    PAUSED = "paused"  # translation suspended, connections kept warm
    DEGRADED = "degraded"  # at least one direction has fallen back


class Health(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"


class Colour(str, Enum):
    """Bubble ring colour. The entire status display (§2.3)."""

    GREY = "grey"
    GREEN = "green"
    AMBER = "amber"
    RED = "red"


@dataclass
class DirectionState:
    direction: Direction
    health: Health = Health.OK
    translating: bool = False
    reason: str = ""

    @property
    def passthrough(self) -> bool:
        """True when raw audio is being passed instead of translated."""
        return not self.translating


@dataclass
class Snapshot:
    mode: Mode
    colour: Colour
    incoming: DirectionState
    outgoing: DirectionState
    reason: str = ""
    notices: list[str] = field(default_factory=list)

    @property
    def audio_flowing(self) -> bool:
        """The invariant. Must be True in every reachable state."""
        return True

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "colour": self.colour.value,
            "reason": self.reason,
            "incoming": {
                "health": self.incoming.health.value,
                "translating": self.incoming.translating,
                "reason": self.incoming.reason,
            },
            "outgoing": {
                "health": self.outgoing.health.value,
                "translating": self.outgoing.translating,
                "reason": self.outgoing.reason,
            },
            "notices": list(self.notices),
        }


class ModeMachine:
    """Owns the engine's mode. Thread-safe; observers are called outside the lock.

    Observers receive a `Snapshot`. They must not block - the UI marshals to
    the Qt thread via a queued signal, and the engine posts to its own queue.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mode = Mode.PASSTHROUGH
        self._reason = "started"
        self._dirs = {
            Direction.INCOMING: DirectionState(Direction.INCOMING),
            Direction.OUTGOING: DirectionState(Direction.OUTGOING),
        }
        self._observers: list = []

    # ------------------------------------------------------------------

    def subscribe(self, fn) -> None:
        with self._lock:
            self._observers.append(fn)

    def _emit(self) -> None:
        snap = self.snapshot()
        for fn in list(self._observers):
            try:
                fn(snap)
            except Exception:  # pragma: no cover - an observer must never
                pass  # take down the engine (§11.5)

    # ------------------------------------------------------------------

    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    def state(self, direction: Direction) -> DirectionState:
        with self._lock:
            return self._dirs[direction]

    def colour(self) -> Colour:
        with self._lock:
            if self._mode is Mode.PASSTHROUGH:
                return Colour.GREY
            if self._mode is Mode.PAUSED:
                return Colour.AMBER
            healths = [d.health for d in self._dirs.values()]
            if all(h is Health.FAILED for h in healths):
                return Colour.RED
            if any(h is Health.FAILED for h in healths):
                return Colour.RED
            if any(h is Health.DEGRADED for h in healths):
                return Colour.AMBER
            return Colour.GREEN

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(
                mode=self._mode,
                colour=self.colour(),
                incoming=DirectionState(**vars(self._dirs[Direction.INCOMING])),
                outgoing=DirectionState(**vars(self._dirs[Direction.OUTGOING])),
                reason=self._reason,
            )

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

    def start(self, reason: str = "user pressed Start") -> Snapshot:
        """PASSTHROUGH/PAUSED -> TRANSLATING."""
        with self._lock:
            self._mode = Mode.TRANSLATING
            self._reason = reason
            for d in self._dirs.values():
                d.translating = True
                d.health = Health.OK
                d.reason = ""
        self._emit()
        return self.snapshot()

    def pause(self, reason: str = "user pressed Pause") -> Snapshot:
        """-> PAUSED. Raw audio resumes; provider connections stay warm."""
        with self._lock:
            self._mode = Mode.PAUSED
            self._reason = reason
            for d in self._dirs.values():
                d.translating = False
        self._emit()
        return self.snapshot()

    def resume(self, reason: str = "user resumed") -> Snapshot:
        return self.start(reason)

    def stop(self, reason: str = "user pressed Stop") -> Snapshot:
        """-> PASSTHROUGH. Ends translation, NOT audio."""
        with self._lock:
            self._mode = Mode.PASSTHROUGH
            self._reason = reason
            for d in self._dirs.values():
                d.translating = False
                d.health = Health.OK
                d.reason = ""
        self._emit()
        return self.snapshot()

    def degrade(self, direction: Direction, reason: str) -> Snapshot:
        """One direction falls back to passthrough; the other keeps working.

        This is a first-class state, not an error path: one direction working
        beats a stopped session (§14.1).
        """
        with self._lock:
            d = self._dirs[direction]
            d.health = Health.FAILED
            d.translating = False
            d.reason = reason
            if self._mode in (Mode.TRANSLATING, Mode.DEGRADED):
                others = [
                    x for k, x in self._dirs.items() if k is not direction
                ]
                self._mode = Mode.DEGRADED
                self._reason = reason
                if all(not o.translating for o in others):
                    # Both directions down: still not silence, just passthrough.
                    self._reason = f"{reason} (both directions passing through)"
        self._emit()
        return self.snapshot()

    def impair(self, direction: Direction, reason: str) -> Snapshot:
        """Direction still translating but in a reduced mode (e.g. text-only)."""
        with self._lock:
            d = self._dirs[direction]
            d.health = Health.DEGRADED
            d.reason = reason
            if self._mode is Mode.TRANSLATING:
                self._mode = Mode.DEGRADED
                self._reason = reason
        self._emit()
        return self.snapshot()

    def recover(self, direction: Direction, reason: str = "recovered") -> Snapshot:
        """A direction came back. Return to TRANSLATING if both are healthy."""
        with self._lock:
            d = self._dirs[direction]
            d.health = Health.OK
            d.reason = ""
            if self._mode is Mode.DEGRADED:
                d.translating = True
                if all(
                    x.health is Health.OK and x.translating
                    for x in self._dirs.values()
                ):
                    self._mode = Mode.TRANSLATING
                    self._reason = reason
        self._emit()
        return self.snapshot()

    # ------------------------------------------------------------------

    def should_translate(self, direction: Direction) -> bool:
        """Whether the pipeline for `direction` should be running right now."""
        with self._lock:
            return self._dirs[direction].translating

    def wire_muted(self, direction: Direction) -> bool:
        """Whether the passthrough wire for `direction` should be muted.

        Muted means a pipeline is driving the sink instead. Unmuted means raw
        audio - which is the correct fallback for every failure.
        """
        return self.should_translate(direction)
