"""Single-producer / single-consumer ring buffer over a pre-allocated array.

Spec reference: §4.3, §5.5.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
A PortAudio callback runs on a C thread that must acquire the GIL before it
can execute a line of Python. If another thread holds the GIL when the
callback fires, the callback is late, and late means an audible click.

So the callback does exactly one thing: a pre-sized numpy slice assignment,
then returns. No allocation, no logging, no locks, no blocking queue, no
format conversion, no provider call.

Every method here is written to that constraint:

  * the backing array is allocated once, in __init__, and never resized;
  * `write` and `read_into` allocate nothing;
  * neither ever blocks - overflow drops the oldest samples and counts it,
    underrun emits silence and counts it.

Correctness note: this is safe for exactly one producer thread and one
consumer thread. The read and write cursors are monotonic Python ints, and
each is written by only one side, so no torn state is observable under
CPython. Do not attach a second producer or a second consumer.
"""

from __future__ import annotations

import numpy as np

from app.audio.formats import DTYPE, samples_to_ms


class RingBuffer:
    """Lock-free SPSC ring buffer holding `capacity` samples.

    Parameters
    ----------
    capacity:
        Size of the backing array in samples. Size it for the worst case you
        intend to tolerate - roughly 200 ms of headroom is the starting point
        in the spec - and tune from the observed `overflows` counter.
    dtype:
        Sample dtype. Defaults to the canonical int16.

    Attributes
    ----------
    overflows:
        Number of `write` calls that had to discard old samples.
    dropped_samples:
        Total samples discarded by overflow.
    underruns:
        Number of `read_into` / `read` calls that had to emit silence.
    silence_samples:
        Total samples of silence emitted by underrun.
    """

    __slots__ = (
        "_buf",
        "_cap",
        "_w",
        "_r",
        "overflows",
        "dropped_samples",
        "underruns",
        "silence_samples",
    )

    def __init__(self, capacity: int, dtype=DTYPE) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._buf = np.zeros(capacity, dtype=dtype)
        self._cap = capacity
        # Monotonic sample counters. available == _w - _r, which removes the
        # classic full/empty ambiguity of head/tail indices.
        self._w = 0
        self._r = 0
        self.overflows = 0
        self.dropped_samples = 0
        self.underruns = 0
        self.silence_samples = 0

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Backing array size in samples."""
        return self._cap

    @property
    def available(self) -> int:
        """Samples currently readable."""
        return self._w - self._r

    @property
    def free(self) -> int:
        """Samples that can be written before the oldest data is displaced."""
        return self._cap - (self._w - self._r)

    def depth_ms(self, rate: int) -> float:
        """Readable depth in milliseconds at `rate`. Used by drift control."""
        return samples_to_ms(self.available, rate)

    # ------------------------------------------------------------------
    # producer side - safe to call from an audio callback
    # ------------------------------------------------------------------

    def write(self, data: np.ndarray) -> int:
        """Append `data`. Never blocks. Returns the number of samples dropped.

        If `data` does not fit, the oldest samples are discarded to make room
        rather than the caller being made to wait. An audio thread must never
        wait for a slow reader.
        """
        n = data.shape[0]
        if n == 0:
            return 0

        dropped = 0

        # A write larger than the whole buffer can only keep its own tail.
        if n >= self._cap:
            dropped = n - self._cap
            data = data[-self._cap:]
            n = self._cap
            # Everything previously held is displaced too, and must be counted:
            # the metrics invariant is consumed + dropped + available == written.
            dropped += self._w - self._r
            self._r = self._w

        need = n - self.free
        if need > 0:
            # Displace the oldest samples by advancing the read cursor.
            self._r += need
            dropped += need

        if dropped:
            self.overflows += 1
            self.dropped_samples += dropped

        start = self._w % self._cap
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = data
        else:
            split = self._cap - start
            self._buf[start:] = data[:split]
            self._buf[: n - split] = data[split:]

        self._w += n
        return dropped

    # ------------------------------------------------------------------
    # consumer side - safe to call from an audio callback
    # ------------------------------------------------------------------

    def read_into(self, out: np.ndarray) -> int:
        """Fill `out` completely, padding with silence. Returns samples padded.

        This is the render-callback entry point: `out` is pre-allocated by the
        caller and reused every period, so nothing is allocated here. Short
        reads are filled with zeros rather than stale audio - repeating the
        last buffer is far more audible than a gap.
        """
        n = out.shape[0]
        if n == 0:
            return 0

        take = min(n, self.available)
        if take:
            start = self._r % self._cap
            end = start + take
            if end <= self._cap:
                out[:take] = self._buf[start:end]
            else:
                split = self._cap - start
                out[:split] = self._buf[start:]
                out[split:take] = self._buf[: take - split]
            self._r += take

        pad = n - take
        if pad:
            out[take:] = 0
            self.underruns += 1
            self.silence_samples += pad
        return pad

    def read(self, n: int) -> np.ndarray:
        """Allocating convenience read. NEVER call this from a callback.

        Provided for the engine thread and for tests, where an allocation is
        harmless. The callback path is `read_into`.
        """
        out = np.empty(n, dtype=self._buf.dtype)
        self.read_into(out)
        return out

    def read_available(self, limit: int | None = None) -> np.ndarray:
        """Read everything currently available, optionally capped at `limit`.

        Allocating - engine thread only. Returns an empty array when idle.
        """
        n = self.available
        if limit is not None:
            n = min(n, limit)
        if n <= 0:
            return np.empty(0, dtype=self._buf.dtype)
        return self.read(n)

    def discard(self, n: int | None = None) -> int:
        """Drop `n` samples (default: all). Used when flushing on barge-in."""
        avail = self.available
        n = avail if n is None else min(n, avail)
        self._r += n
        return n

    # ------------------------------------------------------------------

    def reset_counters(self) -> None:
        """Zero the health counters. Does not touch buffered audio."""
        self.overflows = 0
        self.dropped_samples = 0
        self.underruns = 0
        self.silence_samples = 0

    def health(self) -> dict[str, int]:
        """Snapshot of the counters, for the metrics bus."""
        return {
            "overflows": self.overflows,
            "dropped_samples": self.dropped_samples,
            "underruns": self.underruns,
            "silence_samples": self.silence_samples,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"RingBuffer(capacity={self._cap}, available={self.available}, "
            f"overflows={self.overflows}, underruns={self.underruns})"
        )
