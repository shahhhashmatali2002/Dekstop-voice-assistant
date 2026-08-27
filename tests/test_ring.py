"""Tests for the SPSC ring buffer.

These are the tests that matter most in the whole project: the ring buffer is
the boundary between the real-time audio callbacks and everything else, and a
bug here surfaces as an intermittent click that is almost impossible to trace
back from a user report.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.audio.formats import DTYPE
from app.audio.ring import RingBuffer


def seq(n: int, start: int = 0) -> np.ndarray:
    """A ramp, so wrap-around bugs show up as out-of-order values."""
    return np.arange(start, start + n, dtype=DTYPE)


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_rejects_nonpositive_capacity(bad):
    with pytest.raises(ValueError):
        RingBuffer(bad)


def test_starts_empty():
    r = RingBuffer(100)
    assert r.available == 0
    assert r.free == 100
    assert r.capacity == 100


# ----------------------------------------------------------------------
# basic round trip
# ----------------------------------------------------------------------


def test_write_then_read_roundtrip():
    r = RingBuffer(100)
    data = seq(10)
    assert r.write(data) == 0
    assert r.available == 10
    np.testing.assert_array_equal(r.read(10), data)
    assert r.available == 0


def test_empty_write_is_a_noop():
    r = RingBuffer(10)
    assert r.write(np.empty(0, dtype=DTYPE)) == 0
    assert r.available == 0


def test_interleaved_writes_and_reads_preserve_order():
    r = RingBuffer(64)
    written = []
    read = []
    pos = 0
    for i in range(50):
        chunk = seq(7, pos)
        pos += 7
        r.write(chunk)
        written.append(chunk)
        if i % 2:
            read.append(r.read_available())
    read.append(r.read_available())
    np.testing.assert_array_equal(
        np.concatenate(read), np.concatenate(written)[-len(np.concatenate(read)):]
    )


# ----------------------------------------------------------------------
# wrap-around
# ----------------------------------------------------------------------


def test_wraps_correctly():
    r = RingBuffer(10)
    r.write(seq(8))
    r.read(8)  # cursor now at 8, next write must straddle the end
    data = seq(6, 100)
    r.write(data)
    np.testing.assert_array_equal(r.read(6), data)


def test_many_wraps_stay_consistent():
    r = RingBuffer(13)  # deliberately coprime with the chunk size
    pos = 0
    for _ in range(500):
        chunk = seq(5, pos)
        pos = (pos + 5) % 30000
        r.write(chunk)
        np.testing.assert_array_equal(r.read(5), chunk)
    assert r.overflows == 0
    assert r.underruns == 0


# ----------------------------------------------------------------------
# overflow: drop oldest, count it, never block
# ----------------------------------------------------------------------


def test_overflow_drops_oldest_and_counts():
    r = RingBuffer(10)
    r.write(seq(8))
    dropped = r.write(seq(6, 100))  # needs 6, only 2 free -> drop 4 oldest
    assert dropped == 4
    assert r.overflows == 1
    assert r.dropped_samples == 4
    assert r.available == 10
    out = r.read(10)
    # The four oldest ramp values are gone; the new data is intact at the end.
    np.testing.assert_array_equal(out[:4], seq(4, 4))
    np.testing.assert_array_equal(out[4:], seq(6, 100))


def test_write_larger_than_capacity_keeps_the_tail():
    r = RingBuffer(10)
    r.write(seq(3))
    data = seq(25, 100)
    dropped = r.write(data)
    assert dropped == 18  # 15 beyond capacity + the 3 already held
    assert r.available == 10
    np.testing.assert_array_equal(r.read(10), data[-10:])


def test_write_exactly_capacity_from_empty_drops_nothing():
    r = RingBuffer(10)
    assert r.write(seq(10)) == 0
    assert r.overflows == 0
    assert r.available == 10


def test_overflow_never_raises_under_sustained_pressure():
    """The producer must survive a consumer that has stopped entirely."""
    r = RingBuffer(64)
    for i in range(1000):
        r.write(seq(20, i))
    assert r.available == 64
    assert r.overflows > 0


# ----------------------------------------------------------------------
# underrun: silence, not stale audio
# ----------------------------------------------------------------------


def test_underrun_pads_with_silence_and_counts():
    r = RingBuffer(100)
    r.write(seq(3, 1))
    out = np.full(10, 99, dtype=DTYPE)
    pad = r.read_into(out)
    assert pad == 7
    assert r.underruns == 1
    assert r.silence_samples == 7
    np.testing.assert_array_equal(out[:3], seq(3, 1))
    np.testing.assert_array_equal(out[3:], np.zeros(7, dtype=DTYPE))


def test_read_from_empty_is_all_silence():
    r = RingBuffer(10)
    out = np.full(5, 42, dtype=DTYPE)
    assert r.read_into(out) == 5
    np.testing.assert_array_equal(out, np.zeros(5, dtype=DTYPE))


def test_underrun_never_repeats_stale_audio():
    """Repeating the previous buffer is far more audible than a gap."""
    r = RingBuffer(100)
    r.write(np.full(10, 5000, dtype=DTYPE))
    out = np.empty(10, dtype=DTYPE)
    r.read_into(out)
    r.read_into(out)  # nothing left
    np.testing.assert_array_equal(out, np.zeros(10, dtype=DTYPE))


# ----------------------------------------------------------------------
# no allocation on the callback path
# ----------------------------------------------------------------------


def test_read_into_does_not_allocate_a_new_array():
    r = RingBuffer(100)
    r.write(seq(50))
    out = np.empty(20, dtype=DTYPE)
    before = out.__array_interface__["data"][0]
    r.read_into(out)
    assert out.__array_interface__["data"][0] == before


def test_read_into_zero_length_is_safe():
    r = RingBuffer(10)
    assert r.read_into(np.empty(0, dtype=DTYPE)) == 0


# ----------------------------------------------------------------------
# discard / flush, used on barge-in
# ----------------------------------------------------------------------


def test_discard_all():
    r = RingBuffer(50)
    r.write(seq(30))
    assert r.discard() == 30
    assert r.available == 0


def test_discard_partial_keeps_the_newest():
    r = RingBuffer(50)
    r.write(seq(30))
    assert r.discard(10) == 10
    assert r.available == 20
    np.testing.assert_array_equal(r.read(20), seq(20, 10))


def test_discard_more_than_held_is_clamped():
    r = RingBuffer(50)
    r.write(seq(5))
    assert r.discard(999) == 5
    assert r.available == 0


# ----------------------------------------------------------------------
# health reporting
# ----------------------------------------------------------------------


def test_health_and_reset():
    r = RingBuffer(4)
    r.write(seq(10))
    r.read(10)
    h = r.health()
    assert h["overflows"] > 0 and h["underruns"] > 0
    r.reset_counters()
    assert r.health() == {
        "overflows": 0,
        "dropped_samples": 0,
        "underruns": 0,
        "silence_samples": 0,
    }


def test_depth_ms():
    r = RingBuffer(16_000)
    r.write(np.zeros(1600, dtype=DTYPE))
    assert r.depth_ms(16_000) == pytest.approx(100.0)


# ----------------------------------------------------------------------
# concurrency: one producer, one consumer, no corruption
# ----------------------------------------------------------------------


def test_single_producer_single_consumer_under_threads():
    """Simulates the real topology: a callback thread and an engine thread."""
    r = RingBuffer(2048)
    total = 200_000
    chunk = 320  # one 20 ms frame at 16 kHz
    produced = 0
    stop = threading.Event()

    def producer():
        nonlocal produced
        buf = np.ones(chunk, dtype=DTYPE)
        while produced < total:
            r.write(buf)
            produced += chunk
        stop.set()

    consumed = 0
    corrupt = 0

    def consumer():
        nonlocal consumed, corrupt
        out = np.empty(chunk, dtype=DTYPE)
        while not stop.is_set() or r.available:
            pad = r.read_into(out)
            real = chunk - pad
            if real and not np.all(out[:real] == 1):
                corrupt += 1
            consumed += real

    t1 = threading.Thread(target=producer)
    t2 = threading.Thread(target=consumer)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)

    assert not t1.is_alive() and not t2.is_alive()
    assert corrupt == 0, "ring buffer returned data the producer never wrote"
    # Overflow is legitimate here (the consumer may lag); the invariant is that
    # nothing is corrupted and the accounting adds up.
    assert consumed + r.dropped_samples + r.available == produced
