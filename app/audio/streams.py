"""Capture sources and render sinks.

Spec reference: §4.3, §5.1, §5.5.

WINDOWS ONLY.

THE CALLBACK DISCIPLINE
-----------------------
The callbacks below are the two most safety-critical functions in the project.
They run on PortAudio's C threads and must acquire the GIL to execute. If they
are slow, the user hears a click.

Each one does exactly this and nothing else:

    capture:  numpy view over the incoming bytes -> ring.write() -> return
    render:   ring.read_into(pre-allocated array) -> return its bytes

No allocation of new arrays, no resampling, no downmix, no logging, no locks,
no queue that can block. Format conversion happens on the engine thread, in
`va.audio.convert`. If you are tempted to add a line to a callback, add it to
the engine thread instead.

The one unavoidable allocation is `tobytes()` on the render path, which
PortAudio's non-blocking API requires. It is a fixed-size copy of one period.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from app.audio.devices import AudioUnavailable, DeviceManager, DeviceRef
from app.audio.formats import DTYPE
from app.audio.ring import RingBuffer

if sys.platform == "win32":  # pragma: no cover - platform guard
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        pyaudio = None
else:
    pyaudio = None


@dataclass
class StreamStats:
    """Health of one open stream. Read by the metrics bus; never by a callback."""

    callbacks: int = 0
    frames: int = 0
    late: int = 0  # PortAudio reported an overflow/underflow status flag

    def as_dict(self) -> dict[str, int]:
        return {"callbacks": self.callbacks, "frames": self.frames, "late": self.late}


# ----------------------------------------------------------------------


class CaptureSource:
    """Reads from a WASAPI capture endpoint into a ring buffer.

    Works for both a physical microphone and a loopback endpoint - the only
    difference is which `DeviceRef` you hand it, which is exactly the seam
    §5.2 asks for so that process loopback can be dropped in later.

    The ring holds the device's native format: whatever rate and channel count
    the endpoint negotiated. Conversion to the canonical pipeline format is the
    consumer's job, on the engine thread.
    """

    def __init__(
        self,
        manager: DeviceManager,
        device: DeviceRef,
        *,
        rate: int | None = None,
        channels: int | None = None,
        frames_per_buffer: int = 480,
        ring_ms: int = 500,
    ) -> None:
        if pyaudio is None:  # pragma: no cover
            raise AudioUnavailable("PyAudioWPatch is required for audio capture")

        self._manager = manager
        self.device = manager.resolve(device)
        self.rate = rate or self.device.default_rate
        self.channels = channels or min(self.device.max_input_channels, 2)
        self.frames_per_buffer = frames_per_buffer
        self.stats = StreamStats()

        capacity = int(self.rate * self.channels * ring_ms / 1000)
        self.ring = RingBuffer(capacity)

        self._stream = None

    # ------------------------------------------------------------------

    def _callback(self, in_data, frame_count, time_info, status):  # pragma: no cover
        # ---- REAL-TIME CONTEXT. See the module docstring before editing. ----
        if status:
            self.stats.late += 1
        self.ring.write(np.frombuffer(in_data, dtype=DTYPE))
        self.stats.callbacks += 1
        self.stats.frames += frame_count
        return (None, pyaudio.paContinue)
        # ---------------------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._manager.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.is_active()

    def __enter__(self) -> "CaptureSource":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def health(self) -> dict[str, int]:
        return {**self.ring.health(), **self.stats.as_dict()}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CaptureSource({self.device.name!r}, {self.rate} Hz, "
            f"{self.channels}ch, active={self.active})"
        )


# ----------------------------------------------------------------------


class RenderSink:
    """Writes a ring buffer out to a WASAPI render endpoint.

    Underrun produces silence, not stale audio, and is counted. That counter
    is one of the two numbers that tell you whether the GIL discipline is
    holding (§15.3).
    """

    def __init__(
        self,
        manager: DeviceManager,
        device: DeviceRef,
        *,
        rate: int | None = None,
        channels: int | None = None,
        frames_per_buffer: int = 480,
        ring_ms: int = 500,
    ) -> None:
        if pyaudio is None:  # pragma: no cover
            raise AudioUnavailable("PyAudioWPatch is required for audio render")

        self._manager = manager
        self.device = manager.resolve(device)
        self.rate = rate or self.device.default_rate
        self.channels = channels or min(self.device.max_output_channels, 2)
        self.frames_per_buffer = frames_per_buffer
        self.stats = StreamStats()

        capacity = int(self.rate * self.channels * ring_ms / 1000)
        self.ring = RingBuffer(capacity)

        # Pre-allocated once, reused by every callback: the whole point.
        self._out = np.zeros(frames_per_buffer * self.channels, dtype=DTYPE)
        self._stream = None

    # ------------------------------------------------------------------

    def _callback(self, in_data, frame_count, time_info, status):  # pragma: no cover
        # ---- REAL-TIME CONTEXT. See the module docstring before editing. ----
        if status:
            self.stats.late += 1
        want = frame_count * self.channels
        if want == self._out.shape[0]:
            self.ring.read_into(self._out)
            data = self._out.tobytes()
        else:
            # PortAudio asked for a period we were not sized for. Rare, but it
            # happens on device changes; serve it correctly rather than crash.
            tmp = np.empty(want, dtype=DTYPE)
            self.ring.read_into(tmp)
            data = tmp.tobytes()
        self.stats.callbacks += 1
        self.stats.frames += frame_count
        return (data, pyaudio.paContinue)
        # ---------------------------------------------------------------------

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = self._manager.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            output=True,
            output_device_index=self.device.index,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop_stream()
            self._stream.close()
        finally:
            self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None and self._stream.is_active()

    def write(self, pcm: np.ndarray) -> int:
        """Queue audio for playback. Engine thread only. Returns samples dropped."""
        return self.ring.write(pcm)

    def flush(self) -> int:
        """Discard everything queued. Used on barge-in (§7.2)."""
        return self.ring.discard()

    def __enter__(self) -> "RenderSink":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def health(self) -> dict[str, int]:
        return {**self.ring.health(), **self.stats.as_dict()}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RenderSink({self.device.name!r}, {self.rate} Hz, "
            f"{self.channels}ch, active={self.active})"
        )
