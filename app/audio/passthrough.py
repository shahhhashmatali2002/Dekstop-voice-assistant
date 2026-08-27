"""The always-on wire.

Spec reference: §2.5, §5.7, §14.1.

WHY THIS FILE EXISTS
--------------------
Teams is bound to our virtual microphone permanently, from first-run setup
onwards. If that endpoint is not being fed, Teams transmits silence and the
client hears nothing - without knowing why. That is the default failure of
this whole architecture, and the fix is structural rather than defensive:

    Passthrough is the resting state, not "off".

Whenever the application is running but not translating - before Start, after
Stop, while Paused, while degraded - this wire is still moving audio. The
microphone goes straight to the virtual microphone and Teams goes straight to
the headphones, unprocessed. Translation is a mode that is switched into and
out of, layered on top of a wire that is always live.

Consequences encoded here:
  * the wire starts with the application, never with the Start button;
  * `Stop` ends translation, not audio;
  * muting the wire (because TTS is driving the sink instead) still emits
    comfort noise, never digital silence, so Teams' noise suppressor does not
    treat the stream as dead (§5.7).

This runs on its own pump thread, not in a callback: it resamples, and
resampling never happens in a callback.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.audio import convert
from app.audio.formats import DTYPE
from app.audio.streams import CaptureSource, RenderSink


@dataclass
class WireStats:
    pumps: int = 0
    samples_in: int = 0
    samples_out: int = 0
    dropped: int = 0
    muted_pumps: int = 0
    last_peak_dbfs: float = field(default=float("-inf"))

    def as_dict(self) -> dict:
        return {
            "pumps": self.pumps,
            "samples_in": self.samples_in,
            "samples_out": self.samples_out,
            "dropped": self.dropped,
            "muted_pumps": self.muted_pumps,
            "last_peak_dbfs": round(self.last_peak_dbfs, 1),
        }


class PassthroughWire:
    """Pumps one capture source into one render sink, converting as it goes.

    Parameters
    ----------
    source, sink:
        Already-constructed endpoints. The wire does not own their lifetime;
        start and stop them yourself.
    gain_db:
        Applied to everything passing through. Used for the optional low mix
        of the original English (§5.6 F3).
    comfort_noise_dbfs:
        Level of the noise floor emitted while muted. Set to None to emit true
        silence instead - only do that on the headphone path, never on the
        path feeding Teams.
    interval_s:
        Pump period. Shorter means lower added latency and more wakeups; 5 ms
        is a reasonable balance and adds well under one frame of delay.
    """

    def __init__(
        self,
        source: CaptureSource,
        sink: RenderSink,
        *,
        gain_db: float = 0.0,
        comfort_noise_dbfs: float | None = -70.0,
        interval_s: float = 0.005,
        name: str = "wire",
    ) -> None:
        self.source = source
        self.sink = sink
        self.gain_db = gain_db
        self.comfort_noise_dbfs = comfort_noise_dbfs
        self.interval_s = interval_s
        self.name = name
        self.stats = WireStats()

        self._muted = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # mode
    # ------------------------------------------------------------------

    @property
    def muted(self) -> bool:
        """True when translation is driving the sink instead of this wire."""
        return self._muted.is_set()

    def mute(self) -> None:
        """Stop passing source audio. The sink keeps receiving comfort noise.

        Called when a pipeline takes over the sink - the client must hear the
        synthesised English, not the consultant's raw Urdu.
        """
        self._muted.set()

    def unmute(self) -> None:
        """Resume passing source audio. This is the fallback for every failure."""
        self._muted.clear()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._pump, name=f"passthrough-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None

    def __enter__(self) -> "PassthroughWire":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # the pump - engine thread, may allocate and resample
    # ------------------------------------------------------------------

    def _pump(self) -> None:
        src_rate = self.source.rate
        src_ch = self.source.channels
        dst_rate = self.sink.rate
        dst_ch = self.sink.channels

        # How many samples of comfort noise to emit per pump while muted, so
        # the sink's ring never drains to nothing.
        comfort_n = max(1, int(dst_rate * dst_ch * self.interval_s))

        while not self._stop.is_set():
            self.stats.pumps += 1

            if self._muted.is_set():
                self.stats.muted_pumps += 1
                # Only top up if the sink is running dry: if a pipeline is
                # feeding it, we must not interleave noise into real speech.
                if self.comfort_noise_dbfs is not None and self.sink.ring.available < comfort_n:
                    noise = convert.comfort_noise(
                        comfort_n, self.comfort_noise_dbfs, rng=self._rng
                    )
                    self.sink.ring.write(noise)
                self._stop.wait(self.interval_s)
                continue

            raw = self.source.ring.read_available()
            if raw.size == 0:
                self._stop.wait(self.interval_s)
                continue

            self.stats.samples_in += raw.size

            mono = convert.to_mono(raw, src_ch)
            if self.gain_db:
                mono = convert.apply_gain_db(mono, self.gain_db)
            self.stats.last_peak_dbfs = convert.peak_dbfs(mono)

            out = convert.resample(mono, src_rate, dst_rate)
            out = convert.to_channels(out, dst_ch)

            self.stats.dropped += self.sink.ring.write(out)
            self.stats.samples_out += out.size

            self._stop.wait(self.interval_s)

    # ------------------------------------------------------------------

    def health(self) -> dict:
        return {
            "wire": self.stats.as_dict(),
            "source": self.source.health(),
            "sink": self.sink.health(),
            "muted": self.muted,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PassthroughWire({self.name!r}, "
            f"{self.source.device.name!r} -> {self.sink.device.name!r}, "
            f"muted={self.muted})"
        )


# ----------------------------------------------------------------------


class PassthroughPair:
    """Both wires: microphone -> virtual mic, and Teams -> headphones.

    This is what the application starts at launch and stops at exit. It is the
    object that guarantees the client never hears dead air.
    """

    def __init__(self, outgoing: PassthroughWire, incoming: PassthroughWire) -> None:
        self.outgoing = outgoing  # mic -> virtual mic  (the client's ears)
        self.incoming = incoming  # Teams -> headphones (the consultant's ears)

    def start(self) -> None:
        self.outgoing.start()
        self.incoming.start()

    def stop(self) -> None:
        self.outgoing.stop()
        self.incoming.stop()

    def mute_all(self) -> None:
        """Translation is now driving both sinks."""
        self.outgoing.mute()
        self.incoming.mute()

    def unmute_all(self) -> None:
        """Fall back to raw audio in both directions."""
        self.outgoing.unmute()
        self.incoming.unmute()

    def __enter__(self) -> "PassthroughPair":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def health(self) -> dict:
        return {"outgoing": self.outgoing.health(), "incoming": self.incoming.health()}
