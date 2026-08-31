"""Prove a cable works, instead of believing it does.

Every hour lost on this application's audio has come from the same thing: the
software chose a device, assumed the wire behind it carried sound, and only
found out otherwise in the middle of a call - by which time three other things
had been changed and nobody knew which one mattered.

A cable is trivially testable. Play a tone into the endpoint we write to, and
listen on every endpoint that could plausibly carry it. Either the tone comes
back or it does not, and if it does we know its exact name and level. That is a
measurement, and it takes about a second.

The tone is 997 Hz, which is deliberate: it divides evenly into no common
sample rate, so it never lands exactly on a bin boundary or on a harmonic of
mains hum, and a resampler cannot turn it into silence.

Nothing here touches the pipelines. It opens its own streams, plays its own
tone, and closes everything.
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from dataclasses import dataclass, field

import numpy as np

TONE_HZ = 997.0
TONE_SECONDS = 1.2
AMPLITUDE = 0.25

#: A tone quieter than this counts as absent. Digital silence measures about
#: -96 dB, so this sits well clear of the floor while still catching a cable
#: whose volume has been pulled right down - which is a WORKING cable, just a
#: quiet one, and calling it dead would send somebody hunting for the wrong
#: fault. Purity, below, is what actually distinguishes our tone from noise.
HEARD_DB = -75.0

#: Heard, but far quieter than we played it. Worth saying out loud: at this
#: level speech would reach a recogniser as a mumble, and the cause is almost
#: always an app or endpoint volume slider rather than the wire.
QUIET_DB = -45.0

#: How much of the captured energy has to BE the tone. A cable carrying our
#: tone scores near 1.0. This is what stops a noisy endpoint, or one carrying
#: somebody's music, from being mistaken for a working wire.
HEARD_PURITY = 0.45


def tone(rate: int, seconds: float = TONE_SECONDS, hz: float = TONE_HZ,
         amplitude: float = AMPLITUDE) -> np.ndarray:
    """Mono int16 test tone, faded in and out so it cannot click."""
    n = max(1, int(rate * seconds))
    t = np.arange(n, dtype=np.float64) / float(rate)
    wave = np.sin(2.0 * np.pi * hz * t)

    fade = int(rate * 0.01)
    if fade > 0 and n > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]

    return (wave * amplitude * 32767.0).astype(np.int16)


def tone_amplitude(samples: np.ndarray, rate: int, hz: float = TONE_HZ) -> float:
    """Peak amplitude of the `hz` component, in int16 units.

    A single-frequency fit rather than a full spectrum: we know exactly what we
    played, so there is nothing to search for. For a pure sine of amplitude A
    this returns A.
    """
    n = samples.size
    if n == 0:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    reference = np.exp(-2.0j * np.pi * hz * t / float(rate))
    return float(2.0 * abs(np.dot(samples.astype(np.float64), reference)) / n)


def db(amplitude: float) -> float:
    """int16 amplitude as dB relative to full scale."""
    return 20.0 * math.log10(max(amplitude, 1e-12) / 32768.0)


def purity(samples: np.ndarray, rate: int, hz: float = TONE_HZ) -> float:
    """How much of what we captured is the tone: 1.0 = all of it."""
    if samples.size == 0:
        return 0.0
    rms = math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))
    if rms <= 0.0:
        return 0.0
    return min(1.0, tone_amplitude(samples, rate, hz) / (rms * math.sqrt(2.0)))


def driver_of(name: str) -> str:
    """The driver in brackets: 'CABLE Output (VB-Audio Virtual Cable)' -> the
    part in the last brackets, lowercased.

    Endpoints of one driver are the only candidates worth testing, and the
    bracketed name is how Windows reports which driver an endpoint belongs to.
    """
    matches = re.findall(r"\(([^()]*)\)", name or "")
    if not matches:
        return ""
    return matches[-1].strip().lower()


# ----------------------------------------------------------------------


@dataclass
class Listener:
    """One endpoint we listened on, and what it heard."""

    name: str
    kind: str  # "recording" or "loopback"
    level_db: float = -99.0
    purity: float = 0.0
    error: str = ""

    @property
    def heard(self) -> bool:
        return (
            not self.error
            and self.level_db > HEARD_DB
            and self.purity >= HEARD_PURITY
        )

    @property
    def quiet(self) -> bool:
        """Carrying the tone, but at a level that would hurt recognition."""
        return self.heard and self.level_db < QUIET_DB


@dataclass
class WireReport:
    """One cable, tested end to end."""

    role: str          # human wording: "your English to Teams"
    render: str        # the endpoint we played into
    listeners: list[Listener] = field(default_factory=list)
    error: str = ""

    @property
    def recording(self) -> list[Listener]:
        return [l for l in self.listeners if l.kind == "recording" and l.heard]

    @property
    def loopback(self) -> list[Listener]:
        return [l for l in self.listeners if l.kind == "loopback" and l.heard]

    @property
    def ok(self) -> bool:
        """The wire carries sound somewhere we can reach."""
        return not self.error and bool(self.recording or self.loopback)

    def verdict(self) -> str:
        if self.error:
            return f"CANNOT TEST - {self.error}"
        if self.recording:
            best = max(self.recording, key=lambda l: l.level_db)
            if best.quiet:
                return (
                    f"WORKS BUT QUIET - {best.name!r} carries the tone at "
                    f"{best.level_db:.0f} dB, far below the {db(AMPLITUDE * 32767.0):.0f} dB "
                    f"we played. Check that endpoint's volume and the app's "
                    f"slider in the Windows volume mixer."
                )
            return f"OK - it comes out of {best.name!r}"
        if self.loopback:
            return (
                "PARTIAL - the tone is on the endpoint's own loopback but on no "
                "recording device, so we can hear it and Teams cannot"
            )
        return "DEAD - the tone went in and came out nowhere"


def candidates(manager, render_device):
    """Every endpoint that could carry what we play into `render_device`.

    Its own driver's recording endpoints, plus its loopback. We do NOT try to
    work out which recording endpoint is the partner of which playback one -
    for VB-CABLE it is a fixed pair, for VoiceMeeter it depends on which bus
    buttons are lit inside the mixer, and guessing that is precisely the
    mistake this module exists to stop. Test them all; the tone says which.
    """
    driver = driver_of(render_device.name)
    found = []

    for device in manager.wasapi_devices():
        if not getattr(device, "can_capture", False):
            continue
        if getattr(device, "is_loopback", False):
            continue
        if driver_of(device.name) != driver:
            continue
        found.append((device, "recording"))

    with contextlib.suppress(Exception):
        found.append((manager.loopback_for_output(render_device), "loopback"))

    return found


def probe_wire(manager, render_name: str, *, role: str,
               seconds: float = TONE_SECONDS) -> WireReport:
    """Play a tone into `render_name` and report every endpoint that heard it."""
    from app.audio import convert
    from app.audio.streams import CaptureSource, RenderSink

    report = WireReport(role=role, render=render_name)
    try:
        render_device = manager.find(render_name, capture=False)
    except Exception as exc:
        report.error = str(exc)
        return report

    listeners: list[tuple[Listener, object, list[np.ndarray]]] = []
    for device, kind in candidates(manager, render_device):
        entry = Listener(name=device.name, kind=kind)
        try:
            source = CaptureSource(manager, device)
            source.start()
        except Exception as exc:
            entry.error = str(exc)
            listeners.append((entry, None, []))
            continue
        listeners.append((entry, source, []))

    try:
        sink = RenderSink(manager, render_device)
        sink.start()
    except Exception as exc:
        report.error = f"could not play into it: {exc}"
        for _, source, _ in listeners:
            if source is not None:
                with contextlib.suppress(Exception):
                    source.stop()
        return report

    try:
        pcm = tone(sink.rate, seconds)
        if getattr(sink, "channels", 1) > 1:
            pcm = convert.to_channels(pcm, sink.channels)

        # Drain first, so nothing from before the tone is measured.
        for _, source, _ in listeners:
            if source is not None:
                with contextlib.suppress(Exception):
                    source.ring.discard()

        # Play in slices and drain the listeners as we go: their rings hold
        # 500 ms, and a tone longer than that would overwrite its own start.
        slice_ms = 50
        step = max(1, int(sink.rate * sink.channels * slice_ms / 1000))
        written = 0
        deadline = time.monotonic() + seconds + 0.6
        while time.monotonic() < deadline:
            if written < pcm.size:
                chunk = pcm[written : written + step]
                with contextlib.suppress(Exception):
                    sink.write(chunk)
                written += chunk.size
            time.sleep(slice_ms / 1000.0)
            for _, source, taken in listeners:
                if source is None:
                    continue
                with contextlib.suppress(Exception):
                    heard = source.ring.read_available()
                    channels = max(1, getattr(source, "channels", 1))
                    usable = heard.size - (heard.size % channels)
                    if usable > 0:
                        taken.append(heard[:usable])
            if written >= pcm.size and time.monotonic() > deadline - 0.4:
                break
    finally:
        with contextlib.suppress(Exception):
            sink.stop()
        for _, source, _ in listeners:
            if source is not None:
                with contextlib.suppress(Exception):
                    source.stop()

    for entry, source, taken in listeners:
        if source is None:
            report.listeners.append(entry)
            continue
        if not taken:
            entry.error = entry.error or "nothing was captured at all"
            report.listeners.append(entry)
            continue
        raw = np.concatenate(taken)
        channels = max(1, getattr(source, "channels", 1))
        mono = raw.reshape(-1, channels)[:, 0].copy() if channels > 1 else raw
        entry.level_db = db(tone_amplitude(mono, source.rate))
        entry.purity = purity(mono, source.rate)
        report.listeners.append(entry)

    report.listeners.sort(key=lambda l: -l.level_db)
    return report
