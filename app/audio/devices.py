"""Windows audio endpoint discovery.

Spec reference: §5.1, §5.8.

WINDOWS ONLY. Every function here needs PyAudioWPatch, which is a PortAudio
fork that exposes WASAPI loopback capture - something mainline sounddevice
still does not. Nothing in this module may be imported by the pipeline or the
provider layer (§10.4).

Known limitation, deliberately accepted for Phase 0
---------------------------------------------------
The spec says to pin devices by endpoint ID, never by friendly name, because
names collide and PortAudio indices are not stable across device changes.
PortAudio does not expose the Windows endpoint ID, so `DeviceRef` stores the
name plus host API and re-resolves at open time. That is good enough to prove
the audio spine and no further. Phase 6 replaces the resolver with pycaw,
which does expose the endpoint ID; `DeviceRef.endpoint_id` is already in the
dataclass so that change does not ripple outward.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterator

if sys.platform != "win32":  # pragma: no cover - platform guard
    _IMPORT_ERROR: str | None = (
        "va.audio.devices requires Windows; this module cannot run on "
        f"{sys.platform}"
    )
    pyaudio = None
else:  # pragma: no cover - exercised only on Windows
    try:
        import pyaudiowpatch as pyaudio

        _IMPORT_ERROR = None
    except ImportError:
        pyaudio = None
        _IMPORT_ERROR = (
            "PyAudioWPatch is not installed. Run: pip install PyAudioWPatch"
        )


class AudioUnavailable(RuntimeError):
    """Raised when the Windows audio stack cannot be reached."""


def _require() -> None:
    if _IMPORT_ERROR is not None:
        raise AudioUnavailable(_IMPORT_ERROR)


# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceRef:
    """A resolvable reference to a Windows audio endpoint.

    `index` is a cache, not an identity: it is re-resolved from `name` on every
    open, because PortAudio renumbers devices whenever the device set changes.
    """

    name: str
    index: int
    max_input_channels: int
    max_output_channels: int
    default_rate: int
    host_api: str
    is_loopback: bool = False
    endpoint_id: str | None = field(default=None)  # populated in Phase 6

    @property
    def can_capture(self) -> bool:
        return self.max_input_channels > 0

    @property
    def can_render(self) -> bool:
        return self.max_output_channels > 0

    def describe(self) -> str:
        kind = []
        if self.can_capture:
            kind.append(f"in:{self.max_input_channels}")
        if self.can_render:
            kind.append(f"out:{self.max_output_channels}")
        tag = " [loopback]" if self.is_loopback else ""
        return (
            f"[{self.index:>3}] {self.name}{tag}\n"
            f"      {self.host_api}  {', '.join(kind)}  "
            f"{self.default_rate} Hz"
        )


# ----------------------------------------------------------------------


class DeviceManager:
    """Owns the PortAudio handle and enumerates endpoints.

    Use as a context manager so the handle is always released - a leaked
    PortAudio instance keeps device streams alive and produces confusing
    "device in use" errors on the next run.
    """

    def __init__(self) -> None:
        _require()
        self._pa = pyaudio.PyAudio()

    def __enter__(self) -> "DeviceManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    @property
    def pa(self):
        """The raw PyAudio handle, for stream construction."""
        if self._pa is None:
            raise AudioUnavailable("DeviceManager is closed")
        return self._pa

    # ------------------------------------------------------------------

    def _wasapi_info(self) -> dict:
        try:
            return self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as exc:  # pragma: no cover - Windows-only path
            raise AudioUnavailable(
                "WASAPI host API not available on this machine"
            ) from exc

    def _to_ref(self, info: dict) -> DeviceRef:
        try:
            host = self.pa.get_host_api_info_by_index(info["hostApi"])["name"]
        except Exception:  # pragma: no cover - defensive
            host = f"hostApi {info['hostApi']}"
        return DeviceRef(
            name=info["name"],
            index=int(info["index"]),
            max_input_channels=int(info.get("maxInputChannels", 0)),
            max_output_channels=int(info.get("maxOutputChannels", 0)),
            default_rate=int(info.get("defaultSampleRate", 48_000)),
            host_api=host,
            is_loopback=bool(info.get("isLoopbackDevice", False)),
        )

    def all_devices(self) -> list[DeviceRef]:
        """Every endpoint PortAudio can see, across all host APIs."""
        return [
            self._to_ref(self.pa.get_device_info_by_index(i))
            for i in range(self.pa.get_device_count())
        ]

    def wasapi_devices(self) -> list[DeviceRef]:
        """WASAPI endpoints only - what we actually target."""
        wasapi = self._wasapi_info()
        return [d for d in self.all_devices() if d.host_api == wasapi["name"]]

    def loopback_devices(self) -> list[DeviceRef]:
        """Loopback capture endpoints.

        PyAudioWPatch exposes one of these per render device: opening it as an
        input captures whatever that device is playing. This is Option A in
        §5.2 and the whole basis of the incoming pipeline in v1.
        """
        _require()
        out: list[DeviceRef] = []
        for info in self._loopback_generator():
            out.append(self._to_ref(info))
        return out

    def _loopback_generator(self) -> Iterator[dict]:  # pragma: no cover
        try:
            yield from self.pa.get_loopback_device_info_generator()
        except Exception as exc:
            raise AudioUnavailable(
                "could not enumerate loopback devices; is PyAudioWPatch "
                "installed (not plain PyAudio)?"
            ) from exc

    # ------------------------------------------------------------------

    def default_output(self) -> DeviceRef:
        info = self._wasapi_info()
        return self._to_ref(
            self.pa.get_device_info_by_index(info["defaultOutputDevice"])
        )

    def default_input(self) -> DeviceRef:
        info = self._wasapi_info()
        return self._to_ref(
            self.pa.get_device_info_by_index(info["defaultInputDevice"])
        )

    def loopback_for_output(self, device: DeviceRef) -> DeviceRef:
        """The loopback capture endpoint that mirrors `device`.

        PyAudioWPatch names loopback endpoints after their render device, so
        the match is by name prefix.
        """
        for lb in self.loopback_devices():
            if device.name in lb.name:
                return lb
        raise AudioUnavailable(
            f"no loopback endpoint found for output device {device.name!r}. "
            "Loopback requires PyAudioWPatch and a WASAPI render device."
        )

    # ------------------------------------------------------------------

    def find(
        self,
        needle: str,
        *,
        capture: bool | None = None,
        loopback: bool | None = None,
    ) -> DeviceRef:
        """Resolve one endpoint by case-insensitive substring of its name.

        Raises if the match is ambiguous rather than guessing - silently
        picking the wrong microphone is a confusing failure to debug.
        """
        needle_l = needle.lower()
        hits = [d for d in self.wasapi_devices() if needle_l in d.name.lower()]
        if capture is not None:
            hits = [d for d in hits if d.can_capture == capture]
        if loopback is not None:
            hits = [d for d in hits if d.is_loopback == loopback]

        if not hits:
            raise AudioUnavailable(f"no WASAPI endpoint matching {needle!r}")
        if len(hits) > 1:
            names = "\n  ".join(h.describe().splitlines()[0] for h in hits)
            raise AudioUnavailable(
                f"{needle!r} is ambiguous, {len(hits)} endpoints match:\n  {names}"
            )
        return hits[0]

    def resolve(self, ref: DeviceRef) -> DeviceRef:
        """Re-resolve a stored reference to a current index.

        Call this immediately before opening a stream. Indices move whenever a
        device is added or removed - a Bluetooth headset waking up is enough.
        """
        for d in self.all_devices():
            if d.name == ref.name and d.host_api == ref.host_api:
                return d
        raise AudioUnavailable(
            f"endpoint {ref.name!r} is no longer present "
            f"(was index {ref.index})"
        )


# ----------------------------------------------------------------------


#: Substrings that identify a virtual audio device, whoever made it.
VIRTUAL_HINTS = ("cable", "voicemeeter", "vb-audio", "virtual audio")


def looks_virtual(name: str) -> bool:
    """Is this endpoint a virtual cable rather than something you can hear?"""
    lowered = (name or "").lower()
    return any(hint in lowered for hint in VIRTUAL_HINTS)


def format_device_table(devices: list[DeviceRef]) -> str:
    """Human-readable listing, used by the Phase 0 scripts."""
    if not devices:
        return "  (none)"
    return "\n".join(d.describe() for d in devices)


def feedback_loop_problems(mic, phones, vmic, teams) -> list[str]:
    """Refuse any binding that feeds a device back into itself.

    Spec reference: §5.6 F1 and F2. This is the check that has to happen
    BEFORE a stream opens, because the failure mode is not a wrong result
    — it is a howl out of the speakers that gets louder until somebody
    pulls the plug.

    Two loops are detectable from the device names alone:

        virtual mic == headphones   your microphone is rendered into
                                    something you can hear, and the
                                    microphone hears it back (F1)

        Teams speaker == headphones we loopback-capture the very device we
                                    are rendering into, so our own output
                                    re-enters the capture (F2), amplifying
                                    every pass

    A third is a configuration error rather than a loop: pointing Teams'
    speaker and our virtual microphone at one device sends the English we
    synthesise straight back into the incoming pipeline.
    """
    found: list[str] = []
    name = lambda d: getattr(d, "name", None)  # noqa: E731

    if vmic is not None and phones is not None and name(vmic) == name(phones):
        found.append(
            f"Feedback loop: the translator microphone and your headphones "
            f"are both {name(vmic)!r}. Your own voice would be played out "
            "of it and picked straight back up. These must be different "
            "devices — the translator microphone has to be a virtual audio "
            "cable, not something you can hear."
        )

    if teams is not None and phones is not None and name(teams) == name(phones):
        found.append(
            f"Feedback loop: Teams' speaker and your headphones are both "
            f"{name(teams)!r}. We capture whatever that device plays and "
            "render into it, so the sound would re-enter its own capture "
            "and get louder every pass. Teams must output to a second "
            "virtual cable."
        )

    if vmic is not None and teams is not None and name(vmic) == name(teams):
        found.append(
            f"Both virtual cables are set to {name(vmic)!r}. The English we "
            "generate would be fed straight back into the incoming "
            "pipeline. Two separate cables are needed."
        )

    if mic is not None and vmic is not None and name(mic) == name(vmic):
        found.append(
            f"Your microphone and the translator microphone are both "
            f"{name(mic)!r}. They cannot be the same endpoint."
        )
    return found
