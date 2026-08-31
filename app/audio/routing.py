"""Doing the routing, so nobody has to open the volume mixer.

THE PROBLEM THIS REMOVES
------------------------
Everything about this application is automatic except one step, and that step
is a settings page in Windows: send the browser's audio to the virtual cable,
then put it back afterwards. It is the single most common reason a first run
produces silence, it has to be redone whenever Windows resets it, and it is
the part of the instructions people give up on.

WHAT WINDOWS ACTUALLY ALLOWS
----------------------------
There is no supported API for setting ONE application's output device - that
is why the volume mixer is the documented answer. There IS an undocumented COM
interface, IPolicyConfig, for setting the SYSTEM DEFAULT playback device, and
it is what every third-party audio switcher on Windows has used for fifteen
years. So the trade this makes:

    while translating   the default output is the cable, so everything that
                        follows the default - the browser, Teams, a media
                        player - is captured, and the Urdu is played to the
                        real speakers, which this application opens BY NAME
                        rather than as the default, so it is never affected

    when it stops       the default goes back to exactly what it was

The consequence worth knowing: while it runs, other sounds (notifications,
another app's audio) also go down the cable, which means they are translated
along with everything else rather than played. That is the same behaviour as
routing the browser by hand, applied more widely.

RESTORING IS NOT OPTIONAL
-------------------------
A crash that leaves the machine's default output pointing at a cable with
nothing on the other end is indistinguishable, to the user, from broken sound
hardware - and the fix is not obvious. So the restore runs from stop, from
shutdown, and from an atexit hook, it is idempotent, and it never raises.
"""

from __future__ import annotations

import atexit
import contextlib
import threading

# The undocumented one. Same CLSID every switcher on Windows uses.
POLICY_CONFIG_CLSID = "{870AF99C-171D-4F9E-AF0D-E63DF40C2BC9}"

#: (interface id, how many methods come before SetDefaultEndpoint).
#:
#: The count is the whole game. A COM interface is a table of function
#: pointers and nothing checks that you counted correctly: call slot 13 when
#: the function you want is at slot 11 and Windows runs whatever is there,
#: with your arguments, and returns
#:
#:     (-2147467261, 'Invalid pointer', (None, None, None, 0, None))
#:
#: which is E_POINTER from SetPropertyValue being handed a device id where it
#: expected a PROPERTYKEY. That is what the first version of this did. The
#: modern interface has ten methods before it - GetMixFormat, GetDeviceFormat,
#: ResetDeviceFormat, SetDeviceFormat, Get/SetProcessingPeriod,
#: Get/SetShareMode, Get/SetPropertyValue - and the Vista-era one has twelve.
#: Both are tried, because which one a machine answers to is not something to
#: guess at from a version number.
POLICY_CONFIG_LAYOUTS = (
    ("{F8679F50-850A-41CF-9C72-430F290290C8}", 10),   # Windows 7 and later
    ("{568b9108-44bf-40b4-9006-86afe5b5a620}", 12),   # Vista
)

#: Which of Windows' three roles to move. All of them, because an application
#: picks whichever it likes and the point is that it should not matter.
ROLES = (0, 1, 2)  # eConsole, eMultimedia, eCommunications


class RoutingUnavailable(RuntimeError):
    """This machine will not let us change the default device."""


#: Windows' two data flows. A call needs both: the client's audio comes out of
#: Teams into a cable we capture, and our English goes into a cable Teams uses
#: as its microphone.
RENDER, CAPTURE = 0, 1
DEVICE_STATE_ACTIVE = 1


def _audio_utilities():
    try:
        from pycaw.pycaw import AudioUtilities

        return AudioUtilities
    except ImportError as exc:
        raise RoutingUnavailable(
            "automatic routing needs pycaw:\n"
            "    .venv\\Scripts\\python -m pip install pycaw comtypes"
        ) from exc


def endpoints(capture: bool = False) -> list[tuple[str, str]]:
    """(friendly name, MMDevice id) for every active endpoint of one flow.

    Enumerated per flow rather than filtered afterwards, because a name is not
    enough to tell a render endpoint from a capture one - "CABLE Input" is a
    playback device and "CABLE Output" is a recording device, and a virtual
    cable is precisely a pair with almost the same name.
    """
    utilities = _audio_utilities()
    found: list[tuple[str, str]] = []
    try:
        enumerator = utilities.GetDeviceEnumerator()
        collection = enumerator.EnumAudioEndpoints(
            CAPTURE if capture else RENDER, DEVICE_STATE_ACTIVE
        )
        for index in range(collection.GetCount()):
            try:
                device = utilities.CreateDevice(collection.Item(index))
            except Exception:
                continue
            if device is not None and getattr(device, "id", None):
                found.append((getattr(device, "FriendlyName", "") or "", device.id))
    except Exception:
        # Older pycaw, or an enumerator that will not answer: fall back to the
        # flat list, which is render-only but better than nothing.
        if capture:
            raise RoutingUnavailable("could not enumerate recording devices")
        for device in utilities.GetAllDevices():
            if getattr(device, "id", None):
                found.append((getattr(device, "FriendlyName", "") or "", device.id))
    return found


def recording_partner(render_name: str) -> str:
    """The capture endpoint on the same wire as a render endpoint.

    Not `name.replace("Input", "Output")`, which is what this was first, and
    which silently does nothing at all for endpoints that do not contain the
    word - "CABLE In 16 Ch (2- VB-Audio Virtual Cable)" among them. The default
    microphone was then pointed at a playback device's name, the lookup failed,
    and the user was told automatic routing was unavailable.

    Matched by wire instead, using the driver in brackets, which is the only
    part of the two names that is reliably shared.
    """
    from app.audio.devices import cable_family

    wanted = cable_family(render_name)
    for friendly, identifier in endpoints(capture=True):
        if cable_family(friendly) == wanted:
            return friendly
    raise RoutingUnavailable(
        f"no recording device belongs to the same cable as {render_name!r}"
    )


def endpoint_id(name: str, *, capture: bool = False) -> str:
    """The MMDevice id of an endpoint whose name contains `name`.

    PortAudio, which the rest of the audio layer uses, has no idea what an
    MMDevice id is - it deals in indices and friendly names - so this is the
    one place the two worlds are matched up, by name, and the ambiguity that
    implies is why the match is reported rather than guessed at.
    """
    needle = (name or "").lower()
    if not needle:
        raise RoutingUnavailable("no device name to look for")
    for friendly, identifier in endpoints(capture=capture):
        if needle in friendly.lower():
            return identifier
    kind = "recording" if capture else "playback"
    raise RoutingUnavailable(f"no {kind} device matching {name!r}")


def _interface(iid: str, reserved: int):
    """The policy interface, with SetDefaultEndpoint in the given slot.

    The methods before it are declared as opaque placeholders rather than
    described: they only have to occupy their slots, and writing out signatures
    for functions we never call would be a way to crash the process for no
    benefit.
    """
    import comtypes
    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown
    from ctypes import c_int
    from ctypes.wintypes import LPCWSTR

    class IPolicyConfig(IUnknown):
        _iid_ = GUID(iid)
        _methods_ = [
            COMMETHOD([], HRESULT, f"Reserved{n}") for n in range(reserved)
        ] + [
            COMMETHOD(
                [], HRESULT, "SetDefaultEndpoint",
                (["in"], LPCWSTR, "deviceId"),
                (["in"], c_int, "role"),
            ),
        ]

    return comtypes.CoCreateInstance(
        comtypes.GUID(POLICY_CONFIG_CLSID), IPolicyConfig, comtypes.CLSCTX_ALL
    )


def set_default_endpoint(device_id: str) -> None:
    """Point every role at `device_id`. Raises RoutingUnavailable.

    One function for both flows: SetDefaultEndpoint takes an endpoint id and
    does not care which direction it belongs to, which is exactly why a
    microphone can be routed the same way a speaker can.

    Tries each known interface layout and keeps the first that actually works -
    "works" meaning the call returned success, not merely that the interface
    could be created, because the wrong layout creates perfectly happily and
    then fails on the call.
    """
    try:
        import comtypes  # noqa: F401
    except ImportError as exc:
        raise RoutingUnavailable(
            "automatic routing needs comtypes:\n"
            "    .venv\\Scripts\\python -m pip install comtypes pycaw"
        ) from exc
    import comtypes

    try:
        comtypes.CoInitialize()
    except Exception:
        pass  # already initialised on this thread, which is fine

    problems = []
    for iid, reserved in POLICY_CONFIG_LAYOUTS:
        try:
            config = _interface(iid, reserved)
        except Exception as exc:
            problems.append(f"{iid}: {exc}")
            continue
        try:
            for role in ROLES:
                config.SetDefaultEndpoint(device_id, role)
            return
        except Exception as exc:
            problems.append(f"{iid}: {exc}")

    raise RoutingUnavailable(
        "could not set the default output. " + "; ".join(problems[:2])
    )


#: Kept because the whole application referred to it before there was a second
#: flow to route.
set_default_output = set_default_endpoint


def default_endpoint_id(capture: bool = False) -> str:
    """The current default endpoint's id for one flow, to put back later.

    pycaw's `GetSpeakers` returns different things in different versions - in
    some it is the raw IMMDevice COM pointer, which has a GetId() method, and
    in others a Python `AudioDevice` wrapper, which has an `id` attribute and
    no such method. Asking for the wrong one raises

        'AudioDevice' object has no attribute 'GetId'

    which is how this was found. Both shapes are accepted rather than pinning
    a version, because the failure mode of getting it wrong is that the user
    is left routing by hand.
    """
    AudioUtilities = _audio_utilities()

    try:
        enumerator = AudioUtilities.GetDeviceEnumerator()
        current = enumerator.GetDefaultAudioEndpoint(
            CAPTURE if capture else RENDER, 0
        )
    except Exception:
        if capture:
            raise RoutingUnavailable("could not read the default microphone")
        try:
            current = AudioUtilities.GetSpeakers()
        except Exception as exc:
            raise RoutingUnavailable(f"could not read the default output: {exc}")

    speakers = current
    getter = getattr(speakers, "GetId", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            pass
    for attribute in ("id", "Id", "device_id"):
        found = getattr(speakers, attribute, None)
        if found:
            return str(found)

    raise RoutingUnavailable(
        f"could not read the id of the current default output "
        f"({type(speakers).__name__} has neither GetId() nor .id)"
    )


class DefaultOutputRouter:
    """Moves Windows' defaults where we need them, and always puts them back.

    Both flows, because a call needs both. Teams left on "Default" for its
    speaker and its microphone then needs no configuring at all: the default
    output becomes the cable we capture, the default input becomes the cable we
    speak into, and on Stop both go back to whatever the user had.

    Written as an object with explicit `route`/`restore` rather than a context
    manager because its lifetime is the translating state of the engine, which
    starts and stops from user actions on another thread - not a block of code.
    """

    def __init__(self, *, setter=None, reader=None, resolver=None) -> None:
        # Injectable so the behaviour - especially the restore - can be tested
        # on a machine with no Windows audio stack at all.
        self._set = setter or set_default_endpoint
        self._read = reader or default_endpoint_id
        self._resolve = resolver or endpoint_id
        self._previous: dict[bool, str] = {}
        self._lock = threading.Lock()
        self._registered = False

    @property
    def routed(self) -> bool:
        return bool(self._previous)

    def route(self, device_name: str, *, capture: bool = False) -> str:
        """Send one flow to `device_name`. Returns what it moved away from.

        Idempotent per flow: routing twice does not overwrite the remembered
        original with our own cable, which would make the restore a no-op and
        leave the machine with no working device after a stop.
        """
        with self._lock:
            target = self._resolve(device_name, capture=capture)
            if capture not in self._previous:
                self._previous[capture] = self._read(capture=capture)
                if not self._registered:
                    atexit.register(self.restore)
                    self._registered = True
            self._set(target)
            return self._previous[capture]

    def restore(self) -> None:
        """Put every flow we touched back. Safe to call from anywhere, twice."""
        with self._lock:
            previous, self._previous = self._previous, {}
        for identifier in previous.values():
            if not identifier:
                continue
            with contextlib.suppress(Exception):
                self._set(identifier)
