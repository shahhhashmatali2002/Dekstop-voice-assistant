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


def endpoint_id(name: str) -> str:
    """The MMDevice id of a render endpoint whose name contains `name`.

    PortAudio, which the rest of the audio layer uses, has no idea what an
    MMDevice id is - it deals in indices and friendly names - so this is the
    one place the two worlds are matched up, by name, and the ambiguity that
    implies is why the match is reported rather than guessed at.
    """
    try:
        from pycaw.pycaw import AudioUtilities
    except ImportError as exc:
        raise RoutingUnavailable(
            "automatic routing needs pycaw:\n"
            "    .venv\\Scripts\\python -m pip install pycaw comtypes"
        ) from exc

    needle = (name or "").lower()
    hits = []
    for device in AudioUtilities.GetAllDevices():
        friendly = (getattr(device, "FriendlyName", "") or "")
        if needle and needle in friendly.lower() and getattr(device, "id", None):
            hits.append((friendly, device.id))
    if not hits:
        raise RoutingUnavailable(f"no playback device matching {name!r}")
    return hits[0][1]


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


def set_default_output(device_id: str) -> None:
    """Point every role at `device_id`. Raises RoutingUnavailable.

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


def default_output_id() -> str:
    """The current default render endpoint's id, for putting back later.

    pycaw's `GetSpeakers` returns different things in different versions - in
    some it is the raw IMMDevice COM pointer, which has a GetId() method, and
    in others a Python `AudioDevice` wrapper, which has an `id` attribute and
    no such method. Asking for the wrong one raises

        'AudioDevice' object has no attribute 'GetId'

    which is how this was found. Both shapes are accepted rather than pinning
    a version, because the failure mode of getting it wrong is that the user
    is left routing by hand.
    """
    try:
        from pycaw.pycaw import AudioUtilities
    except ImportError as exc:
        raise RoutingUnavailable(
            "automatic routing needs pycaw:\n"
            "    .venv\\Scripts\\python -m pip install pycaw comtypes"
        ) from exc

    try:
        speakers = AudioUtilities.GetSpeakers()
    except Exception as exc:
        raise RoutingUnavailable(f"could not read the default output: {exc}")

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
    """Moves the default output to the cable, and always puts it back.

    Written as an object with explicit `route`/`restore` rather than a context
    manager because its lifetime is the translating state of the engine, which
    starts and stops from user actions on another thread - not a block of code.
    """

    def __init__(self, *, setter=None, reader=None, resolver=None) -> None:
        # Injectable so the behaviour - especially the restore - can be tested
        # on a machine with no Windows audio stack at all.
        self._set = setter or set_default_output
        self._read = reader or default_output_id
        self._resolve = resolver or endpoint_id
        self._previous: str | None = None
        self._lock = threading.Lock()
        self._registered = False

    @property
    def routed(self) -> bool:
        return self._previous is not None

    def route(self, device_name: str) -> str:
        """Send everything to `device_name`. Returns what it moved away from.

        Idempotent: routing twice does not overwrite the remembered original
        with the cable itself, which would make the restore a no-op and leave
        the machine silent after a stop.
        """
        with self._lock:
            target = self._resolve(device_name)
            if self._previous is None:
                self._previous = self._read()
                if not self._registered:
                    atexit.register(self.restore)
                    self._registered = True
            self._set(target)
            return self._previous

    def restore(self) -> None:
        """Put the default back. Safe to call at any time, from anywhere."""
        with self._lock:
            previous, self._previous = self._previous, None
        if not previous:
            return
        with contextlib.suppress(Exception):
            self._set(previous)
