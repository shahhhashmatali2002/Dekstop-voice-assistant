"""Everything that has to be true before a call, checked before the call.

WHY THIS IS SEPARATE FROM run.py --check
----------------------------------------
`--check` answers "is this installed". This answers "will a two-way call
work", which is a different and much narrower question, and it is worth
asking on its own because the failures it catches are the expensive kind:
they surface thirty seconds into a conversation with somebody else on the
line, and the recovery is to hang up.

Two-way is where the requirements multiply. One-way needs a cable and a pair
of headphones. A call needs two cables that are not the same cable, a
microphone, somewhere audible to play the Urdu, keys for both directions -
and, if the Urdu is going to come out of speakers rather than headphones, a
decision about what to do when the microphone hears it.

Nothing here opens a stream or contacts a provider. It is meant to be
instant, and to be runnable while a meeting invite is still on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.audio.devices import (
    AudioUnavailable,
    DeviceManager,
    feedback_loop_problems,
    feedback_loop_warnings,
    looks_virtual,
)
from app.models.config import Config, missing_secrets


@dataclass
class Report:
    """What is ready, what is not, and what to do about it."""

    ready: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = []
        for item in self.ready:
            lines.append(f"  ok       {item}")
        for item in self.warnings:
            lines.append(f"  warning  {item}")
        for item in self.problems:
            lines.append(f"  BLOCKED  {item}")
        if self.instructions:
            lines.append("")
            lines.append("  In Teams:")
            for step in self.instructions:
                lines.append(f"    - {step}")
        return "\n".join(lines)


def _resolve(manager, name: str):
    """(device, why-not). The reason matters more than the failure does.

    The first version of this swallowed the exception and reported "not present
    on this machine now", which is one of three quite different things: the
    endpoint is missing, or it is disabled, or the saved name matches more than
    one endpoint and resolving it is ambiguous. Telling them apart from the
    outside is impossible, so the reason is carried out with the failure.
    """
    try:
        return manager.find(name, capture=False), ""
    except Exception as exc:
        return None, str(exc)


def _playback_endpoints(manager) -> list[str]:
    try:
        return [d.name for d in manager.wasapi_devices() if d.can_render]
    except Exception:
        return []


def meeting_preflight(config: Config, manager: DeviceManager | None = None) -> Report:
    """Is this machine ready to translate a Teams call?"""
    report = Report()
    devices = config.devices

    # ---- keys. Both directions, and they are different providers.
    missing = missing_secrets(config)
    if missing:
        report.problems.append(f"API keys missing: {', '.join(missing)}")
    else:
        report.ready.append("API keys for both directions")

    # ---- the four device roles
    for label, value in (
        ("Microphone", devices.microphone),
        ("Urdu output", devices.headphones),
        ("Cable carrying your English to Teams", devices.virtual_mic),
        ("Cable carrying Teams' audio to us", devices.teams_speaker),
    ):
        if value:
            report.ready.append(f"{label}: {value}")
        else:
            report.problems.append(f"{label} is not set")

    if devices.virtual_mic and devices.virtual_mic == devices.teams_speaker:
        report.warnings.extend(
            feedback_loop_warnings(None, None, _Named(devices.virtual_mic),
                                   _Named(devices.teams_speaker))
        )

    if not devices.virtual_mic:
        cables = [
            name for name in _playback_endpoints(manager) if looks_virtual(name)
        ] if manager is not None else []
        if cables:
            report.problems.append(
                "Cable carrying your English to Teams is not set, and this "
                "machine has cables it could use:\n        "
                + "\n        ".join(cables)
                + "\n      Open the app once (python run.py) - it assigns "
                  "them on startup - or pick one in Settings > Devices."
            )

    for role, value in (("Cable to Teams", devices.virtual_mic),
                        ("Cable from Teams", devices.teams_speaker)):
        if value and not looks_virtual(value):
            report.problems.append(
                f"{role} is {value!r}, which is not a virtual cable. Whatever "
                f"we render into it would be audible in the room and captured "
                f"straight back."
            )

    # ---- the acoustic loop, which is about the room rather than the routing
    if devices.headphones and not _sounds_like_headphones(devices.headphones):
        report.warnings.append(
            f"The Urdu plays to {devices.headphones!r}, which sounds like "
            f"speakers rather than headphones. Your microphone will hear it, "
            f"so it would be translated back to English and sent to the "
            f"client. Half-duplex is on by default for exactly this, and it "
            f"means your own speech is ignored while the Urdu is playing - "
            f"headphones avoid the compromise entirely."
        )

    # ---- do the endpoints actually exist right now
    if manager is not None:
        for label, name in (
            ("Microphone", devices.microphone),
            ("Urdu output", devices.headphones),
            ("Cable to Teams", devices.virtual_mic),
            ("Cable from Teams", devices.teams_speaker),
        ):
            if not name:
                continue
            if label == "Microphone":
                continue
            found, why = _resolve(manager, name)
            if found is not None:
                continue
            detail = f"{label} {name!r} could not be resolved: {why}"
            if "ambiguous" not in why:
                detail += (
                    "\n      If it is a cable, check it is enabled: mmsys.cpl "
                    "> Playback > right-click > Show Disabled Devices."
                )
            playback = _playback_endpoints(manager)
            if playback:
                detail += "\n      Playback endpoints seen right now:\n        " + \
                    "\n        ".join(playback)
            report.problems.append(detail)

        report.problems.extend(feedback_loop_problems(
            _resolve(manager, devices.microphone)[0],
            _resolve(manager, devices.headphones)[0],
            _resolve(manager, devices.virtual_mic)[0],
            _resolve(manager, devices.teams_speaker)[0],
        ))

    # ---- what the user still has to do by hand, if anything
    # Named explicitly, always. Automatic routing moves the Windows DEFAULT,
    # which does nothing for a Teams already pointed at a real microphone -
    # and Teams remembers a device once chosen. "Leave it on Default" was
    # advice that read as reassuring and left the client listening to the
    # user's raw Urdu.
    partner = _partner(devices.virtual_mic, manager)
    report.instructions.append(
        f"Settings > Devices > Microphone: {partner}"
        + ("  <- this is what makes the client hear English" if partner else "")
    )
    report.instructions.append(
        f"Settings > Devices > Speaker: "
        f"{devices.teams_speaker or 'the second cable'}"
    )
    if getattr(devices, "auto_route", False):
        report.ready.append(
            "Windows' default devices are moved to the cables while it runs, "
            "and put back on Stop"
        )
        report.instructions.append(
            "'Default' also works for both, but only if Teams has never been "
            "pointed at a specific device - it remembers"
        )
    report.instructions.append(
        "make a test call to yourself from your phone before using it with "
        "anybody else"
    )
    return report


class _Named:
    """The warning helpers take device objects; here there are only names."""

    def __init__(self, name: str) -> None:
        self.name = name


def _partner(cable: str, manager=None) -> str:
    """The recording endpoint Teams must select, by name.

    Looked up on the machine rather than guessed by swapping "Input" for
    "Output" in the name - which produces nothing at all for endpoints like
    "Speakers (2- VB-Audio Virtual Cable)", whose recording half is called
    "CABLE Output (2- VB-Audio Virtual Cable)".
    """
    if not cable:
        return "the recording half of the first cable"
    if manager is not None:
        from app.audio.devices import cable_family

        wanted = cable_family(cable)
        try:
            for device in manager.wasapi_devices():
                if (
                    getattr(device, "can_capture", False)
                    and not getattr(device, "is_loopback", False)
                    and cable_family(device.name) == wanted
                ):
                    return device.name
        except Exception:
            pass
    return cable.replace("Input", "Output")


def _sounds_like_headphones(name: str) -> bool:
    lowered = (name or "").lower()
    return any(
        hint in lowered
        for hint in ("headphone", "headset", "earphone", "earbud", "buds",
                     "airpod", "hands-free")
    )
