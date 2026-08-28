"""Choose the device bindings without asking.

Spec reference: §11.7 - the first run should get you to a working state, not
to a form.

WHAT CAN AND CANNOT BE GUESSED
------------------------------
Your microphone and your headphones have an obvious answer: whatever Windows
is already using. Guessing those is safe, and getting them wrong costs a
dropdown.

The virtual cables have no such default, and the reason is worth stating
plainly, because the setup window's three simultaneous complaints do not say
it: two-way mode needs TWO of them. One carries your synthesised English into
Teams; the other carries Teams' audio out where we can capture it. Point both
at the same endpoint - or at anything you can hear - and the output feeds
straight back into the input, which is a howl rather than a wrong answer.

So this fills in everything it can, picks cables when the machine has them,
and when it does not, says the one thing that is actually missing instead of
three consequences of it.
"""

from __future__ import annotations

from app.audio.devices import AudioUnavailable, DeviceManager, looks_virtual
from app.models.config import Config


def virtual_cables(manager: DeviceManager) -> list[str]:
    """Playback endpoints that are virtual cables, in a stable order."""
    seen: dict[str, None] = {}
    for device in manager.wasapi_devices():
        if device.can_render and looks_virtual(device.name):
            seen.setdefault(device.name, None)
    return sorted(seen)


def autoconfigure(
    config: Config,
    manager: DeviceManager,
    *,
    save: bool = True,
    mode: str | None = None,
) -> tuple[list[str], list[str]]:
    """Fill in every unset device role. Returns (notes, problems).

    Only blanks are filled: a choice already made is never overwritten, so
    this can run on every start without fighting the settings window.
    """
    notes: list[str] = []
    problems: list[str] = []
    devices = config.devices
    changed = False
    video = (mode if mode is not None else config.mode) == "video"

    # Video mode needs one cable to listen to and somewhere to speak. It does
    # not need a microphone, and it must not open one: nothing is being sent
    # to anybody, so a live capture of the room would exist for no reason.
    if not devices.microphone and not video:
        try:
            devices.microphone = manager.default_input().name
            notes.append(f"Microphone: {devices.microphone}")
            changed = True
        except AudioUnavailable as exc:
            problems.append(f"No default microphone ({exc})")

    if not devices.headphones:
        try:
            devices.headphones = manager.default_output().name
            notes.append(f"Headphones: {devices.headphones}")
            changed = True
        except AudioUnavailable as exc:
            problems.append(f"No default playback device ({exc})")

    # Never offer something audible as a cable: that is the feedback loop.
    cables = [c for c in virtual_cables(manager) if c != devices.headphones]

    # A stored value is respected - but only where it could possibly work.
    # These two roles MUST be virtual cables. A saved speaker in one of them
    # is not a preference somebody expressed; it is the leftover of a first
    # run where every dropdown defaulted to the same endpoint, and it can
    # only ever produce the feedback-loop refusal (§5.6 F1/F2). Treating it
    # as sacred means the application explains the same failure for ever
    # while a correct cable sits unused in the list.
    for role, label in (("virtual_mic", "Translator microphone"),
                        ("teams_speaker", "Teams speaker")):
        current = getattr(devices, role)
        if not current:
            continue
        if looks_virtual(current) and current != devices.headphones:
            continue
        setattr(devices, role, "")
        changed = True
        notes.append(
            f"{label}: cleared {current!r} - that is a device you can hear, "
            "and this role has to be a virtual cable"
        )

    # One cable in both roles is the same failure as a speaker in one of
    # them: the English we synthesise is fed straight back into the incoming
    # pipeline. Clearing the second is what makes the "needs TWO" check below
    # fire, instead of silently handing out a binding that cannot work.
    if devices.virtual_mic and devices.virtual_mic == devices.teams_speaker:
        notes.append(
            f"Teams speaker: cleared {devices.teams_speaker!r} - both cables "
            "were the same endpoint"
        )
        devices.teams_speaker = ""
        changed = True

    if not devices.virtual_mic and cables:
        devices.virtual_mic = cables[0]
        notes.append(f"Translator microphone: {devices.virtual_mic}")
        changed = True

    spare = [c for c in cables if c != devices.virtual_mic]
    if not devices.teams_speaker and spare:
        devices.teams_speaker = spare[0]
        notes.append(f"Teams speaker: {devices.teams_speaker}")
        changed = True

    # Checked again after the assignments, not only before them: a repair can
    # hand the one cable on the machine to a role whose partner already holds
    # it. Leave the second role empty rather than saving a binding that the
    # feedback check would refuse anyway - an empty dropdown says "missing",
    # a duplicated one says "misconfigured", and missing is the truth.
    if devices.virtual_mic and devices.virtual_mic == devices.teams_speaker:
        devices.teams_speaker = ""
        changed = True

    if video:
        # The only requirement is a cable to capture. Prefer whichever role
        # already holds one so a machine set up for meetings keeps working.
        source = devices.teams_speaker or devices.virtual_mic
        if not source and cables:
            source = cables[0]
        if source:
            if devices.teams_speaker != source:
                devices.teams_speaker = source
                notes.append(f"Source: {source}")
                changed = True
        else:
            problems.append(
                "Video mode needs one virtual cable to listen to, and this "
                "machine has none.\n"
                "      Install VB-CABLE from vb-audio.com, reboot, then point "
                "the app you\n"
                "      are watching at it: Settings > System > Sound > Volume "
                "mixer."
            )
        if changed and save:
            if not problems:
                config.setup_complete = True
            config.save()
        return notes, problems

    if not devices.virtual_mic or not devices.teams_speaker:
        found = ", ".join(cables) if cables else "none"
        problems.append(
            f"Two-way mode needs TWO virtual cables and this machine has "
            f"{len(cables)} ({found}).\n"
            "      One carries your English into Teams; the other carries "
            "Teams' audio\n"
            "      out where we can capture it. They cannot be the same "
            "endpoint.\n\n"
            "      Either install a second cable - VB-CABLE A+B from "
            "vb-audio.com, or\n"
            "      VoiceMeeter, which provides several - or use one-way mode, "
            "which\n"
            "      needs only the cable you already have:\n\n"
            "          python listen.py"
        )

    if changed and save:
        if not problems:
            config.setup_complete = True
        config.save()
    return notes, problems
