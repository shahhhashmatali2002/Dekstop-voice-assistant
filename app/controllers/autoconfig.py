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

from app.audio.devices import (
    AudioUnavailable,
    DeviceManager,
    cable_family,
    distinct_cables,
    looks_virtual,
)
from app.models.config import Config


#: Endpoints that exist but are useless as the app\'s cables: VoiceMeeter\'s
#: numbered insert channels ("Voicemeeter In 1".."In 5") and its 16-channel /
#: VAIO3 variants. They are real Windows devices, so the old code offered them
#: as defaults - and picking "Voicemeeter In 1" (which routes to no bus)
#: silently broke every call. The plainly named inputs ("Voicemeeter Input",
#: "Voicemeeter AUX Input") and the VB-CABLE endpoints are what actually work.
import re as _re
_JUNK_CABLE = _re.compile(r"\b(?:in|out)\s*\d+\b|\b16\s*ch\b|vaio3", _re.I)


def virtual_cables(manager: DeviceManager) -> list[str]:
    """Playback endpoints that are virtual cables - the usable ones first.

    The numbered VoiceMeeter insert channels sort last, so a default is never
    drawn from them while a real cable exists; they remain available so a
    machine that somehow has only those is not left with nothing.
    """
    seen: dict[str, None] = {}
    get_devs = getattr(manager, "wasapi_devices", None)
    devices = get_devs() if callable(get_devs) else []
    for device in devices:
        name = getattr(device, "name", "")
        if getattr(device, "can_render", False) and looks_virtual(name):
            seen.setdefault(name, None)
    # (is-junk, name) - False sorts before True, so real cables come first.
    return sorted(seen, key=lambda n: (bool(_JUNK_CABLE.search(n)), n))





def _multichannel(name: str) -> bool:
    import re

    return bool(re.search(r"\b\d+\s*ch\b", (name or "").lower()))


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
    active_mode = mode if mode is not None else (config.mode or "meeting")
    video = active_mode == "video"



    # Video mode needs one cable to listen to and somewhere to speak. It does
    # not need a microphone, and it must not open one: nothing is being sent
    # to anybody, so a live capture of the room would exist for no reason.
    if not devices.microphone and not video:
        try:
            # NOT merely "can capture". WASAPI exposes a loopback capture
            # endpoint for every playback device, named after it - so the
            # first capturing device on the list is quite often the loopback
            # of the speakers. Picked as the microphone, the outgoing
            # direction would send the client whatever the machine was
            # playing instead of the user's voice, which is both wrong and
            # alarming.
            inputs = [
                d for d in manager.wasapi_devices()
                if d.can_capture
                and not getattr(d, "is_loopback", False)
                and not d.can_render
                and not looks_virtual(d.name)
            ]
            devices.microphone = (
                inputs[0].name if inputs else manager.default_input().name
            )
            notes.append(f"Microphone: {devices.microphone}")
            changed = True
        except AudioUnavailable as exc:
            problems.append(f"No default microphone ({exc})")

    if not devices.headphones:
        try:
            outputs = [d for d in manager.wasapi_devices() if d.can_render and not looks_virtual(d.name)]
            devices.headphones = outputs[0].name if outputs else manager.default_output().name
            notes.append(f"Headphones: {devices.headphones}")
            changed = True
        except AudioUnavailable as exc:
            problems.append(f"No default playback device ({exc})")


    # A saved name that no longer exists is worse than a blank one: it looks
    # configured and fails at open time. Windows renames endpoints when a
    # driver is re-enumerated - installing VoiceMeeter turned "CABLE Input
    # (VB-Audio Virtual Cable)" into "Speakers (2- VB-Audio Virtual Cable)" on
    # the machine this was written for - so saved names are verified against
    # what is actually present, and the missing ones are cleared for the
    # assignment below to fill.
    present = {d.name for d in manager.wasapi_devices()} if manager else set()
    # Only the cable roles. A microphone or a pair of headphones that is not
    # in the list right now is usually unplugged rather than renamed, and
    # clearing a user's chosen headset because it was on charge would be its
    # own bug - the engine reports a missing one clearly enough. The cables
    # are different: a stale cable name cannot be recovered from, and nobody
    # chose it in the first place.
    if present:
        for role, label in (("virtual_mic", "Translator microphone"),
                            ("teams_speaker", "Teams speaker")):
            saved = getattr(devices, role)
            if saved and saved not in present:
                setattr(devices, role, "")
                changed = True
                notes.append(
                    f"{label}: cleared {saved!r} - no endpoint of that name "
                    f"exists any more"
                )

    # A previously-saved JUNK cable ("Voicemeeter In 1" etc.) is cleared so
    # the assignment below can replace it with a real one. It looks
    # configured but routes nowhere, which is worse than blank.
    for role in ("virtual_mic", "teams_speaker"):
        val = getattr(devices, role)
        if val and _JUNK_CABLE.search(val):
            setattr(devices, role, "")
            changed = True
            notes.append(
                f"{role.replace('_', ' ')}: cleared {val!r} - a numbered "
                f"insert channel that routes to no bus"
            )

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
    # One endpoint per PHYSICAL cable. VB-Audio's free driver exposes its one
    # cable twice - "CABLE Input" and "CABLE In 16ch" - and offering both to
    # the two roles is a loop with extra steps.
    valid_cables = distinct_cables(cables)

    # A saved multi-channel endpoint is replaced by the ordinary door onto the
    # same wire when there is one. Both work, but 16 channels is a surprising
    # thing to render two channels of speech into, and the plain endpoint is
    # what every instruction and screenshot in the world refers to.
    for role in ("virtual_mic", "teams_speaker"):
        saved = getattr(devices, role)
        if not saved or not _multichannel(saved):
            continue
        plain = [
            c for c in valid_cables
            if cable_family(c) == cable_family(saved) and not _multichannel(c)
        ]
        if plain:
            setattr(devices, role, plain[0])
            changed = True
            notes.append(
                f"{role.replace('_', ' ')}: {plain[0]} (instead of the "
                f"16-channel endpoint of the same cable)"
            )

    # THE ROLES ARE NOT INTERCHANGEABLE, and this is the asymmetry that broke a
    # real call. What we CAPTURE from Teams is taken by WASAPI loopback off a
    # render endpoint, so any playback cable will do. What we SEND to Teams has
    # to arrive as a recording device Teams can select - so that cable must
    # have a capture half on the same wire.
    #
    # VB-CABLE has one: "Speakers (2- VB-Audio Virtual Cable)" pairs with
    # "CABLE Output (2- VB-Audio Virtual Cable)". VoiceMeeter does not, in any
    # way we can predict: its recording endpoints are buses ("Voicemeeter Out
    # B2") wired to inputs inside its own mixer, not fixed partners of them.
    # Give VoiceMeeter the loopback role and the real cable the microphone
    # role, or the client hears nothing from us at all.
    capture_families = {
        cable_family(d.name)
        for d in (manager.wasapi_devices() if manager else [])
        if getattr(d, "can_capture", False) and not getattr(d, "is_loopback", False)
    }
    sendable = [c for c in valid_cables if cable_family(c) in capture_families]

    if devices.virtual_mic and capture_families and cable_family(
        devices.virtual_mic
    ) not in capture_families and sendable:
        notes.append(
            f"Translator microphone: cleared {devices.virtual_mic!r} - it has "
            f"no recording half, so Teams could never listen to it"
        )
        devices.virtual_mic = ""
        changed = True

    if not devices.virtual_mic and sendable:
        devices.virtual_mic = sendable[0]
        notes.append(f"Translator microphone: {devices.virtual_mic}")
        changed = True
        if cable_family(devices.teams_speaker or "") == cable_family(
            devices.virtual_mic
        ):
            devices.teams_speaker = ""

    # Distinctness is judged by WIRE, not by name. "Speakers (2- VB-Audio
    # Virtual Cable)" and "CABLE In 16 Ch (2- VB-Audio Virtual Cable)" are two
    # names for one cable, so a check that only compared strings happily put
    # one in each role - which is the loop, wearing a disguise.
    if (
        devices.virtual_mic
        and devices.teams_speaker
        and cable_family(devices.virtual_mic) == cable_family(devices.teams_speaker)
    ):
        notes.append(
            f"Translator microphone: cleared {devices.virtual_mic!r} - it is "
            f"the same cable as {devices.teams_speaker!r}, under another name"
        )
        devices.virtual_mic = ""
        changed = True

    if not devices.teams_speaker and valid_cables:
        taken = cable_family(devices.virtual_mic) if devices.virtual_mic else None
        free = [c for c in valid_cables if cable_family(c) != taken]
        devices.teams_speaker = (free or valid_cables)[0]
        notes.append(f"Teams speaker: {devices.teams_speaker}")
        changed = True

    if not devices.virtual_mic and valid_cables:
        # Never the same endpoint as the other role, and never a fallback to
        # it. One cable in both roles is not a degraded setup, it is a loop:
        # the English we render into it is the audio we capture out of it, so
        # it gets recognised as the client speaking, translated back, and
        # played louder every pass. Left EMPTY instead, because an empty
        # dropdown says "missing" - which is the truth - while a duplicated
        # one says "configured" and then fails at open time.
        taken = cable_family(devices.teams_speaker) if devices.teams_speaker else None
        spare = [c for c in valid_cables if cable_family(c) != taken]
        if spare:
            devices.virtual_mic = spare[0]
            notes.append(f"Translator microphone: {devices.virtual_mic}")
            changed = True

    # Checked after the assignments as well as before them, because a repair
    # can hand the one cable on the machine to a role whose partner already
    # holds it.
    if devices.virtual_mic and devices.teams_speaker and cable_family(
        devices.virtual_mic
    ) == cable_family(devices.teams_speaker):
        devices.virtual_mic = ""
        changed = True

    if video:
        source = valid_cables[0] if valid_cables else ""
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
        # Counted in cables, not endpoints. Saying "this machine has 2" when
        # both are ends of the same wire is how somebody spends an afternoon
        # certain they have installed what they need.
        found = ", ".join(valid_cables) if valid_cables else "none"
        extra = ""
        if len(cables) > len(valid_cables):
            spare = ", ".join(c for c in cables if c not in valid_cables)
            extra = (
                f"\n      ({spare} is another endpoint of a cable already "
                f"listed - the\n       same wire, not a second one.)\n"
            )
        problems.append(
            f"Two-way mode needs TWO virtual cables and this machine has "
            f"{len(valid_cables)} ({found}).{extra}\n"
            "      One carries your English into Teams; the other carries "
            "Teams' audio\n"
            "      out where we can capture it. They cannot be the same "
            "endpoint - what\n"
            "      we render into it is what we would capture out of it.\n\n"
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
