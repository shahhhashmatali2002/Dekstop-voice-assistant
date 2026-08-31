"""Meeting-mode readiness, checked before a call rather than during one.

Every failure this catches has the same expensive shape: it surfaces thirty
seconds into a conversation with somebody else on the line, and the only
recovery is to hang up.
"""

import pytest

from app.controllers.preflight import meeting_preflight
from app.models.config import Config


def ready_config(**overrides) -> Config:
    config = Config()
    config.devices.microphone = "Microphone Array"
    config.devices.headphones = "Headset Earphone"
    config.devices.virtual_mic = "CABLE-A Input (VB-Audio Cable A)"
    config.devices.teams_speaker = "CABLE-B Input (VB-Audio Cable B)"
    config.devices.auto_route = True
    for key, value in overrides.items():
        setattr(config.devices, key, value)
    return config


def test_a_complete_two_cable_setup_is_ready():
    report = meeting_preflight(ready_config())
    assert report.ok, report.problems
    assert report.warnings == [], report.warnings
    assert any("default devices are moved" in item for item in report.ready)


def test_the_teams_microphone_is_always_named_explicitly():
    """"Leave it on Default" read as reassuring and left the client listening
    to the user's raw Urdu: automatic routing moves the Windows default, and
    Teams remembers a device once it has been pointed at one."""
    report = meeting_preflight(ready_config())
    microphone = [i for i in report.instructions if "Microphone" in i]
    assert microphone, report.instructions
    assert "CABLE-A Output" in microphone[0], microphone[0]
    assert "hear English" in microphone[0]


def test_a_missing_role_blocks_and_names_itself():
    report = meeting_preflight(ready_config(virtual_mic=""))
    assert not report.ok
    assert any("English to Teams" in p for p in report.problems)


def test_a_speaker_in_a_cable_role_is_blocked():
    """Whatever we render into it would be audible in the room and captured
    straight back - which is the loop, not a quality problem."""
    report = meeting_preflight(ready_config(virtual_mic="Speaker (Realtek)"))
    assert not report.ok
    assert any("not a virtual cable" in p for p in report.problems)


def test_one_cable_in_both_roles_is_a_warning_not_a_block():
    """It reverses an earlier refusal deliberately: one cable is a loop, but
    refusing outright meant no two-way mode at all on a one-cable machine."""
    cable = "CABLE Input (VB-Audio Virtual Cable)"
    report = meeting_preflight(ready_config(virtual_mic=cable, teams_speaker=cable))
    assert report.ok, report.problems
    assert any("come back as the client speaking" in w for w in report.warnings)


def test_speakers_are_allowed_and_the_compromise_is_stated():
    """The user asked for speakers to be usable. They are - and the report says
    what that costs, rather than pretending it is free."""
    report = meeting_preflight(ready_config(headphones="Speaker (Realtek(R) Audio)"))
    assert report.ok, report.problems
    assert any("Half-duplex" in w for w in report.warnings)
    assert any("headphones avoid" in w for w in report.warnings)


def test_headphones_raise_nothing():
    for name in ("Headset Earphone", "Sony WH-1000 Headphones", "AirPods Pro",
                 "Jabra Hands-Free"):
        report = meeting_preflight(ready_config(headphones=name))
        assert report.warnings == [], f"{name}: {report.warnings}"


def test_the_teams_instructions_name_the_devices_either_way():
    automatic = meeting_preflight(ready_config(auto_route=True))
    assert any("remembers" in step for step in automatic.instructions)

    manual = meeting_preflight(ready_config(auto_route=False))
    joined = " ".join(manual.instructions)
    # The recording half, not the playback half: Teams records from CABLE-A
    # Output, which is the other end of the cable we render into.
    assert "CABLE-A Output" in joined, joined
    assert "CABLE-B Input" in joined, joined


def test_a_self_test_is_always_advised():
    report = meeting_preflight(ready_config())
    assert any("test call to yourself" in step for step in report.instructions)


def test_the_report_renders_without_a_device_manager():
    """It has to run on a machine where the audio stack will not open - that is
    the state somebody is in when they are trying to work out why."""
    text = meeting_preflight(ready_config(virtual_mic="")).render()
    assert "BLOCKED" in text and "In Teams:" in text


# ----------------------------------------------------------------------
# saying WHY an endpoint would not resolve
# ----------------------------------------------------------------------


class _FakeManager:
    """A device manager that fails the way the real one does."""

    def __init__(self, playback=(), error="no WASAPI endpoint matching"):
        self._playback = list(playback)
        self._error = error

    def wasapi_devices(self):
        class D:
            def __init__(self, name):
                self.name = name
                self.can_render = True
                self.can_capture = False
                self.is_loopback = False

        return [D(name) for name in self._playback]

    def find(self, needle, *, capture=None, loopback=None):
        for name in self._playback:
            if needle.lower() in name.lower():
                return self.wasapi_devices()[self._playback.index(name)]
        raise RuntimeError(f"{self._error} {needle!r}")


def test_a_missing_endpoint_reports_the_reason_and_what_is_there():
    """"not present on this machine now" is one of three different things -
    missing, disabled, or ambiguous - and the user cannot tell which from the
    outside."""
    manager = _FakeManager(playback=["Speaker (Realtek(R) Audio)",
                                     "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)"])
    report = meeting_preflight(ready_config(), manager)

    blocked = " ".join(report.problems)
    assert "could not be resolved" in blocked
    assert "Playback endpoints seen right now" in blocked
    assert "VoiceMeeter Input" in blocked, "the real list has to be shown"


def test_an_ambiguous_name_does_not_advise_enabling_devices():
    """Ambiguity is the opposite problem - too many matches, not none - and
    telling somebody to un-hide disabled devices sends them the wrong way."""
    manager = _FakeManager(error="is ambiguous, 2 endpoints match:")
    report = meeting_preflight(ready_config(), manager)
    blocked = " ".join(report.problems)
    assert "ambiguous" in blocked
    assert "Show Disabled Devices" not in blocked


def test_an_unset_cable_names_the_cables_that_could_fill_it():
    """The state the user was in: a cable installed ten minutes earlier, and a
    report that just said "not set"."""
    manager = _FakeManager(playback=[
        "Speaker (Realtek(R) Audio)",
        "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)",
        "VoiceMeeter Aux Input (VB-Audio VoiceMeeter AUX VAIO)",
    ])
    report = meeting_preflight(ready_config(virtual_mic=""), manager)
    blocked = " ".join(report.problems)
    assert "VoiceMeeter Aux Input" in blocked
    assert "python run.py" in blocked, "the way to fix it must be given"
