"""The cable prober. Every hour lost to audio on this project came from
believing a device choice instead of measuring it, so the measurement itself
has to be right."""

from __future__ import annotations

import math

import numpy as np

from app.audio.probe import (
    AMPLITUDE,
    HEARD_DB,
    HEARD_PURITY,
    Listener,
    TONE_HZ,
    WireReport,
    candidates,
    db,
    driver_of,
    purity,
    tone,
    tone_amplitude,
)


# ---------------------------------------------------------------- the tone


def test_the_tone_is_the_frequency_it_claims():
    rate = 48_000
    pcm = tone(rate, seconds=0.5)
    found = tone_amplitude(pcm, rate, TONE_HZ)
    expected = AMPLITUDE * 32767.0
    # Within a fade's worth: the ramps at each end remove a little energy.
    assert abs(found - expected) / expected < 0.05


def test_the_tone_does_not_click():
    """A step at either end is heard as a click and, on a real endpoint, is
    what makes a test tone audible to the person on the other end."""
    pcm = tone(48_000, seconds=0.3)
    assert abs(int(pcm[0])) < 200
    assert abs(int(pcm[-1])) < 200


def test_a_different_frequency_is_not_the_tone():
    """440 Hz leaks a little into a 997 Hz measurement - any single-frequency
    fit does. Purity is what rejects it: almost none of that signal IS our
    tone, so it can never be mistaken for a working cable."""
    rate = 48_000
    t = np.arange(rate // 2, dtype=np.float64) / rate
    other = (np.sin(2 * np.pi * 440.0 * t) * 0.25 * 32767).astype(np.int16)
    listener = Listener(
        name="wrong tone", kind="recording",
        level_db=db(tone_amplitude(other, rate, TONE_HZ)),
        purity=purity(other, rate, TONE_HZ),
    )
    assert listener.purity < 0.05
    assert not listener.heard


# ------------------------------------------------------------ what it hears


def test_silence_is_not_heard():
    silence = np.zeros(24_000, dtype=np.int16)
    assert db(tone_amplitude(silence, 48_000)) < HEARD_DB
    assert purity(silence, 48_000) == 0.0


def test_the_tone_at_the_far_end_is_heard():
    rate = 48_000
    heard = tone(rate, seconds=0.5)
    listener = Listener(
        name="CABLE Output (VB-Audio Virtual Cable)",
        kind="recording",
        level_db=db(tone_amplitude(heard, rate)),
        purity=purity(heard, rate),
    )
    assert listener.heard
    assert listener.level_db > -20.0
    assert listener.purity > 0.9


def test_noise_alone_is_not_mistaken_for_a_working_cable():
    """The reason purity exists. An endpoint carrying music, hiss or somebody
    else's call has plenty of energy at -20 dB and would otherwise pass."""
    rng = np.random.default_rng(7)
    rate = 48_000
    noise = (rng.normal(0.0, 0.15, rate // 2) * 32767).astype(np.int16)
    listener = Listener(
        name="noisy", kind="recording",
        level_db=db(tone_amplitude(noise, rate)),
        purity=purity(noise, rate),
    )
    assert listener.purity < HEARD_PURITY
    assert not listener.heard


def test_a_quiet_but_real_tone_is_still_heard():
    """A cable with its volume down is still a working cable. -45 dB is far
    below anything we would play, and must not read as dead."""
    rate = 48_000
    quiet = (tone(rate, seconds=0.5).astype(np.float64) * 0.004).astype(np.int16)
    listener = Listener(
        name="quiet", kind="recording",
        level_db=db(tone_amplitude(quiet, rate)),
        purity=purity(quiet, rate),
    )
    assert -65.0 < listener.level_db < -55.0
    assert listener.heard
    # And it says so, rather than passing silently: a cable this quiet carries
    # speech as a mumble, which reads downstream as "the recogniser is broken".
    assert listener.quiet


def test_a_quiet_cable_is_called_out_in_the_verdict():
    report = WireReport(role="send", render="CABLE Input (VB-Audio Virtual Cable)")
    report.listeners = [
        Listener(name="CABLE Output (VB-Audio Virtual Cable)", kind="recording",
                 level_db=-62.0, purity=0.98),
    ]
    assert report.ok
    assert "QUIET" in report.verdict()


def test_the_tone_survives_a_resampled_cable():
    """Endpoints negotiate their own rate; 997 Hz is chosen so that a
    resampler cannot land it on a bin boundary and lose it."""
    from app.audio import convert

    played = tone(48_000, seconds=0.5)
    arrived = convert.resample(played, 48_000, 16_000)
    assert db(tone_amplitude(arrived, 16_000)) > HEARD_DB
    assert purity(arrived, 16_000) > 0.9


# ------------------------------------------------------------ driver naming


def test_the_driver_is_the_name_in_brackets():
    assert driver_of("CABLE Output (VB-Audio Virtual Cable)") == "vb-audio virtual cable"
    assert driver_of("Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)") == (
        "vb-audio voicemeeter vaio"
    )
    assert driver_of("Speaker (Realtek(R) Audio)") == "r"  # last brackets win
    assert driver_of("No brackets here") == ""


def test_two_endpoints_of_one_cable_share_a_driver():
    assert driver_of("CABLE Input (VB-Audio Virtual Cable)") == driver_of(
        "CABLE Output (VB-Audio Virtual Cable)"
    )


# --------------------------------------------------------------- candidates


class Device:
    def __init__(self, name, *, capture=False, render=True, loopback=False):
        self.name = name
        self.can_capture = capture
        self.can_render = render
        self.is_loopback = loopback


class Manager:
    def __init__(self, devices):
        self._devices = devices

    def wasapi_devices(self):
        return list(self._devices)

    def loopback_for_output(self, device):
        return Device(f"{device.name} [Loopback]", capture=True, loopback=True)


def test_candidates_are_the_same_driver_plus_the_loopback():
    render = Device("CABLE Input (VB-Audio Virtual Cable)")
    manager = Manager([
        render,
        Device("CABLE Output (VB-Audio Virtual Cable)", capture=True, render=False),
        Device("Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)", capture=True,
               render=False),
        Device("Microphone Array (Intel Smart Sound)", capture=True, render=False),
    ])

    found = candidates(manager, render)
    names = [d.name for d, _ in found]
    kinds = [k for _, k in found]

    assert "CABLE Output (VB-Audio Virtual Cable)" in names
    assert "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)" not in names
    assert "Microphone Array (Intel Smart Sound)" not in names
    assert kinds.count("loopback") == 1


def test_voicemeeter_buses_are_all_tested_rather_than_guessed():
    """Which bus carries the audio depends on which B button is lit inside the
    mixer. The application cannot know that, so it must try all of them."""
    render = Device("Voicemeeter Input (VB-Audio Voicemeeter VAIO)")
    manager = Manager([
        render,
        Device("Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)", capture=True,
               render=False),
        Device("Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)", capture=True,
               render=False),
        Device("Voicemeeter Out A1 (VB-Audio Voicemeeter VAIO)", capture=True,
               render=False),
        Device("CABLE Output (VB-Audio Virtual Cable)", capture=True, render=False),
    ])

    names = [d.name for d, _ in candidates(manager, render)]
    assert "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)" in names
    assert "Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)" in names
    assert "Voicemeeter Out A1 (VB-Audio Voicemeeter VAIO)" in names
    assert "CABLE Output (VB-Audio Virtual Cable)" not in names


# ----------------------------------------------------------------- verdicts


def heard_listener(name, kind="recording"):
    return Listener(name=name, kind=kind, level_db=-14.0, purity=0.98)


def silent_listener(name, kind="recording"):
    return Listener(name=name, kind=kind, level_db=-96.7, purity=0.01)


def test_a_working_send_cable_names_the_device_teams_needs():
    report = WireReport(role="send", render="CABLE Input (VB-Audio Virtual Cable)")
    report.listeners = [
        heard_listener("CABLE Output (VB-Audio Virtual Cable)"),
        heard_listener("CABLE Input [Loopback]", "loopback"),
    ]
    assert report.ok
    assert "CABLE Output (VB-Audio Virtual Cable)" in report.verdict()


def test_a_dead_cable_says_so():
    report = WireReport(role="send", render="CABLE Input (VB-Audio Virtual Cable)")
    report.listeners = [silent_listener("CABLE Output (VB-Audio Virtual Cable)")]
    assert not report.ok
    assert "DEAD" in report.verdict()


def test_loopback_only_is_reported_as_partial():
    """This is the VoiceMeeter trap: we can hear our own output on the
    endpoint's loopback, so it looks fine from inside the application, while
    no recording device exists for Teams to select. The client hears nothing
    and nothing in the app looks wrong."""
    report = WireReport(role="send", render="Voicemeeter Input (VB-Audio Voicemeeter VAIO)")
    report.listeners = [
        heard_listener("Voicemeeter Input [Loopback]", "loopback"),
        silent_listener("Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)"),
        silent_listener("Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)"),
    ]
    assert report.ok
    assert "PARTIAL" in report.verdict()
    assert not report.recording


def test_a_device_that_will_not_open_is_not_silently_passed():
    report = WireReport(role="send", render="CABLE Input (VB-Audio Virtual Cable)")
    report.listeners = [
        Listener(name="CABLE Output (VB-Audio Virtual Cable)", kind="recording",
                 error="[Errno -9996] Invalid device"),
    ]
    assert not report.listeners[0].heard
    assert not report.ok


def test_a_missing_cable_cannot_be_tested_and_says_that():
    report = WireReport(role="send", render="CABLE In 99 Ch", error="no such endpoint")
    assert not report.ok
    assert "CANNOT TEST" in report.verdict()
