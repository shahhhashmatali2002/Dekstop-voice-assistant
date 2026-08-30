"""Voice conversion: the parts that must work without the model installed.

The converter itself needs torch and a model download, so it is not exercised
here. What IS exercised is everything around it, because that is where the
failures would be silent: choosing a reference clip that is actually a voice,
degrading cleanly when nothing is installed, and never turning itself on.
"""

import numpy as np
import pytest

from app.providers.base import ProviderMisconfigured
from app.providers.voice_clone import (
    BACKENDS,
    VoiceLearner,
    KnnVoiceConverter,
    NullVoiceConverter,
    Reference,
    SeedVoiceConverter,
    best_device,
    build_converter,
    pick_reference,
)

RATE = 24_000


def tone(seconds: float, hz: float, amp: int) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


# ----------------------------------------------------------------------
# choosing what to sound like
# ----------------------------------------------------------------------


def test_the_reference_is_not_just_the_first_ten_seconds():
    """A video's first ten seconds are an intro sting or a logo, and a
    reference clip of music produces a converted voice that sounds like a
    fault rather than a person."""
    sting = tone(4, 80, 14_000)          # loud, low, and not a voice
    speech = tone(30, 200, 5_000)        # ordinary speaking level
    got = pick_reference(np.concatenate([sting, speech]), RATE, seconds=8)

    assert got.seconds == pytest.approx(8.0, abs=0.2)
    # It should have skipped past the sting rather than starting at zero.
    assert not np.array_equal(got.pcm[: RATE], sting[: RATE])


def test_a_clip_shorter_than_asked_for_is_used_whole():
    short = tone(3, 200, 5_000)
    got = pick_reference(short, RATE, seconds=10)
    assert got.pcm.size == short.size


def test_silence_in_silence_out():
    got = pick_reference(np.empty(0, dtype=np.int16), RATE)
    assert got.pcm.size == 0
    assert got.seconds == 0.0


def test_the_reference_knows_its_own_length():
    assert Reference(tone(2.5, 200, 4000), RATE).seconds == pytest.approx(2.5)


# ----------------------------------------------------------------------
# degrading honestly
# ----------------------------------------------------------------------


def test_the_default_converter_changes_nothing():
    """The application must run identically with conversion switched off, so
    the off switch is a converter rather than a branch."""
    pcm = tone(1, 200, 4000)
    out, rate = NullVoiceConverter().convert(pcm, RATE, Reference(pcm, RATE))
    assert rate == RATE
    assert np.array_equal(out, pcm)


def test_a_missing_model_says_what_to_install():
    """Loudly, and with the command - not a stack trace about torch."""
    pcm = tone(1, 200, 4000)
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(ProviderMisconfigured) as caught:
            KnnVoiceConverter().convert(pcm, RATE, Reference(pcm, RATE))
        message = str(caught.value)
        assert "pip install torch" in message
        assert "clone_bench" in message
    else:  # pragma: no cover - only on a machine that has it
        pytest.skip("torch is installed here")


def test_the_seed_backend_explains_why_it_will_not_install():
    """The failure a user actually hit: seed-vc pins numpy below 2.0, there is
    no numpy 1.x wheel for Python 3.13, so pip tries to compile it and asks for
    a C compiler. The message has to name the way out, not the symptom."""
    pcm = tone(1, 200, 4000)
    try:
        import seed_vc  # noqa: F401
    except ImportError:
        with pytest.raises(ProviderMisconfigured) as caught:
            SeedVoiceConverter().convert(pcm, RATE, Reference(pcm, RATE))
        message = str(caught.value)
        assert "numpy" in message and "knn-vc" in message
    else:  # pragma: no cover
        pytest.skip("seed-vc is installed here")


def test_the_backends_are_addressable_by_name():
    assert set(BACKENDS) == {"knn", "light", "seed", "off"}
    assert build_converter("light").name == "light-match"
    assert build_converter("knn").name == "knn-vc"
    assert build_converter("seed").name == "seed-vc"
    assert build_converter("off").name == "null-converter"
    with pytest.raises(ProviderMisconfigured):
        build_converter("magic")


def test_the_default_backend_is_the_one_that_installs():
    """knn-vc needs torch, torchaudio and any numpy. Every one of those has a
    wheel; seed-vc's numpy pin does not."""
    assert build_converter().name == "knn-vc"


def test_knn_asks_for_minutes_and_seed_asks_for_seconds():
    """It converts by matching frames against the target's own audio, so more
    reference genuinely helps - up to about five minutes. Using the ten-second
    figure here would throw quality away for nothing, since a video's whole
    audio is already downloaded."""
    assert KnnVoiceConverter().preferred_reference_s >= 60.0
    assert SeedVoiceConverter().preferred_reference_s <= 30.0


def test_the_device_is_never_asserted_without_torch():
    assert best_device() in ("cpu", "cuda")
    assert best_device("cpu") == "cpu"
    assert KnnVoiceConverter(device="cpu").device == "cpu"


@pytest.mark.parametrize("factory", [SeedVoiceConverter, KnnVoiceConverter])
def test_conversion_without_a_reference_is_refused_before_loading_anything(factory):
    """An empty reference is a caller bug, and finding out after a 1 GB model
    download would be a poor way to learn it."""
    with pytest.raises(ProviderMisconfigured):
        factory().convert(
            tone(1, 200, 4000), RATE, Reference(np.empty(0, dtype=np.int16), RATE)
        )


@pytest.mark.parametrize("factory", [SeedVoiceConverter, KnnVoiceConverter])
def test_no_audio_is_returned_untouched_without_loading_anything(factory):
    empty = np.empty(0, dtype=np.int16)
    out, rate = factory().convert(empty, RATE, Reference(empty, RATE))
    assert out.size == 0 and rate == RATE


def test_the_matching_set_is_built_once_per_reference_not_once_per_line():
    """Building it from two minutes of audio is the expensive half of the whole
    operation and is identical for every line, so recomputing it per line would
    cost more than all the conversions put together."""
    calls = []

    class FakeModel:
        def get_matching_set(self, wavs):
            calls.append(len(wavs))
            return "features"

    class NoTorch(KnnVoiceConverter):
        """Torch is an optional install; the caching logic is not."""

        def _as_tensor(self, pcm, rate):
            return pcm

    converter = NoTorch()
    converter._model = FakeModel()
    reference = Reference(tone(2, 200, 4000), RATE)

    assert converter._matching_set(reference) == "features"
    assert converter._matching_set(reference) == "features"
    assert calls == [1], "the matching set was rebuilt"

    # A different reference must not reuse the previous speaker's features.
    converter._matching_set(Reference(tone(2, 300, 4000), RATE))
    assert calls == [1, 1]


def test_speed_is_measured_on_real_speech_rather_than_looked_up():
    """Two things at once.

    The number has to come from the machine it will run on - the same code
    differs by more than an order of magnitude between a laptop CPU and a
    desktop GPU.

    And it has to be measured on REAL audio. The first version generated a
    sine tone so no provider call was needed, which looked tidy and produced,
    on the user's machine:

        cannot reshape tensor of 0 elements into shape [-1, 0]

    kNN-VC trims non-speech before matching; a pure tone is entirely
    non-speech, so it was trimmed away and the model got an empty tensor.
    Hence the signature: caller-supplied audio, not something invented here.
    """
    converter = NullVoiceConverter()
    speech = tone(1, 200, 4000)
    factor = converter.measure(speech, RATE, Reference(speech, RATE))
    assert factor >= 0.0
    assert converter.realtime_factor is not None


def test_the_measured_factor_is_compute_per_second_of_audio():
    """Not per call: a two-second line and a ten-second line have to produce
    comparable numbers, or the budget means nothing."""
    import time as _time

    class Slow(NullVoiceConverter):
        def convert(self, pcm, rate, reference):
            _time.sleep(0.05)
            return pcm, rate

    one_second = tone(1, 200, 4000)
    two_seconds = tone(2, 200, 4000)
    reference = Reference(one_second, RATE)

    fast = Slow().measure(two_seconds, RATE, reference)
    slow = Slow().measure(one_second, RATE, reference)
    assert slow > fast, "the same work over less audio must read as slower"


# ----------------------------------------------------------------------
# learning the voice live, with no file to read ahead into
# ----------------------------------------------------------------------


def test_nothing_is_converted_until_enough_has_been_heard():
    """Converting against two seconds of audio sounds worse than not
    converting at all, so the first half-minute is the stock voice."""
    learner = VoiceLearner(RATE, min_seconds=5.0)
    assert learner.reference() is None

    for _ in range(20):                      # 20 x 0.2s = 4s: still short
        learner.hear(tone(0.2, 200, 4000))
    assert learner.reference() is None

    for _ in range(10):
        learner.hear(tone(0.2, 200, 4000))
    assert learner.reference() is not None


def test_the_same_reference_object_comes_back_until_it_is_worth_rebuilding():
    """Converters cache their expensive per-reference work on object identity.
    Handing back a fresh object every utterance would silently make every line
    cost as much as the first."""
    learner = VoiceLearner(RATE, min_seconds=1.0, upgrade_every=10.0)
    for _ in range(10):
        learner.hear(tone(0.2, 200, 4000))   # 2 seconds

    first = learner.reference()
    for _ in range(5):
        learner.hear(tone(0.2, 200, 4000))
    assert learner.reference() is first, "rebuilt for one extra second"


def test_it_improves_once_enough_new_speech_has_arrived():
    learner = VoiceLearner(RATE, min_seconds=1.0, upgrade_every=3.0)
    for _ in range(10):
        learner.hear(tone(0.2, 200, 4000))
    first = learner.reference()

    for _ in range(30):                      # six more seconds
        learner.hear(tone(0.2, 200, 4000))
    grown = learner.reference()
    assert grown is not first
    assert grown.seconds > first.seconds


def test_it_stops_growing_at_the_cap():
    """Quality stops improving after a few minutes; the memory would not."""
    learner = VoiceLearner(RATE, min_seconds=1.0, cap_seconds=3.0)
    for _ in range(100):
        learner.hear(tone(0.2, 200, 4000))
    assert learner.seconds <= 3.2


def test_a_reset_forgets_the_speaker():
    """Stopping and starting again may well be a different video."""
    learner = VoiceLearner(RATE, min_seconds=1.0)
    for _ in range(10):
        learner.hear(tone(0.2, 200, 4000))
    assert learner.reference() is not None
    learner.reset()
    assert learner.reference() is None
    assert learner.seconds == 0.0


# ----------------------------------------------------------------------
# the cheap resemblance, for machines with no GPU
# ----------------------------------------------------------------------


def voiceish(seconds: float, f0: float, amp: int = 6000) -> np.ndarray:
    """Something with a pitch and some harmonics. Not speech, but enough of a
    voice for pitch and spectrum to mean something."""
    t = np.arange(int(RATE * seconds)) / RATE
    x = sum((amp / (k + 1)) * np.sin(2 * np.pi * f0 * (k + 1) * t) for k in range(6))
    return (x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.int16)


def test_the_pitch_moves_towards_the_speaker():
    from app.providers.voice_clone import LightVoiceMatcher

    matcher = LightVoiceMatcher()
    reference = Reference(voiceish(4, 120), RATE)
    ours = voiceish(2, 150)

    out, rate = matcher.convert(ours, RATE, reference)
    assert rate == RATE
    before = matcher.median_f0(ours, RATE)
    after = matcher.median_f0(out, RATE)
    assert abs(after - 120) < abs(before - 120), (
        f"pitch went from {before:.0f} to {after:.0f}, target 120"
    )


def test_the_length_is_preserved_exactly():
    """A per cent per utterance is a drift that accumulates against a video's
    clock over an hour."""
    from app.providers.voice_clone import LightVoiceMatcher

    ours = voiceish(2, 150)
    out, _ = LightVoiceMatcher().convert(ours, RATE, Reference(voiceish(4, 110), RATE))
    assert out.size == ours.size


def test_an_absurd_pitch_gap_is_clamped_rather_than_forced():
    """Two voices an octave apart are not the same person, and pretending
    otherwise produces artefacts rather than resemblance."""
    from app.providers.voice_clone import LightVoiceMatcher

    matcher = LightVoiceMatcher()
    out, _ = matcher.convert(voiceish(2, 240), RATE, Reference(voiceish(4, 90), RATE))
    after = matcher.median_f0(out, RATE)
    assert after > 240 * matcher.MIN_RATIO * 0.9, f"shifted all the way to {after:.0f}"


def test_it_is_fast_enough_to_run_live_on_a_cpu():
    """The entire reason it exists: kNN-VC measured 4.3s per second of audio
    on the user's laptop, which is a factor of twelve away from usable."""
    from app.providers.voice_clone import LightVoiceMatcher

    matcher = LightVoiceMatcher()
    speech = voiceish(2, 150)
    factor = matcher.measure(speech, RATE, Reference(voiceish(4, 110), RATE))
    assert factor < 0.35, f"{factor:.2f}s per second of audio"


def test_the_urdu_voice_is_chosen_by_the_speakers_pitch():
    """Worth more than the processing that follows it: reading with the male
    voice starts an octave closer to a man than pitch-shifting the female one
    ever gets, and with no artefacts."""
    from app.providers.voice_clone import voice_for_pitch

    assert voice_for_pitch(105) == "ur-PK-AsadNeural"
    assert voice_for_pitch(215) == "ur-PK-UzmaNeural"
    assert voice_for_pitch(0) == "", "an unvoiced sample must not pick a voice"


def test_no_reference_no_conversion():
    from app.providers.voice_clone import LightVoiceMatcher

    with pytest.raises(ProviderMisconfigured):
        LightVoiceMatcher().convert(
            voiceish(1, 150), RATE, Reference(np.empty(0, dtype=np.int16), RATE)
        )
