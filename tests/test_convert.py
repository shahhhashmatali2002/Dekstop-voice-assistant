"""Tests for the format conversion boundary."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio import convert
from app.audio.formats import DTYPE, RATE_ASR, RATE_S2ST, frame_samples


# ----------------------------------------------------------------------
# formats
# ----------------------------------------------------------------------


def test_frame_samples():
    assert frame_samples(16_000) == 320
    assert frame_samples(24_000) == 480
    assert frame_samples(48_000) == 960


def test_frame_samples_rejects_fractional():
    with pytest.raises(ValueError):
        frame_samples(44_100, 1)


# ----------------------------------------------------------------------
# dtype conversion
# ----------------------------------------------------------------------


def test_float_int16_roundtrip_is_close():
    f = np.linspace(-1.0, 1.0, 1001, dtype=np.float32)
    back = convert.int16_to_float32(convert.float_to_int16(f))
    np.testing.assert_allclose(back, f, atol=1e-4)


def test_float_to_int16_clips_rather_than_wraps():
    """A wrapped sample is a full-scale discontinuity - much worse than a clip."""
    loud = np.array([3.0, -3.0], dtype=np.float32)
    out = convert.float_to_int16(loud)
    assert out[0] == 32767
    assert out[1] == -32767
    assert out.dtype == DTYPE


def test_float_to_int16_passes_int16_through():
    x = np.array([1, 2, 3], dtype=DTYPE)
    assert convert.float_to_int16(x) is x


# ----------------------------------------------------------------------
# channels
# ----------------------------------------------------------------------


def test_to_mono_averages_stereo():
    stereo = np.array([100, 200, 300, 400], dtype=DTYPE)  # L,R,L,R
    np.testing.assert_array_equal(convert.to_mono(stereo, 2), [150, 350])


def test_to_mono_does_not_overflow_on_loud_stereo():
    """The naive int16 sum of two near-full-scale samples wraps. This must not."""
    stereo = np.array([32000, 32000], dtype=DTYPE)
    assert convert.to_mono(stereo, 2)[0] == 32000


def test_to_mono_passthrough_for_mono():
    x = np.array([1, 2, 3], dtype=DTYPE)
    np.testing.assert_array_equal(convert.to_mono(x, 1), x)


def test_to_channels_duplicates():
    mono = np.array([1, 2], dtype=DTYPE)
    np.testing.assert_array_equal(convert.to_channels(mono, 2), [1, 1, 2, 2])


def test_channel_roundtrip():
    mono = np.array([5, 10, 15], dtype=DTYPE)
    np.testing.assert_array_equal(convert.to_mono(convert.to_channels(mono, 2), 2), mono)


# ----------------------------------------------------------------------
# resampling
# ----------------------------------------------------------------------


def test_resample_same_rate_is_identity():
    x = np.arange(100, dtype=DTYPE)
    assert convert.resample(x, 16_000, 16_000) is x


@pytest.mark.parametrize(
    "src,dst", [(48_000, 16_000), (48_000, 24_000), (16_000, 24_000), (24_000, 16_000)]
)
def test_resample_length_is_proportional(src, dst):
    x = np.zeros(src, dtype=DTYPE)  # one second
    out = convert.resample(x, src, dst)
    assert abs(len(out) - dst) <= max(2, dst // 1000)


def test_resample_preserves_dtype():
    x = np.zeros(4800, dtype=DTYPE)
    assert convert.resample(x, 48_000, 16_000).dtype == DTYPE
    f = np.zeros(4800, dtype=np.float32)
    assert convert.resample(f, 48_000, 16_000).dtype == np.float32


def test_resample_preserves_a_sine_wave():
    """Downsampling a 440 Hz tone must leave a 440 Hz tone, not aliasing mush."""
    src, dst, freq = 48_000, 16_000, 440.0
    t = np.arange(src, dtype=np.float32) / src
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    out = convert.resample(tone, src, dst)

    spectrum = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    peak_hz = np.fft.rfftfreq(len(out), 1 / dst)[int(np.argmax(spectrum))]
    assert abs(peak_hz - freq) < 5.0


# ----------------------------------------------------------------------
# level
# ----------------------------------------------------------------------


def test_gain_zero_db_is_identity():
    x = np.array([100, 200], dtype=DTYPE)
    assert convert.apply_gain_db(x, 0.0) is x


def test_gain_minus_six_db_roughly_halves():
    x = np.full(16, 10_000, dtype=DTYPE)
    out = convert.apply_gain_db(x, -6.0)
    assert 4900 < out[0] < 5100


def test_gain_clips_rather_than_wrapping():
    x = np.full(4, 30_000, dtype=DTYPE)
    out = convert.apply_gain_db(x, 12.0)
    assert np.all(out == 32767)


def test_duck_level_used_by_bargein():
    """-20 dB is the ducking level in the barge-in handler (§7.2)."""
    x = np.full(8, 20_000, dtype=DTYPE)
    out = convert.apply_gain_db(x, -20.0)
    assert 1900 < out[0] < 2100


def test_peak_and_rms_of_silence_are_negative_infinity():
    z = np.zeros(64, dtype=DTYPE)
    assert convert.peak_dbfs(z) == float("-inf")
    assert convert.rms_dbfs(z) == float("-inf")


def test_peak_dbfs_of_full_scale_is_about_zero():
    x = np.full(16, 32767, dtype=DTYPE)
    assert convert.peak_dbfs(x) == pytest.approx(0.0, abs=0.01)


def test_rms_of_sine_is_about_three_db_below_peak():
    t = np.arange(48_000, dtype=np.float32) / 48_000
    tone = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    assert convert.rms_dbfs(tone) == pytest.approx(-3.01, abs=0.1)


# ----------------------------------------------------------------------
# comfort noise (§5.7)
# ----------------------------------------------------------------------


def test_comfort_noise_is_quiet_but_not_silent():
    n = convert.comfort_noise(4800, level_dbfs=-70.0)
    assert n.dtype == DTYPE
    assert len(n) == 4800
    assert not np.all(n == 0), "comfort noise must not be digital silence"
    assert convert.rms_dbfs(n) < -60.0


def test_comfort_noise_is_reproducible_with_a_seed():
    a = convert.comfort_noise(100, rng=np.random.default_rng(7))
    b = convert.comfort_noise(100, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# ----------------------------------------------------------------------
# the two boundary helpers
# ----------------------------------------------------------------------


def test_endpoint_to_pipeline_from_typical_windows_format():
    """48 kHz float32 stereo is what a Windows endpoint usually hands us."""
    frames = 4800  # 100 ms
    stereo_f32 = np.zeros(frames * 2, dtype=np.float32)
    out = convert.endpoint_to_pipeline(stereo_f32, 48_000, 2, RATE_ASR)
    assert out.dtype == DTYPE
    assert abs(len(out) - 1600) <= 2


def test_pipeline_to_endpoint_back_to_device_format():
    pcm = np.zeros(2400, dtype=DTYPE)  # 100 ms at 24 kHz
    out = convert.pipeline_to_endpoint(pcm, RATE_S2ST, 48_000, 2, as_float=True)
    assert out.dtype == np.float32
    assert abs(len(out) - 4800 * 2) <= 4


def test_full_boundary_roundtrip_preserves_a_tone():
    """Device -> pipeline -> device must still be recognisably the same tone."""
    src_rate, frames, freq = 48_000, 48_000, 440.0
    t = np.arange(frames, dtype=np.float32) / src_rate
    mono = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = convert.to_channels(mono, 2)

    pipeline = convert.endpoint_to_pipeline(stereo, src_rate, 2, RATE_ASR)
    back = convert.pipeline_to_endpoint(pipeline, RATE_ASR, src_rate, 1, as_float=True)

    spectrum = np.abs(np.fft.rfft(back * np.hanning(len(back))))
    peak_hz = np.fft.rfftfreq(len(back), 1 / src_rate)[int(np.argmax(spectrum))]
    assert abs(peak_hz - freq) < 5.0
