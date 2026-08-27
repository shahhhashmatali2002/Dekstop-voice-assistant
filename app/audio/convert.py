"""Format conversion at the WASAPI boundary.

Spec reference: §5.4.

Conversion happens exactly twice per pipeline, both times at the device edge:

    endpoint (48 kHz f32 stereo) -> downmix -> resample -> int16  -> pipeline
    pipeline (int16 mono)        -> resample -> f32 -> upmix      -> endpoint

Everything here runs on the ENGINE thread, never in an audio callback, even
though it would be convenient there. numpy and soxr release the GIL for the
bulk of their work, which is what makes that affordable.
"""

from __future__ import annotations

import numpy as np

from app.audio.formats import DTYPE

# soxr is the preferred resampler: fast, high quality, releases the GIL.
# scipy is an acceptable fallback. If neither is present we refuse loudly at
# call time rather than silently degrading audio quality.
try:  # pragma: no cover - import-path branch
    import soxr

    _BACKEND = "soxr"
except ImportError:  # pragma: no cover
    soxr = None
    try:
        from scipy import signal as _scipy_signal

        _BACKEND = "scipy"
    except ImportError:
        _scipy_signal = None
        _BACKEND = None

_INT16_PEAK = 32767.0


def backend() -> str | None:
    """Which resampler is in use: 'soxr', 'scipy', or None."""
    return _BACKEND


# ----------------------------------------------------------------------
# dtype
# ----------------------------------------------------------------------


def float_to_int16(x: np.ndarray) -> np.ndarray:
    """Convert float32/64 in [-1, 1] to int16, clipping rather than wrapping.

    Clipping matters: a wrapped sample is a full-scale discontinuity and is
    far more audible than a clipped one.
    """
    if x.dtype == DTYPE:
        return x
    scaled = np.clip(x, -1.0, 1.0) * _INT16_PEAK
    return scaled.astype(DTYPE)


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 to float32 in [-1, 1]."""
    if x.dtype == np.float32:
        return x
    return (x.astype(np.float32)) / _INT16_PEAK


# ----------------------------------------------------------------------
# channels
# ----------------------------------------------------------------------


def to_mono(x: np.ndarray, channels: int) -> np.ndarray:
    """Downmix interleaved `channels`-channel audio to mono.

    Averaging in a wider dtype avoids the int16 overflow that a naive sum
    would produce on loud material.
    """
    if channels == 1:
        return x.reshape(-1)
    frames = x.reshape(-1, channels)
    if frames.dtype == DTYPE:
        return frames.astype(np.int32).mean(axis=1).astype(DTYPE)
    return frames.mean(axis=1).astype(frames.dtype)


def to_channels(x: np.ndarray, channels: int) -> np.ndarray:
    """Upmix mono to interleaved `channels`-channel audio by duplication."""
    mono = x.reshape(-1)
    if channels == 1:
        return mono
    return np.repeat(mono, channels)


# ----------------------------------------------------------------------
# rate
# ----------------------------------------------------------------------


def resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono `x` from `src_rate` to `dst_rate`, preserving dtype.

    Raises RuntimeError if no resampler backend is installed - silently
    returning wrong-rate audio would surface as a bizarre recognition failure
    much later.
    """
    if src_rate == dst_rate:
        return x
    if _BACKEND is None:  # pragma: no cover - environment branch
        raise RuntimeError(
            "no resampler available: pip install soxr (preferred) or scipy"
        )

    was_int = x.dtype == DTYPE
    f = int16_to_float32(x) if was_int else x.astype(np.float32)

    if _BACKEND == "soxr":
        out = soxr.resample(f, src_rate, dst_rate)
    else:  # pragma: no cover - fallback branch
        from math import gcd

        g = gcd(src_rate, dst_rate)
        out = _scipy_signal.resample_poly(f, dst_rate // g, src_rate // g)

    out = np.asarray(out, dtype=np.float32)
    return float_to_int16(out) if was_int else out


# ----------------------------------------------------------------------
# level
# ----------------------------------------------------------------------


def apply_gain_db(x: np.ndarray, db: float) -> np.ndarray:
    """Apply `db` of gain. Clips on the int16 path rather than wrapping.

    Used for ducking during barge-in (§7.2) and for the optional low mix of
    the original English (§5.6 F3).
    """
    if db == 0.0:
        return x
    factor = 10.0 ** (db / 20.0)
    if x.dtype == DTYPE:
        scaled = x.astype(np.float32) * factor
        return float_to_int16(scaled / _INT16_PEAK)
    return (x * factor).astype(x.dtype)


def peak_dbfs(x: np.ndarray) -> float:
    """Peak level in dBFS. Returns -inf for digital silence."""
    if x.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(x.astype(np.float64))))
    if x.dtype == DTYPE:
        peak /= _INT16_PEAK
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * np.log10(peak)


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS. Returns -inf for digital silence."""
    if x.size == 0:
        return float("-inf")
    f = x.astype(np.float64)
    if x.dtype == DTYPE:
        f = f / _INT16_PEAK
    ms = float(np.mean(f * f))
    if ms <= 0.0:
        return float("-inf")
    return 10.0 * np.log10(ms)


def comfort_noise(n: int, level_dbfs: float = -70.0, rng=None) -> np.ndarray:
    """Generate `n` samples of very low-level noise.

    Spec reference: §5.7. Teams' noise suppressor treats an unnaturally clean
    digital silence as suspicious and can gate the speech that follows it, so
    the virtual microphone is never handed true zeros between utterances.
    """
    rng = rng or np.random.default_rng()
    amp = (10.0 ** (level_dbfs / 20.0)) * _INT16_PEAK
    return (rng.standard_normal(n) * amp).astype(DTYPE)


# ----------------------------------------------------------------------
# the two boundary helpers
# ----------------------------------------------------------------------


def endpoint_to_pipeline(
    raw: np.ndarray, src_rate: int, src_channels: int, dst_rate: int
) -> np.ndarray:
    """Device capture buffer -> canonical int16 mono at `dst_rate`."""
    mono = to_mono(raw, src_channels)
    mono = float_to_int16(mono) if mono.dtype != DTYPE else mono
    return resample(mono, src_rate, dst_rate)


def pipeline_to_endpoint(
    pcm: np.ndarray, src_rate: int, dst_rate: int, dst_channels: int, as_float: bool
) -> np.ndarray:
    """Canonical int16 mono -> interleaved device render buffer."""
    out = resample(pcm, src_rate, dst_rate)
    if as_float:
        out = int16_to_float32(out)
    return to_channels(out, dst_channels)
