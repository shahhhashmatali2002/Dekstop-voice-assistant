"""Canonical internal audio format.

Spec reference: §5.4.

Windows endpoints hand us whatever they please - commonly 48 kHz float32
stereo. Providers want something specific and usually different. We convert
once, at the WASAPI boundary, into one of exactly two canonical rates, and
nothing else in the codebase knows other formats exist.
"""

from __future__ import annotations

import numpy as np

#: Sample format used everywhere inside the pipeline.
DTYPE = np.int16

#: Channel count used everywhere inside the pipeline.
CHANNELS = 1

#: Rate for recognition paths (ASR).
RATE_ASR = 16_000

#: Rate for synthesis and speech-to-speech paths (TTS / S2ST).
RATE_S2ST = 24_000

#: Canonical frame duration. Matches the VAD window and divides cleanly into
#: typical WASAPI device periods.
FRAME_MS = 20


def frame_samples(rate: int, frame_ms: int = FRAME_MS) -> int:
    """Samples in one frame at `rate`.

    >>> frame_samples(16_000)
    320
    >>> frame_samples(24_000)
    480
    """
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    if (rate * frame_ms) % 1000:
        raise ValueError(
            f"{frame_ms} ms is not a whole number of samples at {rate} Hz"
        )
    return rate * frame_ms // 1000


def ms_to_samples(ms: float, rate: int) -> int:
    """Round `ms` at `rate` to whole samples."""
    return int(round(ms * rate / 1000.0))


def samples_to_ms(n: int, rate: int) -> float:
    """Duration of `n` samples at `rate`, in milliseconds."""
    return n * 1000.0 / rate
