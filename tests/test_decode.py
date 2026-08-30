"""Reading audio files that were not written for us.

The bug: a downloaded YouTube track is Opus in a WebM container, miniaudio
reads WAV, MP3, FLAC and Ogg Vorbis, and the whole clone path fell over on
    miniaudio.DecodeError: ('failed to decode file', -1)
"""

import wave
from pathlib import Path

import numpy as np
import pytest

from app.audio.decode import DecodeUnavailable, ffmpeg_exe, read_audio_file


def write_wav(path: Path, pcm: np.ndarray, rate: int, channels: int = 1) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.astype(np.int16).tobytes())
    return path


def tone(seconds: float, rate: int, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (4000 * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def test_a_plain_wav_is_read(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone(1, 24_000), 24_000)
    pcm, rate = read_audio_file(path, 24_000)
    assert rate == 24_000
    assert pcm.dtype == np.int16
    assert pcm.size == pytest.approx(24_000, rel=0.05)


def test_it_is_resampled_to_what_was_asked_for(tmp_path):
    """The reference clip and the synthesised Urdu have to meet at one rate."""
    path = write_wav(tmp_path / "b.wav", tone(1, 48_000), 48_000)
    pcm, rate = read_audio_file(path, 16_000)
    assert rate == 16_000
    assert pcm.size == pytest.approx(16_000, rel=0.05)


def test_stereo_is_mixed_down(tmp_path):
    mono = tone(1, 24_000)
    stereo = np.repeat(mono, 2)
    path = write_wav(tmp_path / "c.wav", stereo, 24_000, channels=2)
    pcm, _ = read_audio_file(path, 24_000)
    assert pcm.size == pytest.approx(mono.size, rel=0.05)


def test_a_file_that_is_not_there_says_so(tmp_path):
    with pytest.raises(DecodeUnavailable) as caught:
        read_audio_file(tmp_path / "missing.webm")
    assert "does not exist" in str(caught.value)


def test_something_undecodable_names_the_fix(tmp_path):
    """Not a stack trace out of miniaudio: the message has to say what to
    install, because on Windows "install ffmpeg" is a half-hour of work and
    a pip wheel that carries one is thirty seconds."""
    junk = tmp_path / "d.webm"
    junk.write_bytes(b"this is not audio" * 100)
    with pytest.raises(DecodeUnavailable) as caught:
        read_audio_file(junk)
    message = str(caught.value)
    if ffmpeg_exe():
        assert "ffmpeg could not decode" in message
    else:
        assert "imageio-ffmpeg" in message


def test_ffmpeg_is_looked_for_in_both_places():
    """PATH first, then the imageio-ffmpeg wheel. Returns "" rather than
    raising, so a machine with neither still loads this module."""
    assert isinstance(ffmpeg_exe(), str)
