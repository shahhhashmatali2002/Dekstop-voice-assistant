"""Reading an audio file that was not written for us.

A downloaded video's audio is whatever the site happened to serve - on
YouTube usually Opus in a WebM container, sometimes AAC in an MP4 one. The
transcription path never had to care, because it uploads the bytes and
Deepgram decodes them. Anything that wants SAMPLES does have to care, and
miniaudio - which is what the rest of this application decodes with - handles
WAV, MP3, FLAC and Ogg Vorbis, none of which is what arrived.

    miniaudio.DecodeError: ('failed to decode file', -1)

That is the whole bug this module exists to fix. The fallback is ffmpeg,
which decodes everything, found either on PATH or inside the imageio-ffmpeg
wheel - the second matters on Windows, where "install ffmpeg" means finding a
build, unpacking it somewhere sensible and editing PATH, and a pip install
does not.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from app.audio.formats import DTYPE


class DecodeUnavailable(RuntimeError):
    """Nothing on this machine can read that file."""


def ffmpeg_exe() -> str:
    """ffmpeg on PATH, or the one imageio-ffmpeg carries. "" if neither."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def _read_wav(path: Path, rate: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        got = handle.getframerate()
    if width != 2:
        raise DecodeUnavailable(f"{path.name}: expected 16-bit samples")
    pcm = np.frombuffer(frames, dtype=DTYPE)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(DTYPE)
    if got != rate:
        from app.audio import convert

        pcm = convert.resample(pcm, got, rate)
    return pcm, rate


def _via_ffmpeg(path: Path, rate: int) -> tuple[np.ndarray, int]:
    exe = ffmpeg_exe()
    if not exe:
        raise DecodeUnavailable(
            f"Could not decode {path.name}. It is probably Opus or AAC, which "
            "needs ffmpeg:\n"
            "    .venv\\Scripts\\python -m pip install imageio-ffmpeg\n"
            "(that wheel carries its own ffmpeg, so nothing has to be added to "
            "PATH), or install ffmpeg yourself."
        )
    with tempfile.TemporaryDirectory(prefix="decode-") as scratch:
        out = Path(scratch) / "audio.wav"
        try:
            subprocess.run(
                [exe, "-hide_banner", "-loglevel", "error", "-nostdin",
                 "-i", str(path), "-ac", "1", "-ar", str(rate),
                 "-acodec", "pcm_s16le", str(out)],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", "replace").strip()
            raise DecodeUnavailable(
                f"ffmpeg could not decode {path.name}: {detail[:200]}"
            ) from exc
        if not out.exists() or out.stat().st_size == 0:
            raise DecodeUnavailable(f"ffmpeg produced nothing from {path.name}")
        return _read_wav(out, rate)


def read_audio_file(path: str | Path, rate: int = 24_000) -> tuple[np.ndarray, int]:
    """Any audio file as mono int16 at `rate`.

    miniaudio first because it is already a dependency and needs no external
    binary; ffmpeg when the container is one it does not know.
    """
    path = Path(path)
    if not path.exists():
        raise DecodeUnavailable(f"{path} does not exist")

    try:
        import miniaudio

        decoded = miniaudio.decode_file(str(path), nchannels=1, sample_rate=rate)
        pcm = np.frombuffer(decoded.samples, dtype=DTYPE)
        if pcm.size:
            return pcm, rate
    except ImportError:
        pass
    except Exception:
        pass  # the format is not one of miniaudio's four; that is expected

    return _via_ffmpeg(path, rate)
