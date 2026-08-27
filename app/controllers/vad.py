"""Voice activity detection and utterance boundaries.

Spec reference: §6.2, §9.3.

VAD is the gate that decides whether a frame is worth anything. Without it we
stream silence to a metered API for the eighty per cent of a meeting when our
side is not talking, and we hand the recogniser long stretches of room tone to
hallucinate over.

    property              choice                       why
    -------------------------------------------------------------------------
    model                 Silero via onnxruntime       robust in real rooms;
                                                       releases the GIL
    window                20 ms, smoothed over ~100 ms matches the frame size
    onset                 3 consecutive speech frames  ignores keyboard clicks
    offset (hangover)     300-600 ms of silence        THE latency knob
    pre-roll              200-300 ms retained          recovers the first
                                                       phoneme

THE HANGOVER IS THE DOMINANT LATENCY KNOB (§9.3)
------------------------------------------------
If the product feels slow, look here before you look at the model or the
network. Too short chops sentences; too long adds delay to every single
utterance. Tune it against recorded audio, never in a live meeting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.audio.formats import RATE_ASR, ms_to_samples
from app.models.telemetry import now


class SpeechEvent(str, Enum):
    NONE = "none"
    ONSET = "onset"  # speech started; pre-roll is attached
    CONTINUE = "continue"
    OFFSET = "offset"  # hangover elapsed; utterance is over


@dataclass
class VadResult:
    event: SpeechEvent
    speech: bool
    probability: float
    audio: np.ndarray | None = None  # frames to forward, pre-roll included
    at: float = 0.0


class SileroDetector:
    """Silero VAD over onnxruntime.

    Loaded lazily so that importing this module - which the tests and the
    offline replay tool do - never costs a model load.
    """

    def __init__(self, rate: int = RATE_ASR) -> None:
        if rate not in (8000, 16000):
            raise ValueError("Silero VAD supports 8 kHz or 16 kHz only")
        self.rate = rate
        self._model = None

    def _load(self):  # pragma: no cover - requires the model
        if self._model is not None:
            return self._model
        try:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)
        except ImportError as exc:
            raise RuntimeError(
                "silero-vad is not installed. Run: pip install silero-vad onnxruntime"
            ) from exc
        return self._model

    def probability(self, frame: np.ndarray) -> float:  # pragma: no cover
        import torch

        model = self._load()
        x = torch.from_numpy(frame.astype(np.float32) / 32768.0)
        return float(model(x, self.rate).item())

    def reset(self) -> None:  # pragma: no cover
        if self._model is not None and hasattr(self._model, "reset_states"):
            self._model.reset_states()


class EnergyDetector:
    """RMS-threshold fallback.

    Deliberately kept: it makes every test and the offline replay tool run
    without a model download, and it is a usable degraded mode if onnxruntime
    fails to load on a user's machine. It is NOT good enough for production in
    a noisy room - that is what Silero is for.
    """

    def __init__(self, threshold_dbfs: float = -45.0) -> None:
        self.threshold_dbfs = threshold_dbfs

    def probability(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        f = frame.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(f * f)))
        if rms <= 0:
            return 0.0
        db = 20.0 * np.log10(rms)
        # Map [threshold-10, threshold+10] dB onto [0, 1].
        return float(np.clip((db - self.threshold_dbfs + 10.0) / 20.0, 0.0, 1.0))

    def reset(self) -> None:
        pass


class VoiceActivityGate:
    """Turns a stream of frames into onset / continue / offset events."""

    def __init__(
        self,
        detector=None,
        *,
        rate: int = RATE_ASR,
        threshold: float = 0.5,
        onset_frames: int = 3,
        hangover_ms: float = 400.0,
        preroll_ms: float = 250.0,
        frame_ms: int = 20,
    ) -> None:
        self.detector = detector or EnergyDetector()
        self.rate = rate
        self.threshold = threshold
        self.onset_frames = onset_frames
        self.hangover_ms = hangover_ms
        self.frame_ms = frame_ms

        preroll_frames = max(1, int(round(preroll_ms / frame_ms)))
        self._preroll: deque[np.ndarray] = deque(maxlen=preroll_frames)

        self._speaking = False
        self._run = 0
        self._silence_ms = 0.0

    # ------------------------------------------------------------------

    @property
    def speaking(self) -> bool:
        return self._speaking

    def reset(self) -> None:
        self._preroll.clear()
        self._speaking = False
        self._run = 0
        self._silence_ms = 0.0
        self.detector.reset()

    # ------------------------------------------------------------------

    def push(self, frame: np.ndarray, at: float | None = None) -> VadResult:
        at = at if at is not None else now()
        p = self.detector.probability(frame)
        is_speech = p >= self.threshold

        if not self._speaking:
            self._preroll.append(frame)
            if is_speech:
                self._run += 1
                if self._run >= self.onset_frames:
                    # Onset. Emit the retained pre-roll with the frame, so the
                    # recogniser sees the beginning of the first word.
                    self._speaking = True
                    self._silence_ms = 0.0
                    audio = np.concatenate(list(self._preroll))
                    self._preroll.clear()
                    return VadResult(SpeechEvent.ONSET, True, p, audio, at)
            else:
                self._run = 0
            return VadResult(SpeechEvent.NONE, False, p, None, at)

        # Speaking.
        if is_speech:
            self._silence_ms = 0.0
            return VadResult(SpeechEvent.CONTINUE, True, p, frame, at)

        self._silence_ms += self.frame_ms
        if self._silence_ms >= self.hangover_ms:
            self._speaking = False
            self._run = 0
            self._silence_ms = 0.0
            self.detector.reset()
            # The trailing silence goes too: providers endpoint better with it.
            return VadResult(SpeechEvent.OFFSET, False, p, frame, at)

        # Inside the hangover: still forwarding, still one utterance.
        return VadResult(SpeechEvent.CONTINUE, True, p, frame, at)


def frames_of(pcm: np.ndarray, rate: int = RATE_ASR, frame_ms: int = 20):
    """Split a buffer into fixed frames, discarding any short tail."""
    n = ms_to_samples(frame_ms, rate)
    for i in range(0, len(pcm) - n + 1, n):
        yield pcm[i : i + n]
