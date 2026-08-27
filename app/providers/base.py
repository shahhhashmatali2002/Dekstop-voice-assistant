"""Provider interfaces.

Spec reference: §8.6, §8.2.

Every provider sits behind an abstract base class. Nothing in the pipeline
knows a vendor name; provider choice is configuration, and swapping one is a
config change plus one new module.

THE ASYMMETRY (§8.2)
--------------------
Purpose-built real-time speech translation models accept a very wide set of
input languages but emit a much narrower set of output languages. Urdu is an
accepted *input* for the realtime translation models, and English is one of
their supported outputs - but Urdu is NOT an available output.

    Pipeline B, outgoing (UR -> EN):  one speech-to-speech model.
    Pipeline A, incoming (EN -> UR):  cascade, ASR -> translation -> TTS.

The two directions have different architectures, different latency profiles
and different failure modes. This is not a temporary state to be tidied away;
`SpeechTranslator` below is deliberately shaped like `SpeechToText` so the
orchestration layer can be configured either way without branching on vendor.

CANCELLATION
------------
Handled by asyncio task cancellation, not an explicit token. Every
implementation must therefore be cancellation-safe: a cancelled task closes
its socket cleanly rather than leaving a half-open connection for the
reconnect logic to fight with (§6.5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

from app.models.telemetry import now


# ----------------------------------------------------------------------
# value types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PcmChunk:
    """A chunk of synthesised audio."""

    pcm: np.ndarray
    rate: int
    at: float = field(default_factory=now)
    final: bool = False

    @property
    def duration_ms(self) -> float:
        return len(self.pcm) * 1000.0 / self.rate


@dataclass(frozen=True)
class TranscriptDelta:
    """Incremental transcript text from a provider that emits one."""

    text: str
    is_final: bool = False
    language: str | None = None
    speaker: str | None = None
    at: float = field(default_factory=now)


@dataclass
class AsrConfig:
    language: str  # "en", "ur"
    rate: int
    interim_results: bool = True
    keyterms: list[str] = field(default_factory=list)  # §7.7 keyterm prompting
    model: str | None = None
    region: str | None = None


@dataclass
class S2stConfig:
    source_language: str | None  # None = let the model detect
    target_language: str  # must be one of the model's output languages
    rate: int = 24_000
    model: str | None = None
    voice: str | None = None


@dataclass
class VoiceId:
    vendor: str
    name: str  # e.g. "ur-PK-UzmaNeural"
    rate: int = 24_000
    style: str | None = None


@dataclass
class TranslationUnit:
    text: str
    source_language: str
    target_language: str
    speaker: str | None = None
    prefer_brevity: bool = False  # set by the drift controller (§7.6)


@dataclass
class TranslatedUnit:
    text: str
    source: TranslationUnit
    at: float = field(default_factory=now)


class RollingContext:
    """The last few exchanges, in both directions.

    Spec reference: §8.4.1. Translating each clause in isolation produces
    stilted and sometimes wrong output. Keep 6-10 exchanges, not the whole
    meeting: cost and latency scale with the window, relevance does not.
    """

    def __init__(self, max_turns: int = 8, max_chars: int = 1200) -> None:
        self.max_turns = max_turns
        self.max_chars = max_chars
        self._turns: list[tuple[str, str]] = []

    def add(self, speaker: str, text: str) -> None:
        if not text.strip():
            return
        self._turns.append((speaker, text.strip()))
        del self._turns[: max(0, len(self._turns) - self.max_turns)]

    def render(self) -> str:
        out: list[str] = []
        total = 0
        for speaker, text in reversed(self._turns):
            line = f"{speaker}: {text}"
            if total + len(line) > self.max_chars:
                break
            out.append(line)
            total += len(line)
        return "\n".join(reversed(out))

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------


class ProviderError(RuntimeError):
    """Base class. Carries whether a retry could plausibly succeed."""

    retryable = True


class ProviderUnavailable(ProviderError):
    """Connection failed or dropped. Reconnect with backoff."""


class ProviderRateLimited(ProviderError):
    """Quota or rate limit. Do NOT retry into the limit (§14.2)."""

    retryable = False


class ProviderMisconfigured(ProviderError):
    """Bad credentials, unsupported language, wrong model. Retrying is futile."""

    retryable = False


# ----------------------------------------------------------------------
# interfaces
# ----------------------------------------------------------------------


class AsrStream(ABC):
    """One open streaming-recognition session."""

    @abstractmethod
    async def push_audio(self, pcm: np.ndarray) -> None:
        """Send audio. Must not block the caller for network time."""

    @abstractmethod
    def hypotheses(self) -> AsyncIterator[TranscriptDelta]:
        """Partial and final hypotheses, in order."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close cleanly. Must be safe to call twice, and from a cancelled task."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


class SpeechToText(ABC):
    name: str = "asr"

    @abstractmethod
    async def open(self, cfg: AsrConfig) -> AsrStream: ...


class Translator(ABC):
    name: str = "mt"

    @abstractmethod
    async def translate(
        self,
        unit: TranslationUnit,
        ctx: RollingContext,
        terms: "Terminology | None" = None,
    ) -> TranslatedUnit: ...


class TextToSpeech(ABC):
    """Streaming synthesis: the first chunk matters far more than total time."""

    name: str = "tts"

    @abstractmethod
    def synthesize(self, text: str, voice: VoiceId) -> AsyncIterator[PcmChunk]: ...


class S2stStream(ABC):
    """One open speech-to-speech translation session."""

    @abstractmethod
    async def push_audio(self, pcm: np.ndarray) -> None: ...

    @abstractmethod
    def audio_out(self) -> AsyncIterator[PcmChunk]: ...

    @abstractmethod
    def transcripts(self) -> AsyncIterator[TranscriptDelta]: ...

    @abstractmethod
    async def aclose(self) -> None: ...

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()


class SpeechTranslator(ABC):
    name: str = "s2st"

    #: Output languages the model can actually emit. Urdu is deliberately
    #: absent from every current implementation - see the module docstring.
    output_languages: tuple[str, ...] = ()

    def supports_output(self, language: str) -> bool:
        return language.lower() in self.output_languages

    @abstractmethod
    async def open(self, cfg: S2stConfig) -> S2stStream: ...


# Imported late to avoid a cycle: Terminology needs nothing from here.
from app.models.terminology import Terminology  # noqa: E402
