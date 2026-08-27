"""OpenAI realtime speech-to-speech translation - Pipeline B (UR -> EN).

Spec reference: §8.2, §8.3, §8.7.

Connects to the dedicated translation endpoint, which is separate from the
voice-agent realtime API:

    wss://api.openai.com/v1/realtime?model=gpt-realtime-translate

Contract, per the provider docs verified during specification:
  * 24 kHz PCM16 in, base64 over the socket
  * ~200 ms PCM16 chunks out, plus target-language transcript deltas
  * target language set via session.audio.output.language
  * one session per output language
  * REQUIRES CONTINUOUS AUDIO, including the silence between phrases -
    there is no response.create and no turn to manage

That last point matters here: the caller must keep feeding, even when VAD says
nobody is speaking. `keepalive_silence` exists for exactly that.

Urdu is a supported INPUT language. It is not one of the 13 output languages,
which is why this class only ever runs in the outgoing direction.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from typing import AsyncIterator

import numpy as np

from app.audio.formats import DTYPE
from app.models.telemetry import now
from app.providers.base import (
    PcmChunk,
    ProviderMisconfigured,
    ProviderRateLimited,
    ProviderUnavailable,
    S2stConfig,
    S2stStream,
    SpeechTranslator,
    TranscriptDelta,
)

ENDPOINT = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-translate"

#: The 13 output languages the model can emit. Urdu is deliberately absent -
#: see §8.2. Keeping this list explicit means a misconfiguration fails at
#: startup with a clear message instead of producing silence in a meeting.
OUTPUT_LANGUAGES = (
    "en", "es", "pt", "fr", "ja", "ru", "zh", "de", "ko", "hi", "id", "vi", "it",
)


class OpenAiS2stStream(S2stStream):
    def __init__(self, ws, rate: int) -> None:
        self._ws = ws
        self._rate = rate
        self._audio: asyncio.Queue[PcmChunk | None] = asyncio.Queue(maxsize=256)
        self._text: asyncio.Queue[TranscriptDelta | None] = asyncio.Queue(maxsize=256)
        self._reader = asyncio.create_task(self._read_loop())
        self._closed = False
        self.last_rtt_ms = 0.0
        self._sent_at = 0.0

    # ------------------------------------------------------------------

    async def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE)
        payload = base64.b64encode(pcm.tobytes()).decode("ascii")
        self._sent_at = now()
        try:
            await self._ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": payload})
            )
        except Exception as exc:  # pragma: no cover - network path
            raise ProviderUnavailable(f"s2st send failed: {exc}") from exc

    async def keepalive_silence(self, ms: int = 20) -> None:
        """Feed silence so the session stays in its streaming contract.

        The model expects continuous audio including gaps. Stopping the feed
        during a pause changes its endpointing behaviour.
        """
        await self.push_audio(np.zeros(self._rate * ms // 1000, dtype=DTYPE))

    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:  # pragma: no cover - network path
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind = msg.get("type", "")

                if kind.endswith("output_audio.delta") or kind == "response.audio.delta":
                    b = msg.get("delta") or msg.get("audio")
                    if b:
                        pcm = np.frombuffer(base64.b64decode(b), dtype=DTYPE)
                        if self._sent_at:
                            self.last_rtt_ms = (now() - self._sent_at) * 1000.0
                        await self._put(self._audio, PcmChunk(pcm, self._rate))

                elif "transcript" in kind and "delta" in kind:
                    text = msg.get("delta", "")
                    if text:
                        await self._put(
                            self._text, TranscriptDelta(text, is_final=False)
                        )

                elif "transcript" in kind and (
                    "done" in kind or "completed" in kind
                ):
                    text = msg.get("transcript", "")
                    if text:
                        await self._put(self._text, TranscriptDelta(text, is_final=True))

                elif kind == "error":
                    err = msg.get("error", {})
                    code = str(err.get("code", ""))
                    detail = err.get("message", raw[:200])
                    if "rate" in code or "quota" in code:
                        raise ProviderRateLimited(detail)
                    raise ProviderUnavailable(detail)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._audio.put_nowait(None)
                self._text.put_nowait(None)

    @staticmethod
    async def _put(q: asyncio.Queue, item) -> None:
        """Bounded put that drops the oldest rather than blocking the reader."""
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover
            with contextlib.suppress(asyncio.QueueEmpty):
                q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(item)

    # ------------------------------------------------------------------

    async def audio_out(self) -> AsyncIterator[PcmChunk]:
        while True:
            item = await self._audio.get()
            if item is None:
                return
            yield item

    async def transcripts(self) -> AsyncIterator[TranscriptDelta]:
        while True:
            item = await self._text.get()
            if item is None:
                return
            yield item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._reader
        with contextlib.suppress(Exception):
            await self._ws.close()


class OpenAiSpeechTranslator(SpeechTranslator):
    name = "openai-realtime-translate"
    output_languages = OUTPUT_LANGUAGES

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.model = model


    async def open(self, cfg: S2stConfig) -> S2stStream:
        if not self.api_key:
            raise ProviderMisconfigured("OPENAI_API_KEY is not set")
        if not self.supports_output(cfg.target_language):
            raise ProviderMisconfigured(
                f"{self.name} cannot emit {cfg.target_language!r}. "
                f"Supported outputs: {', '.join(self.output_languages)}. "
                "Urdu output must use the cascade path (§8.2)."
            )
        if not self.api_key:
            raise ProviderMisconfigured("OPENAI_API_KEY is not set")
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover
            raise ProviderMisconfigured("pip install websockets") from exc


        url = f"{ENDPOINT}?model={cfg.model or self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        try:  # pragma: no cover - network path
            ws = await websockets.connect(url, additional_headers=headers)
        except Exception as exc:
            raise ProviderUnavailable(f"could not open s2st session: {exc}") from exc

        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {"format": {"type": "audio/pcm", "rate": cfg.rate}},
                            "output": {
                                "format": {"type": "audio/pcm", "rate": cfg.rate},
                                "language": cfg.target_language,
                                **({"voice": cfg.voice} if cfg.voice else {}),
                            },
                        }
                    },
                }
            )
        )
        return OpenAiS2stStream(ws, cfg.rate)
