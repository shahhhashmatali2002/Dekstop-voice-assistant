"""Deepgram Nova-3 streaming recognition.

Spec reference: §8.3, §7.7.

Used for English in Pipeline A, and as the fallback Urdu recogniser for
Pipeline B when the speech-to-speech path is unavailable.

Keyterm prompting is the reason this provider is a primary candidate: it takes
the terminology store directly, and §8.5 step 3 expects a large measurable
improvement on exactly the English technical tokens that code-switched Urdu
carries. Measure the delta on the evaluation set before and after.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import AsyncIterator
from urllib.parse import urlencode

import numpy as np

from app.audio.formats import DTYPE
from app.providers.base import (
    AsrConfig,
    AsrStream,
    ProviderMisconfigured,
    ProviderRateLimited,
    ProviderUnavailable,
    SpeechToText,
    TranscriptDelta,
)

ENDPOINT = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"


class DeepgramAsrStream(AsrStream):
    def __init__(self, ws) -> None:
        self._ws = ws
        self._q: asyncio.Queue[TranscriptDelta | None] = asyncio.Queue(maxsize=128)
        self._reader = asyncio.create_task(self._read_loop())
        self._closed = False

    async def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE)
        try:
            await self._ws.send(pcm.tobytes())
        except Exception as exc:  # pragma: no cover
            raise ProviderUnavailable(f"asr send failed: {exc}") from exc

    async def keepalive(self) -> None:
        """Tell the recogniser we are still here.

        Deepgram closes a stream that has received neither audio nor a text
        message for ten seconds, with NET-0001:

            "Deepgram did not receive audio data or a text message within the
             timeout window."

        Every gap in a conversation longer than ten seconds hits that. It must
        be a TEXT frame, not binary - a binary KeepAlive is documented as
        mishandled - which is why this cannot just be a chunk of silence, and
        also why it is better: silence would be transcribed, and billed as
        audio.
        """
        if self._closed:
            return
        try:
            await self._ws.send(json.dumps({"type": "KeepAlive"}))
        except Exception as exc:
            raise ProviderUnavailable(f"asr keepalive failed: {exc}") from exc

    async def _read_loop(self) -> None:  # pragma: no cover - network path
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                if msg.get("type") == "Error":
                    detail = msg.get("description", "")
                    if "rate" in detail.lower() or "quota" in detail.lower():
                        raise ProviderRateLimited(detail)
                    raise ProviderUnavailable(detail)

                channel = msg.get("channel") or {}
                alts = channel.get("alternatives") or []
                if not alts:
                    continue
                text = (alts[0].get("transcript") or "").strip()
                if not text:
                    continue

                speaker = None
                words = alts[0].get("words") or []
                if words and "speaker" in words[0]:
                    speaker = f"speaker {words[0]['speaker']}"

                await self._put(
                    TranscriptDelta(
                        text=text,
                        is_final=bool(msg.get("is_final") or msg.get("speech_final")),
                        speaker=speaker,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._q.put_nowait(None)

    async def _put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover
            # Hypotheses are superseded anyway - drop the oldest (§6.4).
            with contextlib.suppress(asyncio.QueueEmpty):
                self._q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._q.put_nowait(item)

    async def hypotheses(self) -> AsyncIterator[TranscriptDelta]:
        while True:
            item = await self._q.get()
            if item is None:
                return
            yield item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"type": "CloseStream"}))
        self._reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._reader
        with contextlib.suppress(Exception):
            await self._ws.close()


class DeepgramSpeechToText(SpeechToText):
    name = "deepgram-nova-3"

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.model = model

    def _url(self, cfg: AsrConfig) -> str:
        params = {
            "model": cfg.model or self.model,
            "language": cfg.language,
            "encoding": "linear16",
            "sample_rate": str(cfg.rate),
            "channels": "1",
            "interim_results": "true" if cfg.interim_results else "false",
            "punctuate": "true",
            "smart_format": "true",
            "endpointing": "300",
        }
        query = urlencode(params)
        # Keyterms repeat as separate parameters (§7.7).
        for term in cfg.keyterms[:100]:
            query += "&" + urlencode({"keyterm": term})
        return f"{ENDPOINT}?{query}"

    async def open(self, cfg: AsrConfig) -> AsrStream:
        if not self.api_key:
            raise ProviderMisconfigured("DEEPGRAM_API_KEY is not set")
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover
            raise ProviderMisconfigured("pip install websockets") from exc
        try:  # pragma: no cover - network path
            ws = await websockets.connect(
                self._url(cfg),
                additional_headers={"Authorization": f"Token {self.api_key}"},
            )
        except Exception as exc:
            raise ProviderUnavailable(f"could not open ASR stream: {exc}") from exc
        return DeepgramAsrStream(ws)
