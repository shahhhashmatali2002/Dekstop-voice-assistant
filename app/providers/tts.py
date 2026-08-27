"""Streaming Urdu synthesis - the last stage of Pipeline A.

Spec reference: §8.3, §5.7, §7.6.

Two implementations, because §18 O5 is an open decision that must be settled
by ear rather than by spec sheet:

  * Azure Neural TTS  - ur-PK-UzmaNeural (female), ur-PK-AsadNeural (male).
    Correct locale for a Pakistani listener.
  * ElevenLabs Flash v2.5 - ~75 ms model latency, 32 languages including Urdu.
    Attractive latency; naturalness must be judged by native listeners.

WHAT MATTERS HERE IS TIME TO FIRST CHUNK, NOT TOTAL TIME
--------------------------------------------------------
Playback starts on the first chunk. A voice that takes 400 ms to produce its
first 200 ms of audio beats one that takes 250 ms to produce all of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import AsyncIterator

import numpy as np

from app.audio.formats import DTYPE
from app.providers.base import (
    PcmChunk,
    ProviderMisconfigured,
    ProviderRateLimited,
    ProviderUnavailable,
    TextToSpeech,
    VoiceId,
)

# Verified during specification (§8.3).
AZURE_URDU_VOICES = (
    "ur-PK-UzmaNeural",
    "ur-PK-AsadNeural",
    "ur-IN-GulNeural",
    "ur-IN-SalmanNeural",
)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_ssml(text: str, voice: VoiceId, locale: str = "ur-PK") -> str:
    """SSML for one utterance.

    Kept as a free function so it can be unit-tested without a network or an
    API key - escaping bugs here surface as silent synthesis failures.
    """
    style_open, style_close = "", ""
    if voice.style:
        style_open = f'<mstts:express-as style="{_xml_escape(voice.style)}">'
        style_close = "</mstts:express-as>"
    return (
        '<speak version="1.0" '
        'xmlns="http://www.w3.org/2001/10/synthesis" '
        'xmlns:mstts="http://www.w3.org/2001/mstts" '
        f'xml:lang="{locale}">'
        f'<voice name="{_xml_escape(voice.name)}">'
        f"{style_open}{_xml_escape(text)}{style_close}"
        "</voice></speak>"
    )


# ----------------------------------------------------------------------


class AzureTextToSpeech(TextToSpeech):
    """Azure Neural TTS over the streaming REST endpoint."""

    name = "azure-neural-tts"

    def __init__(
        self,
        api_key: str | None = None,
        region: str | None = None,
        rate: int = 24_000,
        timeout_s: float = 6.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("AZURE_SPEECH_KEY", "")
        self.region = region or os.environ.get("AZURE_SPEECH_REGION", "")
        self.rate = rate
        self.timeout_s = timeout_s

    @property
    def _url(self) -> str:
        return f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    @property
    def _output_format(self) -> str:
        return f"raw-{self.rate // 1000}khz-16bit-mono-pcm"

    async def synthesize(  # pragma: no cover - network path
        self, text: str, voice: VoiceId
    ) -> AsyncIterator[PcmChunk]:
        if not text.strip():
            return
        if not self.api_key or not self.region:
            raise ProviderMisconfigured(
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set"
            )
        try:
            import aiohttp
        except ImportError as exc:
            raise ProviderMisconfigured("pip install aiohttp") from exc

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": self._output_format,
            "User-Agent": "voice-assistant",
        }
        body = build_ssml(text, voice).encode("utf-8")
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self._url, headers=headers, data=body) as resp:
                    if resp.status == 429:
                        raise ProviderRateLimited(await resp.text())
                    if resp.status == 401:
                        raise ProviderMisconfigured("Azure Speech key rejected")
                    if resp.status >= 400:
                        raise ProviderUnavailable(
                            f"Azure TTS {resp.status}: {await resp.text()}"
                        )
                    tail = b""
                    async for raw in resp.content.iter_chunked(4096):
                        buf = tail + raw
                        usable = len(buf) - (len(buf) % 2)  # int16 alignment
                        tail = buf[usable:]
                        if usable:
                            yield PcmChunk(
                                np.frombuffer(buf[:usable], dtype=DTYPE), self.rate
                            )
            yield PcmChunk(np.empty(0, dtype=DTYPE), self.rate, final=True)
        except (ProviderUnavailable, ProviderRateLimited, ProviderMisconfigured):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"Azure TTS failed: {exc}") from exc


# ----------------------------------------------------------------------


class ElevenLabsTextToSpeech(TextToSpeech):
    """ElevenLabs streaming synthesis. Flash v2.5 for the latency profile."""

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "eleven_flash_v2_5",
        rate: int = 24_000,
        timeout_s: float = 6.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.model = model
        self.rate = rate
        self.timeout_s = timeout_s

    async def synthesize(  # pragma: no cover - network path
        self, text: str, voice: VoiceId
    ) -> AsyncIterator[PcmChunk]:
        if not text.strip():
            return
        if not self.api_key:
            raise ProviderMisconfigured("ELEVENLABS_API_KEY is not set")
        try:
            import aiohttp
        except ImportError as exc:
            raise ProviderMisconfigured("pip install aiohttp") from exc

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice.name}/stream"
            f"?output_format=pcm_{self.rate}"
        )
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        raise ProviderRateLimited(await resp.text())
                    if resp.status == 401:
                        raise ProviderMisconfigured("ElevenLabs key rejected")
                    if resp.status >= 400:
                        raise ProviderUnavailable(
                            f"ElevenLabs {resp.status}: {await resp.text()}"
                        )
                    tail = b""
                    async for raw in resp.content.iter_chunked(4096):
                        buf = tail + raw
                        usable = len(buf) - (len(buf) % 2)
                        tail = buf[usable:]
                        if usable:
                            yield PcmChunk(
                                np.frombuffer(buf[:usable], dtype=DTYPE), self.rate
                            )
            yield PcmChunk(np.empty(0, dtype=DTYPE), self.rate, final=True)
        except (ProviderUnavailable, ProviderRateLimited, ProviderMisconfigured):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"ElevenLabs TTS failed: {exc}") from exc


# ----------------------------------------------------------------------


class NullTextToSpeech(TextToSpeech):
    """Silent synthesis at a realistic duration.

    Not a toy: it is what `tools/replay.py` uses to measure the pipeline
    end-to-end without spending provider credits, and what the degraded
    text-only mode falls back to (§14.2).
    """

    name = "null-tts"

    def __init__(self, rate: int = 24_000, chars_per_second: float = 14.0) -> None:
        self.rate = rate
        self.chars_per_second = chars_per_second

    async def synthesize(self, text: str, voice: VoiceId) -> AsyncIterator[PcmChunk]:
        if not text.strip():
            return
        total = int(self.rate * len(text) / self.chars_per_second)
        chunk = self.rate // 5  # 200 ms, matching the realtime models
        emitted = 0
        while emitted < total:
            n = min(chunk, total - emitted)
            await asyncio.sleep(0)
            yield PcmChunk(np.zeros(n, dtype=DTYPE), self.rate)
            emitted += n
        yield PcmChunk(np.empty(0, dtype=DTYPE), self.rate, final=True)


# ----------------------------------------------------------------------


class EdgeTextToSpeech(TextToSpeech):
    """Free Neural TTS synthesis via Edge TTS (Microsoft Neural Voices)."""

    name = "edge-tts"

    def __init__(self, rate: int = 24_000) -> None:
        self.rate = rate

    async def synthesize(
        self, text: str, voice: VoiceId
    ) -> AsyncIterator[PcmChunk]:
        if not text.strip():
            return
        try:
            import edge_tts
            import miniaudio
        except ImportError as exc:
            raise ProviderMisconfigured("pip install edge-tts miniaudio") from exc

        voice_name = voice.name if (voice and voice.name and voice.name != "azure") else "ur-PK-UzmaNeural"
        try:
            comm = edge_tts.Communicate(text, voice_name)
            mp3 = b""
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    mp3 += chunk["data"]
            if not mp3:
                return
            decoded = miniaudio.decode(mp3, nchannels=1, sample_rate=self.rate)
            samples = np.frombuffer(decoded.samples, dtype=DTYPE)

            chunk_size = self.rate // 5
            for i in range(0, len(samples), chunk_size):
                chunk_pcm = samples[i : i + chunk_size]
                yield PcmChunk(chunk_pcm, self.rate)
                await asyncio.sleep(0.005)
            yield PcmChunk(np.empty(0, dtype=DTYPE), self.rate, final=True)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"Edge TTS failed: {exc}") from exc


