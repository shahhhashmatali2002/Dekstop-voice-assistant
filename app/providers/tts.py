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
import os
from typing import AsyncIterator

import numpy as np

from app.audio.formats import DTYPE
from app.providers.urdu_speech import close, pronounceable
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


def speakable(text: str, *, final: bool = True) -> str:
    """Tidy a line so a neural voice reads it as speech.

    Three things matter to how fluent the result sounds, and none of them is
    which voice was chosen:

    The marks a text renderer treats as decoration are instructions to a
    speech engine: a stray quote or bracket becomes a pause in the middle of
    a phrase.

    An English word left in Latin script inside an Urdu sentence is a word the
    voice has to guess at, and it guesses badly. `pronounceable` says the ones
    we know the way an Urdu speaker says them - see app/providers/urdu_speech.

    And a clause needs an ending, because a fragment with no final mark is read
    flat. Which ending depends on whether the thought is finished: "۔" makes
    the voice fall and stop, so putting one after every clause of a sentence is
    the robotic reading described in urdu_speech. `final` should be false unless
    the utterance really ended.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    clean = clean.strip("\"'\u201c\u201d\u2018\u2019()[]")
    return close(pronounceable(clean), final=final)


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def rate_percent(factor: float) -> str:
    """A speed factor as the percentage both Azure and Edge expect.

    1.08 -> "+8%". Used to ask the VOICE to speak faster when playback is
    behind, which is what a person does, instead of time-stretching audio the
    voice has already produced - see §7.6 and the note in incoming._speak.
    """
    return f"{int(round((factor - 1.0) * 100)):+d}%"


def build_ssml(
    text: str, voice: VoiceId, locale: str = "ur-PK", rate: float = 1.0
) -> str:
    """SSML for one utterance.

    Kept as a free function so it can be unit-tested without a network or an
    API key - escaping bugs here surface as silent synthesis failures.
    """
    style_open, style_close = "", ""
    if voice.style:
        style_open = f'<mstts:express-as style="{_xml_escape(voice.style)}">'
        style_close = "</mstts:express-as>"
    if rate > 1.0001 or rate < 0.9999:
        style_open += f'<prosody rate="{rate_percent(rate)}">'
        style_close = "</prosody>" + style_close
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


class SpeedHint:
    """Lets the pipeline ask the VOICE to hurry instead of stretching audio.

    When playback falls behind, §7.6 speeds the output up. There are two ways
    to do that and they do not sound the same. Time-stretching PCM the voice
    has already produced is done per chunk, on a few hundred milliseconds at a
    time, with no state carried across the seam - and every seam is a small
    discontinuity. A dozen of those a second is the metallic edge users
    describe as robotic.

    The alternative is to tell the voice to speak faster before it says
    anything. Azure, Edge and ElevenLabs all accept a speed with the request,
    and a neural voice reading faster sounds like a person reading faster:
    no seams, no pitch shift, nothing to repair afterwards.

    The cap is deliberate. Past about 1.2x a voice stops sounding hurried and
    starts sounding comic, and dropping a unit (§7.6) is the honest response to
    being that far behind.
    """

    supports_speed = True
    MAX_SPEED = 1.20
    _speed = 1.0

    def set_speed(self, factor: float) -> float:
        """Speed for the NEXT utterance. Returns what was actually taken."""
        try:
            wanted = float(factor)
        except (TypeError, ValueError):
            wanted = 1.0
        self._speed = min(max(1.0, wanted), self.MAX_SPEED)
        return self._speed

    @property
    def speed(self) -> float:
        return self._speed


# ----------------------------------------------------------------------


class AzureTextToSpeech(SpeedHint, TextToSpeech):
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
        body = build_ssml(speakable(text), voice, rate=self._speed).encode(
            "utf-8"
        )
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


class ElevenLabsTextToSpeech(SpeedHint, TextToSpeech):
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
            "text": speakable(text),
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": round(self._speed, 2),
            },
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


class EdgeTextToSpeech(SpeedHint, TextToSpeech):
    """Free neural synthesis via Edge TTS, decoded as it arrives.

    WHY THIS DECODES INCREMENTALLY
    ------------------------------
    Edge TTS streams MP3 over a websocket while it is still synthesising. The
    obvious implementation - collect every chunk, decode once at the end -
    throws that away: nothing is audible until the whole sentence has been
    generated AND transferred, so a five-second sentence is five seconds of
    silence followed by five seconds of speech. Time to first sound is the
    number a listener actually experiences, and it becomes the length of the
    sentence rather than the length of the network round trip.

    MP3 is a sequence of self-contained frames, so a partial stream decodes
    fine. We decode what has arrived, emit only the samples we have not
    emitted yet, and hold back a frame's worth at the boundary - the last
    frame of a truncated buffer can decode differently once its remainder
    arrives, and re-emitting a corrected sample is a click.
    """

    name = "edge-tts"

    #: Decode once this much new MP3 has arrived. At Edge's 48 kbit/s that is
    #: roughly 300 ms of audio: often enough to keep the sink fed, rarely
    #: enough to spend real time re-decoding.
    DECODE_EVERY_BYTES = 2_000

    #: Samples held back from a partial decode, comfortably over one MP3
    #: frame (1152 samples).
    TAIL_GUARD = 2_304

    #: Spoken a little under the default. Edge's neural voices read at a
    #: brisk pace that is fine for a caption being read aloud and hurried for
    #: a language the listener is relying on to follow a conversation.
    SPEAKING_RATE = "-8%"

    def __init__(self, rate: int = 24_000, speaking_rate: str | None = None) -> None:
        self.rate = rate
        self.speaking_rate = speaking_rate or self.SPEAKING_RATE
        self._speed = 1.0

    @property
    def base_factor(self) -> float:
        """`speaking_rate` as a multiplier. "-8%" is 0.92."""
        try:
            return 1.0 + int(self.speaking_rate.strip().rstrip("%")) / 100.0
        except (AttributeError, ValueError):
            return 1.0

    @property
    def request_rate(self) -> str:
        """The rate sent with the request: the base pace, sped up if asked.

        Kept as one number rather than two adjustments because Edge applies a
        single rate; asking for -8% and then separately for +10% would mean
        one of them silently winning.
        """
        return rate_percent(self.base_factor * self._speed)

    def _decode(self, mp3: bytes):
        """Seam: overridden in tests so the streaming logic can be exercised."""
        try:
            import miniaudio
        except ImportError as exc:
            raise ProviderMisconfigured("pip install miniaudio") from exc

        decoded = miniaudio.decode(mp3, nchannels=1, sample_rate=self.rate)
        return np.frombuffer(decoded.samples, dtype=DTYPE)

    async def _stream_mp3(self, text: str, voice_name: str):
        """The network seam, overridden in tests. Receives the line already
        prepared for speech - see `synthesize` - so that a fake network still
        sees exactly what the real one would."""
        try:
            import edge_tts
        except ImportError as exc:
            raise ProviderMisconfigured("pip install edge-tts") from exc

        comm = edge_tts.Communicate(text, voice_name, rate=self.request_rate)
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize(
        self, text: str, voice: VoiceId
    ) -> AsyncIterator[PcmChunk]:
        if not text.strip():
            return
        voice_name = (
            voice.name
            if (voice and voice.name and voice.name != "azure")
            else "ur-PK-UzmaNeural"
        )
        line = speakable(text)
        if not line:
            return
        try:
            mp3 = b""
            decoded_at = 0
            emitted = 0

            async for data in self._stream_mp3(line, voice_name):
                mp3 += data
                if len(mp3) - decoded_at < self.DECODE_EVERY_BYTES:
                    continue
                decoded_at = len(mp3)
                try:
                    samples = self._decode(mp3)
                except Exception:
                    continue  # not a frame boundary yet; wait for more
                safe = max(0, samples.size - self.TAIL_GUARD)
                if safe > emitted:
                    yield PcmChunk(samples[emitted:safe].copy(), self.rate)
                    emitted = safe

            if not mp3:
                return
            samples = self._decode(mp3)
            if samples.size > emitted:
                yield PcmChunk(samples[emitted:].copy(), self.rate)
            yield PcmChunk(np.empty(0, dtype=DTYPE), self.rate, final=True)

        except asyncio.CancelledError:
            raise
        except ProviderMisconfigured:
            raise
        except Exception as exc:
            raise ProviderUnavailable(f"Edge TTS failed: {exc}") from exc


