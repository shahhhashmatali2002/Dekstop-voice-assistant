"""Streaming Urdu synthesis - the last stage of Pipeline A.

Spec reference: §8.3, §5.7, §7.6.

Two implementations, because §18 O5 is an open decision that must be settled
by ear rather than by spec sheet:

  * Azure Neural TTS  - ur-PK-UzmaNeural (female), ur-PK-AsadNeural (male).
    Correct locale for a Pakistani listener.
  * ElevenLabs - Flash v2.5 is ~75 ms to first byte across 32 languages, and
    NOT one of them is Urdu; Urdu lives in the v3 family, which is slower
    (~280 ms) and more expressive. So the model is chosen from the language
    rather than pinned - see ELEVEN_MODELS - and Flash keeps the outgoing
    English direction, where time to first byte is the whole game, while Urdu
    goes to v3. Professional Voice Cloning is aligned to Flash v2.5's list, so
    a cloned Urdu voice is still not available.

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
from app.providers.urdu_speech import close, pronounceable
from app.providers.base import (
    PcmChunk,
    ProviderMisconfigured,
    ProviderRateLimited,
    ProviderUnavailable,
    TextToSpeech,
    VoiceId,
)

#: xml:lang for the SSML wrapper. Handing an English sentence to a voice
#: inside xml:lang="ur-PK" is another way to get an accent nobody asked for.
LOCALES = {"ur": "ur-PK", "en": "en-US"}

#: Free fallbacks, one per direction. The Urdu one is what shipped; the
#: English one exists because Edge is now also the safety net UNDER
#: ElevenLabs for the outgoing direction, and defaulting that to a Pakistani
#: Urdu voice reading English is exactly the accent complaint.
DEFAULT_URDU_EDGE_VOICE = "ur-PK-UzmaNeural"
DEFAULT_ENGLISH_EDGE_VOICE = "en-US-BrianNeural"

# Verified during specification (§8.3).
AZURE_URDU_VOICES = (
    "ur-PK-UzmaNeural",
    "ur-PK-AsadNeural",
    "ur-IN-GulNeural",
    "ur-IN-SalmanNeural",
)


#: Latin-script endings, for text that is going to be read in a Latin-script
#: language. The Urdu equivalents live in app/providers/urdu_speech.
_LATIN_TERMINAL = ".!?\u2026"
_LATIN_CONTINUATION = ",;:"


def speakable(text: str, *, final: bool = True, language: str = "ur") -> str:
    """Tidy a line so a neural voice reads it as speech.

    Three things matter to how fluent the result sounds, and none of them is
    which voice was chosen:

    The marks a text renderer treats as decoration are instructions to a
    speech engine: a stray quote or bracket becomes a pause in the middle of
    a phrase.

    An English word left in Latin script inside an URDU sentence is a word the
    Urdu voice has to guess at, and it guesses badly. `pronounceable` says the
    ones we know the way an Urdu speaker says them - see
    app/providers/urdu_speech.

    And a clause needs an ending, because a fragment with no final mark is read
    flat. Which ending depends on whether the thought is finished: the Urdu
    full stop makes the voice fall and stop, so putting one after every clause
    of a sentence is the robotic reading described in urdu_speech. `final`
    should be false unless the utterance really ended.

    `language` IS NOT OPTIONAL DECORATION
    -------------------------------------
    This function used to apply the Urdu preparation to everything it was
    given, including the English the client hears on the outgoing direction.
    That is not a cosmetic mistake. `pronounceable` rewrites a word it knows
    into Urdu script, so a good English sentence reached the English voice as

        "I work at Macrosoft and I am a developer using the API"

    turned into a line where "I", "a", "developer" and "API" had all been
    replaced by Urdu spellings. An English voice handed Urdu script does what
    any voice does with letters it cannot read - it approximates, or spells,
    or skips - and four wrong words in a fourteen-word sentence is heard,
    correctly, as the accent being bad. The Urdu preparation now runs only for
    Urdu.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    clean = clean.strip("\"'\u201c\u201d\u2018\u2019()[]")
    if not clean:
        return ""

    if (language or "ur").lower().startswith("ur"):
        return close(pronounceable(clean), final=final)

    # Latin-script languages: the same "give the clause an ending" rule, in
    # the punctuation the voice was actually trained on.
    if clean[-1] in _LATIN_TERMINAL or clean[-1] in _LATIN_CONTINUATION:
        return clean
    return clean + ("." if final else ",")


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
        language = (voice.language or "ur").lower()
        locale = LOCALES.get(language.split("-")[0], "ur-PK")
        body = build_ssml(
            speakable(text, language=language),
            voice,
            locale=locale,
            rate=self._speed,
        ).encode(
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


#: Which ElevenLabs model can actually speak which language. This is not a
#: quality preference, it is a capability table, and getting it wrong is
#: silent: Flash v2.5 covers 32 languages and Urdu is not one of them, so
#: Urdu sent to Flash comes back as an English voice sounding out Urdu script.
#: Urdu lives in the v3 family. English stays on Flash, where ~75 ms to first
#: byte is worth more than expressiveness on a business call.
ELEVEN_MODELS = {
    "ur": os.environ.get("ELEVEN_URDU_MODEL") or "eleven_v3",
    "en": os.environ.get("ELEVEN_ENGLISH_MODEL") or "eleven_flash_v2_5",
}
ELEVEN_FALLBACK_MODEL = "eleven_multilingual_v2"

#: v3 rejects a free-floating stability; it takes one of three settings.
V3_STABILITY = (0.0, 0.5, 1.0)

#: Words in a voice's own labels and description that mean it will not sound
#: like a foreigner reading Urdu. Accent is the whole complaint here: any
#: ElevenLabs voice CAN pronounce Urdu under v3, and a Midwestern American
#: voice doing it is precisely what "not natural" means.
SOUTH_ASIAN_HINTS = (
    "urdu", "pakistan", "pakistani", "hindi", "india", "indian",
    "south asian", "desi", "punjabi",
)


def _is_elevenlabs_id(name: str) -> bool:
    """Does this look like a voice ID rather than a platform voice name?

    ElevenLabs IDs are opaque 20-ish character alphanumerics. Azure and Edge
    names are locale-prefixed and contain hyphens: "ur-PK-UzmaNeural". The
    config file holds one of those by default, and handing it to ElevenLabs as
    a voice ID is a 404 at the worst possible moment.
    """
    name = (name or "").strip()
    return bool(name) and name.isalnum() and 15 <= len(name) <= 40


class ElevenLabsTextToSpeech(SpeedHint, TextToSpeech):
    """ElevenLabs streaming synthesis, told which language it is speaking.

    Three things this does that the first version did not, all of them
    audible:

      * picks the MODEL from the language, because Urdu is not in Flash v2.5
      * picks the VOICE from the account when it was handed an Azure/Edge
        name, and prefers one whose own labels say it is South Asian
      * prepares the text for the language it is about to be read in, rather
        than transliterating English into Urdu script (see `speakable`)
    """

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        rate: int = 24_000,
        timeout_s: float = 6.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        #: An explicit model overrides the capability table, for testing a
        #: new model without a code change. Normally None.
        self.model = model
        self.rate = rate
        self.timeout_s = timeout_s
        self._discovered: dict[str, str] = {}

    # ------------------------------------------------------------------

    def model_for(self, language: str) -> str:
        if self.model:
            return self.model
        return ELEVEN_MODELS.get(
            (language or "en").split("-")[0].lower(), ELEVEN_FALLBACK_MODEL
        )

    def settings_for(self, model_id: str) -> dict:
        """Voice settings the chosen model will actually accept.

        Deliberately the settings that were already working, not better ones.
        Every extra field here is a field the endpoint can reject with a 422,
        and a 422 is the client hearing silence - so `style` and
        `use_speaker_boost` are NOT sent on a guess. The only difference from
        the original is v3, which takes one of three fixed stabilities rather
        than a continuous value and has no speed control.
        """
        if "_v3" in model_id:
            return {
                "stability": min(V3_STABILITY, key=lambda v: abs(v - 0.5)),
                "similarity_boost": 0.75,
            }
        return {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "speed": round(self._speed, 2),
        }

    async def voice_for(self, voice: VoiceId) -> str:
        """The ElevenLabs voice ID to speak with.

        In order: a real ID on the VoiceId, an ID pinned in the environment,
        one discovered from the account, and finally whatever the account's
        first voice is - because a voice with the wrong accent still says the
        sentence, and no voice says nothing.
        """
        if _is_elevenlabs_id(voice.name):
            return voice.name

        language = (getattr(voice, "language", None) or "en").split("-")[0].lower()
        pinned = os.environ.get(f"ELEVEN_{language.upper()}_VOICE_ID") or (
            os.environ.get("ELEVENLABS_VOICE_ID") if language == "en" else None
        )
        if pinned:
            return pinned
        if language in self._discovered:
            return self._discovered[language]

        found = await self._discover(language)
        if not found:
            raise ProviderMisconfigured(
                f"no ElevenLabs voice available for {language!r}; set "
                f"ELEVEN_{language.upper()}_VOICE_ID in .env "
                f"(run tools/check_voices.py to list them)"
            )
        self._discovered[language] = found
        return found

    async def _discover(self, language: str) -> str:
        """Ask the account what voices it has and pick the best fit."""
        try:
            import aiohttp
        except ImportError as exc:
            raise ProviderMisconfigured("pip install aiohttp") from exc

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    "https://api.elevenlabs.io/v2/voices?page_size=100",
                    headers={"xi-api-key": self.api_key},
                ) as resp:
                    if resp.status == 401:
                        raise ProviderMisconfigured("ElevenLabs key rejected")
                    if resp.status >= 400:
                        return ""
                    data = await resp.json()
        except ProviderMisconfigured:
            raise
        except Exception:
            return ""

        voices = data.get("voices") or []
        if not voices:
            return ""
        if language.startswith("ur"):
            for entry in voices:
                haystack = " ".join(
                    str(part).lower()
                    for part in (
                        entry.get("name", ""),
                        entry.get("description", ""),
                        *(entry.get("labels") or {}).values(),
                        *(entry.get("verified_languages") or []),
                    )
                )
                if any(hint in haystack for hint in SOUTH_ASIAN_HINTS):
                    return entry.get("voice_id", "")
        return voices[0].get("voice_id", "")

    # ------------------------------------------------------------------

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

        language = (getattr(voice, "language", None) or "en").lower()
        model_id = self.model_for(language)
        voice_id = await self.voice_for(voice)
        line = speakable(text, language=language)
        if not line:
            return

        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
            f"?output_format=pcm_{self.rate}"
        )
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "text": line,
            "model_id": model_id,
            "voice_settings": self.settings_for(model_id),
        }
        timeout = aiohttp.ClientTimeout(
            total=4.5, connect=2.0, sock_read=2.5
        )

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 429:
                        raise ProviderRateLimited(await resp.text())
                    if resp.status == 401:
                        raise ProviderMisconfigured("ElevenLabs key rejected")
                    if resp.status >= 400:
                        raise ProviderUnavailable(
                            f"ElevenLabs {resp.status} ({model_id}, {language}): "
                            f"{await resp.text()}"
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
            # The type, always. An exception whose str() is empty - a timeout,
            # a dropped payload - otherwise reaches the console as
            # "ElevenLabs TTS failed:" and says nothing about what went wrong.
            raise ProviderUnavailable(
                f"ElevenLabs TTS failed: {type(exc).__name__}: {exc}"
                f" (model {model_id}, voice {voice_id})"
            ) from exc


# ----------------------------------------------------------------------


class FallbackTextToSpeech(SpeedHint, TextToSpeech):
    """One voice, with another one behind it.

    The incoming direction had no safety net: if synthesis failed, the
    listener got silence and a warning in a window they are not looking at,
    which is the failure mode the spec calls out as the worst one available.
    Now a paid voice that is down, out of quota, or refusing a language falls
    through to the free one MID-UTTERANCE - as long as nothing has been
    emitted yet, because a sentence half in one voice and half in another is
    worse than either.
    """

    name = "fallback-tts"

    def __init__(self, primary: TextToSpeech, backup: TextToSpeech,
                 on_notice=None) -> None:
        self.primary = primary
        self.backup = backup
        self.on_notice = on_notice
        self._demoted = False

    @property
    def active(self) -> TextToSpeech:
        return self.backup if self._demoted else self.primary

    def set_speed(self, factor: float) -> float:
        for engine in (self.primary, self.backup):
            if getattr(engine, "supports_speed", False):
                with contextlib.suppress(Exception):
                    engine.set_speed(factor)
        return super().set_speed(factor)

    async def synthesize(self, text: str, voice: VoiceId) -> AsyncIterator[PcmChunk]:
        if not self._demoted:
            emitted = False
            try:
                async for chunk in self.primary.synthesize(text, voice):
                    emitted = emitted or chunk.pcm.size > 0
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if emitted:
                    raise
                self._demoted = True
                if self.on_notice:
                    with contextlib.suppress(Exception):
                        self.on_notice(
                            f"{getattr(self.primary, 'name', 'the paid voice')} "
                            f"is unavailable ({exc}); using the free voice."
                        )

        # The backup speaks the platform's own voice name, not an ID.
        backup_voice = VoiceId(
            vendor=getattr(self.backup, "name", "edge"),
            name=(
                voice.name
                if not _is_elevenlabs_id(voice.name)
                else (
                    DEFAULT_URDU_EDGE_VOICE
                    if (voice.language or "ur").startswith("ur")
                    else DEFAULT_ENGLISH_EDGE_VOICE
                )
            ),
            rate=voice.rate,
            style=voice.style,
            language=voice.language,
        )
        async for chunk in self.backup.synthesize(text, backup_voice):
            yield chunk


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
        language = (getattr(voice, "language", None) or "ur").lower()
        default_voice = (
            DEFAULT_ENGLISH_EDGE_VOICE
            if not language.startswith("ur")
            else DEFAULT_URDU_EDGE_VOICE
        )
        voice_name = (
            voice.name
            if (voice and voice.name and voice.name != "azure")
            else default_voice
        )
        line = speakable(text, language=language)
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


