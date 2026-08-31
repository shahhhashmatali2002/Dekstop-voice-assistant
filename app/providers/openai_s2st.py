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
    ProviderError,
    ProviderMisconfigured,
    ProviderRateLimited,
    ProviderUnavailable,
    S2stConfig,
    S2stStream,
    SpeechTranslator,
    TranscriptDelta,
)

#: The TRANSLATION endpoint, which is a different path from the voice-agent
#: one - "/v1/realtime/translations", not "/v1/realtime". Getting this wrong
#: fails in a way that points at the wrong thing entirely: the voice-agent
#: endpoint accepts the connection, accepts the audio, and rejects the session
#: with
#:
#:     Unknown parameter: 'session.audio.output.language'
#:
#: which reads as "that field is wrong" when the field is right and the URL is
#: wrong. Verified against developers.openai.com in August 2026.
ENDPOINT = "wss://api.openai.com/v1/realtime/translations"

#: The protocol, verbatim from the provider's translation guide (August 2026).
#: Written out as constants because getting these names wrong is the single
#: most expensive mistake available here: every one of them fails as SILENCE
#: rather than as an error, and each wrong guess costs a live call to discover.
#: On this endpoint the client events are namespaced under "session." - which
#: the voice-agent API does not do, and which produced:
#:
#:     Invalid value: 'inp...end'. Supported values are: 'session.update',
#:     'session.input_audio_buffer.append', and 'session.close'
SESSION_UPDATE = "session.update"
APPEND_AUDIO = "session.input_audio_buffer.append"
CLEAR_AUDIO = "session.input_audio_buffer.clear"
CLOSE_SESSION = "session.close"

#: Server events. The two transcripts are NOT interchangeable: input is the
#: Urdu that was heard, output is the English that was produced. Treating both
#: as the translation puts the source language in the transcript window as
#: though it were the result.
OUTPUT_AUDIO_DELTA = "session.output_audio.delta"
OUTPUT_TRANSCRIPT_DELTA = "session.output_transcript.delta"
INPUT_TRANSCRIPT_DELTA = "session.input_transcript.delta"
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
        #: Why the reader stopped, if it stopped badly. Kept rather than
        #: discarded: this socket failing quietly is indistinguishable, from
        #: the outside, from a session that is working and simply has nothing
        #: to say - and that ambiguity cost a whole live call.
        self.failure: Exception | None = None
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
                json.dumps(
                    {"type": APPEND_AUDIO, "audio": payload}
                )
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

                if kind == OUTPUT_AUDIO_DELTA or kind.endswith("output_audio.delta"):
                    b = msg.get("delta") or msg.get("audio")
                    if b:
                        pcm = np.frombuffer(base64.b64decode(b), dtype=DTYPE)
                        if self._sent_at:
                            self.last_rtt_ms = (now() - self._sent_at) * 1000.0
                        await self._put(self._audio, PcmChunk(pcm, self._rate))

                elif kind.startswith("session.output_transcript"):
                    # The English. "delta" is a fragment, "done" carries the
                    # whole segment.
                    if "delta" in kind:
                        text = msg.get("delta", "")
                        if text:
                            await self._put(
                                self._text, TranscriptDelta(text, is_final=False)
                            )
                    else:
                        text = msg.get("transcript", "") or msg.get("text", "")
                        if text:
                            await self._put(
                                self._text, TranscriptDelta(text, is_final=True)
                            )

                elif kind.startswith("session.input_transcript"):
                    # The Urdu that was heard. Useful one day for showing the
                    # user what was understood; never the translation.
                    pass

                elif kind == "error":
                    err = msg.get("error", {})
                    code = str(err.get("code", ""))
                    detail = err.get("message", raw[:200])
                    if "rate" in code or "quota" in code:
                        raise ProviderRateLimited(detail)
                    raise ProviderUnavailable(detail)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # NEVER swallowed. The previous version raised ProviderUnavailable
            # for a server error event three lines above and then discarded it
            # here, so a rejected session - a model name the account cannot
            # use, an unsupported session shape - looked exactly like silence:
            # the pipeline pushed audio into a socket that would never answer,
            # printed nothing, and the client heard nothing. Recorded here and
            # re-raised to whoever is reading the stream.
            self.failure = exc
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
                self._raise_failure()
                return
            yield item

    async def transcripts(self) -> AsyncIterator[TranscriptDelta]:
        while True:
            item = await self._text.get()
            if item is None:
                self._raise_failure()
                return
            yield item

    def _raise_failure(self) -> None:
        """Report why the stream ended, if the reason was a failure.

        Raised at the END of the stream rather than logged, so the caller's
        existing error path handles it: the outgoing pipeline already knows how
        to degrade and say why.
        """
        failure = self.failure
        if failure is None:
            return
        self.failure = None
        if isinstance(failure, ProviderError):
            raise failure
        raise ProviderUnavailable(f"s2st stream ended: {failure}")

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._ws.send(json.dumps({"type": CLOSE_SESSION}))
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


        url = self.url(cfg)
        try:  # pragma: no cover - network path
            ws = await websockets.connect(
                url, additional_headers=self.headers(), ping_interval=None
            )

        except Exception as exc:
            raise ProviderUnavailable(f"could not open s2st session: {exc}") from exc

        await ws.send(json.dumps(self.session_update(cfg)))
        return OpenAiS2stStream(ws, cfg.rate)

    # ------------------------------------------------------------------
    # the wire shape, kept as pure functions so it can be tested without a
    # socket - this is the part that broke when the beta API was retired
    # ------------------------------------------------------------------

    def url(self, cfg: S2stConfig) -> str:
        return f"{ENDPOINT}?model={cfg.model or self.model}"

    def headers(self) -> dict:
        """Authorization only.

        `OpenAI-Beta: realtime=v1` selected the beta wire shape and MUST NOT
        be sent: the beta shape is retired, and a session opened with it is
        closed immediately with 4000 / invalid_request_error
        .beta_api_shape_disabled - which reaches the user as "outgoing
        translation unavailable" with no hint that a header is the cause.
        """
        return {"Authorization": f"Bearer {self.api_key}"}

    def session_update(self, cfg: S2stConfig) -> dict:
        """The translation session, which is deliberately almost empty.

        A translation session is not a conversation: there are no
        instructions, no modalities, no turn detection and no response.create.
        The model consumes audio continuously and emits translated audio plus
        transcript deltas, and the ONLY thing it needs told is the output
        language.

        Everything the previous version sent here - a session type, an
        instructions prompt telling it to translate and not to chat, input and
        output formats - belonged to the voice-agent API. Sent to this
        endpoint they are unknown parameters, and one rejected session update
        is the difference between a working call and a socket that listens
        politely and never speaks.
        """
        return {
            "type": SESSION_UPDATE,
            "session": {
                "audio": {
                    "input": {
                        "transcription": {"model": "gpt-realtime-whisper"},
                        "noise_reduction": {"type": "near_field"},
                    },
                    "output": {"language": cfg.target_language},
                },
            },
        }

