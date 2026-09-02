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

#: Deepgram counts keyterm prompting in tokens across the whole request, not
#: in list entries. Documented limit is 500; stay under it.
KEYTERM_TOKEN_BUDGET = 460


class DeepgramAsrStream(AsrStream):
    def __init__(self, ws, language: str = "") -> None:
        self._ws = ws
        self._language = language
        self._q: asyncio.Queue[TranscriptDelta | None] = asyncio.Queue(maxsize=128)
        self._reader = asyncio.create_task(self._read_loop())
        self._closed = False
        #: Why the read loop died, when it died badly. Surfaced through
        #: `hypotheses()` so the consumer can tell a polite close from a
        #: NET-0001 - swallowing this (as this class used to) made every
        #: death look voluntary, and nobody redials after a goodbye.
        self._failure: Exception | None = None

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

                # UtteranceEnd fires on a word-gap (see utterance_end_ms),
                # even when no silence tripped normal endpointing - the
                # signal that a fluent reader has finished a clause. Forward
                # it as a final marker so the segmenter flushes what it holds
                # instead of waiting for a pause that never comes. It carries
                # no transcript of its own; the text is whatever the last
                # interim confirmed.
                if msg.get("type") == "UtteranceEnd":
                    await self._put(
                        TranscriptDelta(
                            text="", is_final=True,
                            language=self._language or None,
                        )
                    )
                    continue

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

                # Confidence is what lets two recognisers listening to the
                # same microphone be compared (see dual_asr). Deepgram reports
                # it per alternative; on an interim it is often absent.
                confidence = alts[0].get("confidence")
                try:
                    confidence = float(confidence) if confidence is not None else None
                except (TypeError, ValueError):
                    confidence = None

                await self._put(
                    TranscriptDelta(
                        text=text,
                        is_final=bool(msg.get("is_final") or msg.get("speech_final")),
                        speaker=speaker,
                        language=self._language or None,
                        confidence=confidence,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
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
                failure, self._failure = self._failure, None
                if failure is not None and not self._closed:
                    if isinstance(failure, ProviderError):
                        raise failure
                    raise ProviderUnavailable(str(failure)) from failure
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
            # smart_format holds finalisation back while it waits to see the
            # whole of an entity it is formatting - a number, a date, a
            # currency amount - and the documented hold is up to three
            # seconds. In an offline transcript that is free; in a live
            # cascade it is three seconds added to exactly the sentences
            # that contain the figures a client call is about. no_delay
            # trades the formatting of a partially-heard entity away for
            # transcripts that arrive when the words were said.
            # How much silence ends an utterance (speech_final). Tunable
            # because it trades against the VAD hangover, which fires our own
            # flush at vad_hangover_ms - the two should stay in the same
            # region or one of them is dead weight.
            "endpointing": os.environ.get("DEEPGRAM_ENDPOINTING", "300"),
        }
        # OPT-IN extras, OFF by default. These touch the wire protocol and a
        # bad combination makes Deepgram close the connection, which reads as
        # "it stopped listening" - so they are only added when explicitly
        # asked for, and can be turned on one at a time to test on a live
        # call rather than blind.
        #
        #   DEEPGRAM_NO_DELAY=1          - stop smart_format holding finals
        #                                  back up to 3 s for entity formatting
        #   DEEPGRAM_UTTERANCE_END_MS=1000 - close a clause on a word-gap, not
        #                                  only on silence (the paragraph fix);
        #                                  also turns on vad_events, which it
        #                                  requires
        def _on(name: str) -> bool:
            return (os.environ.get(name) or "").strip().lower() in (
                "1", "true", "on", "yes",
            )

        extras = {}
        if _on("DEEPGRAM_NO_DELAY"):
            extras["no_delay"] = "true"
        uend = (os.environ.get("DEEPGRAM_UTTERANCE_END_MS") or "").strip()
        if uend and uend not in ("0", "off", "false"):
            extras["utterance_end_ms"] = uend
            extras["vad_events"] = "true"
        params.update(extras)
        query = urlencode(params)
        # Keyterms repeat as separate parameters (§7.7). The cap is Deepgram's
        # and it is counted in TOKENS across the whole list - 500 - not in
        # entries, so a hundred multi-word phrases silently overruns it and
        # the request is rejected or the tail is ignored. Budget in words, and
        # spend it on the front of the list, which terminology.keyterms()
        # already orders by how much a mis-hearing would cost.
        # MEASURED, not assumed: on this project's own Urdu audio, keyterm
        # prompting made recognition WORSE, and specifically worse on the
        # thing it was added to protect -
        #
        #   without keyterms   ... مائیکروسافٹ پاکستان   (Microsoft Pakistan)
        #   with keyterms      ... مائکرو صاحب پاکستان   ("micro sahib")
        #
        # An English keyterm list biases an Urdu model towards Urdu spellings
        # of English words that it was already getting right, and the nearest
        # Urdu word to a hint it half-matches can be worse than its own first
        # answer. So hints go to English only unless explicitly forced.
        latin = cfg.language.split("-")[0].lower() in ("en", "multi")
        forced = (os.environ.get("KEYTERMS_ALL_LANGUAGES") or "").lower() in (
            "1", "true", "yes", "on",
        )
        spent = 0
        for term in (cfg.keyterms if (latin or forced) else []):
            cost = len(term.split()) + 1
            if spent + cost > KEYTERM_TOKEN_BUDGET:
                break
            spent += cost
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
        return DeepgramAsrStream(ws, language=cfg.language)
