"""Google Gemini Live realtime speech-to-speech translation (Urdu <-> English).

Connects to the Gemini Multimodal Live API WebSocket endpoint:
    wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent

Features:
  * Native direct audio-in -> audio-out streaming via bidirectional WebSocket.
  * Ultra-low latency (~300-500ms): no waiting for full sentences or pauses.
  * Understands mixed code-switched South Asian speech (Urdu + English loanwords,
    names, tech jargon) without phonetic mangling.
  * 24 kHz mono PCM16 output directly matches the outgoing pipeline rate.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from typing import AsyncIterator

import numpy as np

from app.audio import convert
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

DEFAULT_MODEL = "models/gemini-2.5-flash-native-audio-latest"
DEFAULT_VOICE = "Puck"  # Clear, dynamic English voice
RATE_IN = 16_000
RATE_OUT = 24_000

SYSTEM_INSTRUCTION = """\
You are a live, real-time simultaneous speech interpreter on a live business call.
Your job is to listen to spoken Urdu (which may contain English loanwords, technical \
terms, and names) and IMMEDIATELY speak the accurate, natural English translation.

Rules:
1. Speak ONLY the translation. Never add greetings, explanations, commentary, or notes.
2. Translate in the first person ("I am...", "My name is..."), preserving the speaker's voice and tone.
3. Preserve names of people (e.g. Hashmat Ali), companies, and software engineering \
terminology accurately in English.
4. Begin speaking the translation immediately without delay.
"""


import re

# Devanagari Unicode range: \u0900 - \u097F
DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")

HINDI_URDU_WORDS = {
    "कैसे": "کیسے",
    "हैं": "ہیں",
    "आप": "آپ",
    "हेलो": "ہیلو",
    "है": "ہے",
    "हूं": "ہوں",
    "हूँ": "ہوں",
    "मेरा": "میرا",
    "मेरी": "میری",
    "मेरे": "میرے",
    "नाम": "نام",
    "हस्मत": "حشمت",
    "हशमत": "حشمت",
    "अली": "علی",
    "क्या": "کیا",
    "हाल": "حال",
    "चाल": "چال",
    "ठीक": "ٹھیک",
    "शुक्रिया": "شکریہ",
    "अच्छा": "اچھا",
    "हाँ": "ہاں",
    "हां": "ہاں",
    "नहीं": "نہیں",
    "मेकओवर": "میک اوور",
    "पाकिस्तान": "پاکستان",
    "डेवलपर": "ڈویلپر",
    "काम": "کام",
    "करता": "کرتا",
    "करती": "کرتی",
    "करते": "کرتے",
    "बात": "بات",
    "करना": "کرنا",
    "चाहता": "چاہتا",
    "चाहती": "چاہتی",
    "चाहते": "چاہتے",
    "बहुत": "بہت",
    "धन्यवाद": "شکریہ",
    "नमस्ते": "سلام",
    "नमस्कार": "سلام",
}

CHAR_MAP = {
    "अ": "ا", "आ": "آ", "इ": "اِ", "ई": "ای", "उ": "اُ", "ऊ": "او",
    "ए": "اے", "ऐ": "اے", "ओ": "او", "औ": "او",
    "क": "ک", "ख": "کھ", "ग": "گ", "घ": "گھ", "ङ": "ن",
    "च": "چ", "छ": "چھ", "ज": "ج", "झ": "جھ", "ञ": "ن",
    "ट": "ٹ", "ठ": "ٹھ", "ड": "ڈ", "ढ": "ڈھ", "ण": "ن",
    "त": "ت", "थ": "تھ", "द": "د", "ध": "دھ", "न": "ن",
    "प": "پ", "फ": "ف", "ब": "ب", "भ": "بھ", "म": "م",
    "य": "ی", "र": "ر", "ल": "ل", "व": "و",
    "श": "ش", "ष": "ش", "स": "س", "ह": "ہ",
    "क़": "ق", "ख़": "خ", "ग़": "غ", "ज़": "ز", "ड़": "ڑ", "ढ़": "ڑھ", "फ़": "ف",
    "ा": "ا", "ि": "ِ", "ी": "ی", "ु": "ُ", "ू": "و",
    "े": "ے", "ै": "ے", "ो": "و", "ौ": "و",
    "ं": "ں", "ँ": "ں", "्": "", "ः": "ہ", "़": "",
    "।": "۔",
}


def to_urdu_script(text: str) -> str:
    """Ensure any South Asian Hindustani text is rendered in Urdu Arabic script."""
    if not text or not DEVANAGARI_RE.search(text):
        return text

    tokens = re.split(r"(\s+|[^\w\u0900-\u097f]+)", text)
    result = []
    for token in tokens:
        if token in HINDI_URDU_WORDS:
            result.append(HINDI_URDU_WORDS[token])
        elif DEVANAGARI_RE.search(token):
            out = []
            i = 0
            while i < len(token):
                if i + 1 < len(token) and token[i : i + 2] in CHAR_MAP:
                    out.append(CHAR_MAP[token[i : i + 2]])
                    i += 2
                elif token[i] in CHAR_MAP:
                    out.append(CHAR_MAP[token[i]])
                    i += 1
                else:
                    out.append(token[i])
                    i += 1
            result.append("".join(out))
        else:
            result.append(token)
    return "".join(result)


class GeminiLiveS2stStream(S2stStream):
    """One active Gemini Live WebSocket translation session."""

    def __init__(
        self,
        ws,
        *,
        asr_stream=None,
        rate_in: int = RATE_IN,
        rate_out: int = RATE_OUT,
    ) -> None:
        self._ws = ws
        self._asr_stream = asr_stream
        self._rate_in = rate_in
        self._rate_out = rate_out
        self._audio: asyncio.Queue[PcmChunk | None] = asyncio.Queue(maxsize=256)
        self._text: asyncio.Queue[TranscriptDelta | None] = asyncio.Queue(maxsize=256)
        self._current_input_text: str = ""
        self._current_output_text: str = ""
        self._last_asr_text: str = ""
        self._reader = asyncio.create_task(self._read_loop())
        self._asr_task = (
            asyncio.create_task(self._read_asr()) if asr_stream is not None else None
        )
        self._closed = False
        self._last_push = now()

    async def _read_asr(self) -> None:
        if self._asr_stream is None:
            return
        try:
            async for delta in self._asr_stream.hypotheses():
                if self._closed:
                    break
                txt = (delta.text or "").strip()
                if txt:
                    self._last_asr_text = txt
        except (asyncio.CancelledError, Exception):
            pass

    async def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        if pcm.dtype != DTYPE:
            pcm = pcm.astype(DTYPE)
        if len(pcm) == 0:
            return

        # Resample to input rate if necessary
        if self._rate_in != RATE_IN:
            pcm = convert.resample(pcm, self._rate_in, RATE_IN)

        # Forward audio to native Urdu ASR stream for authentic Urdu transcript
        if self._asr_stream is not None:
            with contextlib.suppress(Exception):
                await self._asr_stream.push_audio(pcm)

        b64_audio = base64.b64encode(pcm.tobytes()).decode("ascii")
        msg = {
            "realtimeInput": {
                "mediaChunks": [
                    {
                        "mimeType": f"audio/pcm;rate={RATE_IN}",
                        "data": b64_audio,
                    }
                ]
            }
        }
        try:
            await self._ws.send(json.dumps(msg))
            self._last_push = now()
        except Exception as exc:  # pragma: no cover
            raise ProviderUnavailable(f"Gemini Live send failed: {exc}") from exc

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                if self._closed:
                    break
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                if "error" in msg:
                    err = msg["error"]
                    code = err.get("code")
                    message = err.get("message", "Gemini Live error")
                    if code == 429 or "quota" in message.lower():
                        raise ProviderRateLimited(message)
                    raise ProviderUnavailable(f"Gemini Live {code}: {message}")

                server_content = msg.get("serverContent") or {}
                
                # Check for interruption (barge-in)
                if server_content.get("interrupted"):
                    # Clear pending audio
                    while not self._audio.empty():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            self._audio.get_nowait()
                    self._current_input_text = ""
                    self._current_output_text = ""
                    continue

                # 1. Real-time transcription from Gemini Live:
                # User's spoken speech:
                input_transcription = server_content.get("inputTranscription")
                if input_transcription and isinstance(input_transcription, dict):
                    in_text = input_transcription.get("text", "")
                    if in_text:
                        self._current_input_text += in_text

                # Model's translated speech:
                output_transcription = server_content.get("outputTranscription")
                if output_transcription and isinstance(output_transcription, dict):
                    out_text = output_transcription.get("text", "")
                    if out_text:
                        self._current_output_text += out_text

                # 2. Audio data & model turns
                model_turn = server_content.get("modelTurn") or {}
                for part in model_turn.get("parts", []):
                    inline_data = part.get("inlineData")
                    if inline_data:
                        raw_pcm = base64.b64decode(inline_data.get("data", ""))
                        if raw_pcm:
                            chunk_pcm = np.frombuffer(raw_pcm, dtype=DTYPE)
                            await self._put_audio(PcmChunk(chunk_pcm, self._rate_out))

                    # If part text is delivered and not an internal thought
                    if not part.get("thought"):
                        txt = (part.get("text") or "").strip()
                        if txt and not txt.startswith("**"):
                            if not self._current_output_text:
                                self._current_output_text = txt
                            elif txt not in self._current_output_text:
                                self._current_output_text += " " + txt

                # 3. When turn completes, emit the accumulated transcript
                if server_content.get("turnComplete"):
                    final_translated = self._current_output_text.strip()
                    raw_source = (
                        self._last_asr_text.strip()
                        or self._current_input_text.strip()
                    )
                    final_source = to_urdu_script(raw_source)
                    if final_translated or final_source:
                        await self._put_text(
                            TranscriptDelta(
                                text=final_translated,
                                is_final=True,
                                language="en",
                                source_text=final_source,
                            )
                        )
                    self._current_input_text = ""
                    self._current_output_text = ""
                    self._last_asr_text = ""

        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover
            pass
        finally:
            with contextlib.suppress(Exception):
                self._audio.put_nowait(None)
            with contextlib.suppress(Exception):
                self._text.put_nowait(None)

    async def _put_audio(self, item: PcmChunk | None) -> None:
        try:
            self._audio.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover
            with contextlib.suppress(asyncio.QueueEmpty):
                self._audio.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._audio.put_nowait(item)

    async def _put_text(self, item: TranscriptDelta | None) -> None:
        try:
            self._text.put_nowait(item)
        except asyncio.QueueFull:  # pragma: no cover
            with contextlib.suppress(asyncio.QueueEmpty):
                self._text.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._text.put_nowait(item)

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
        if self._asr_task is not None:
            self._asr_task.cancel()
        if self._asr_stream is not None:
            with contextlib.suppress(Exception):
                await self._asr_stream.aclose()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._reader
        with contextlib.suppress(Exception):
            await self._ws.close()


class GeminiLiveSpeechTranslator(SpeechTranslator):
    """Direct speech-to-speech translator using Gemini Multimodal Live WebSocket."""

    name = "gemini-live-s2st"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        asr=None,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        )
        self.model = os.environ.get("GEMINI_LIVE_MODEL") or model
        self.voice = os.environ.get("GEMINI_LIVE_VOICE") or voice
        self.asr = asr

    def _url(self) -> str:
        return (
            "wss://generativelanguage.googleapis.com/ws/"
            "google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent"
            f"?key={self.api_key}"
        )

    async def open(self, cfg: S2stConfig) -> S2stStream:
        if not self.api_key:
            raise ProviderMisconfigured("GEMINI_API_KEY is not set")
        try:
            import websockets
        except ImportError as exc:  # pragma: no cover
            raise ProviderMisconfigured("pip install websockets") from exc

        url = self._url()
        try:
            ws = await websockets.connect(url)
        except Exception as exc:  # pragma: no cover
            raise ProviderUnavailable(f"could not connect to Gemini Live: {exc}") from exc

        # Initial handshake: setup message
        setup_payload = {
            "setup": {
                "model": self.model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self.voice,
                            }
                        }
                    },
                    "thinkingConfig": {
                        "thinkingBudget": 0,
                    },
                },
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
                "systemInstruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION}],
                },
            }
        }
        try:
            await ws.send(json.dumps(setup_payload))
            raw_resp = await asyncio.wait_for(ws.recv(), timeout=6.0)
            data = json.loads(raw_resp)
            if "error" in data:
                raise ProviderUnavailable(f"Gemini Live handshake error: {data['error']}")
        except Exception as exc:  # pragma: no cover
            with contextlib.suppress(Exception):
                await ws.close()
            raise ProviderUnavailable(f"Gemini Live setup failed: {exc}") from exc

        asr_stream = None
        if self.asr is not None:
            try:
                from app.providers.base import AsrConfig
                asr_stream = await self.asr.open(
                    AsrConfig(language="ur", rate=RATE_IN)
                )
            except Exception as exc:
                print(f"[Gemini Live] Failed to open ASR stream: {exc}", flush=True)

        return GeminiLiveS2stStream(
            ws, asr_stream=asr_stream, rate_in=cfg.rate, rate_out=RATE_OUT
        )
