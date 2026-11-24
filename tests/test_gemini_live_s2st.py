import asyncio
import base64
import json
import pytest
import numpy as np

from app.providers.base import PcmChunk, ProviderMisconfigured, ProviderUnavailable, S2stConfig
from app.providers.gemini_live_s2st import (
    GeminiLiveSpeechTranslator,
    GeminiLiveS2stStream,
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    RATE_IN,
    RATE_OUT,
)


class FakeWebSocket:
    """Mock WebSocket for testing Gemini Live."""

    def __init__(self, incoming_messages=None):
        self.sent = []
        self.incoming = list(incoming_messages or [])
        self.closed = False

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self.incoming:
            await asyncio.sleep(0.01)
            return json.dumps({"setupComplete": {}})
        return self.incoming.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while not self.incoming and not self.closed:
            await asyncio.sleep(0.01)
        if self.closed:
            raise StopAsyncIteration
        return self.incoming.pop(0)

    async def close(self):
        self.closed = True


def test_translator_requires_key():
    t = GeminiLiveSpeechTranslator(api_key="")
    with pytest.raises(ProviderMisconfigured, match="GEMINI_API_KEY"):
        asyncio.run(t.open(S2stConfig("ur", "en", RATE_IN)))


def test_translator_url():
    t = GeminiLiveSpeechTranslator(api_key="test_key_123")
    url = t._url()
    assert "key=test_key_123" in url
    assert "BidiGenerateContent" in url


def test_stream_push_audio():
    async def run():
        ws = FakeWebSocket()
        stream = GeminiLiveS2stStream(ws, rate_in=RATE_IN, rate_out=RATE_OUT)

        pcm = np.ones(320, dtype=np.int16) * 1000
        await stream.push_audio(pcm)

        assert len(ws.sent) == 1
        msg = json.loads(ws.sent[0])
        assert "realtimeInput" in msg
        chunks = msg["realtimeInput"]["mediaChunks"]
        assert len(chunks) == 1
        assert chunks[0]["mimeType"] == f"audio/pcm;rate={RATE_IN}"

        decoded = np.frombuffer(base64.b64decode(chunks[0]["data"]), dtype=np.int16)
        np.testing.assert_array_equal(decoded, pcm)

        await stream.aclose()

    asyncio.run(run())


def test_stream_receives_audio_and_transcript():
    async def run():
        audio_pcm = np.ones(480, dtype=np.int16) * 500
        b64_audio = base64.b64encode(audio_pcm.tobytes()).decode("ascii")

        server_msg = json.dumps({
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": f"audio/pcm;rate={RATE_OUT}",
                                "data": b64_audio,
                            }
                        },
                        {
                            "text": "Hello, my name is Hashmat."
                        }
                    ]
                },
                "turnComplete": True,
            }
        })

        ws = FakeWebSocket(incoming_messages=[server_msg])
        stream = GeminiLiveS2stStream(ws, rate_in=RATE_IN, rate_out=RATE_OUT)

        # Receive audio
        chunks = []
        async for chunk in stream.audio_out():
            chunks.append(chunk)
            break

        assert len(chunks) == 1
        assert chunks[0].rate == RATE_OUT
        np.testing.assert_array_equal(chunks[0].pcm, audio_pcm)

        # Receive transcript
        deltas = []
        async for delta in stream.transcripts():
            deltas.append(delta)
            break

        assert len(deltas) == 1
        assert deltas[0].text == "Hello, my name is Hashmat."
        assert deltas[0].is_final is True

        await stream.aclose()

    asyncio.run(run())


def test_stream_interrupted():
    async def run():
        audio_pcm = np.ones(480, dtype=np.int16) * 500
        b64_audio = base64.b64encode(audio_pcm.tobytes()).decode("ascii")

        server_msg = json.dumps({
            "serverContent": {
                "modelTurn": {
                    "parts": [{
                        "inlineData": {
                            "mimeType": f"audio/pcm;rate={RATE_OUT}",
                            "data": b64_audio,
                        }
                    }]
                }
            }
        })
        interrupted_msg = json.dumps({
            "serverContent": {
                "interrupted": True
            }
        })

        ws = FakeWebSocket(incoming_messages=[server_msg, interrupted_msg])
        stream = GeminiLiveS2stStream(ws, rate_in=RATE_IN, rate_out=RATE_OUT)

        # Wait briefly for reader task to process messages
        await asyncio.sleep(0.05)

        # Interruption preserves the pending audio queue so previous sentences finish speaking
        assert not stream._audio.empty()

        await stream.aclose()

    asyncio.run(run())


def test_stream_receives_input_and_output_transcription():
    async def run():
        msg1 = json.dumps({
            "serverContent": {
                "inputTranscription": {"text": "میرا نام "},
                "outputTranscription": {"text": "My name "},
            }
        })
        msg2 = json.dumps({
            "serverContent": {
                "inputTranscription": {"text": "حشمت علی ہے۔"},
                "outputTranscription": {"text": "is Hashmat Ali."},
                "turnComplete": True,
            }
        })

        ws = FakeWebSocket(incoming_messages=[msg1, msg2])
        stream = GeminiLiveS2stStream(ws, rate_in=RATE_IN, rate_out=RATE_OUT)

        deltas = []
        async for delta in stream.transcripts():
            deltas.append(delta)
            break

        assert len(deltas) == 1
        assert deltas[0].text == "My name is Hashmat Ali."
        assert deltas[0].source_text == "میرا نام حشمت علی ہے۔"
        assert deltas[0].is_final is True

        await stream.aclose()

    asyncio.run(run())


def test_to_urdu_script_converts_hindi():
    from app.providers.gemini_live_s2st import to_urdu_script

    assert to_urdu_script("कैसे हैं आप") == "کیسے ہیں آپ"
    assert to_urdu_script("हेलो कैसे हैं आप") == "ہیلو کیسے ہیں آپ"
    assert to_urdu_script("मेरा नाम हस्मत अली है") == "میرا نام حشمت علی ہے"
    assert to_urdu_script("Hello. कैसे हैं आप?") == "Hello. کیسے ہیں آپ?"
    assert to_urdu_script("کیسے ہیں آپ") == "کیسے ہیں آپ"


