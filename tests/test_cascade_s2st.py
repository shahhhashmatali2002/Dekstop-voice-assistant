"""The OpenAI-free outgoing path: Urdu speech -> recogniser -> translator ->
English speech.

This carries the direction the user cares about most - the client hearing
them - so it is tested without a network, a device or a key. Everything below
is a stand-in whose behaviour can be made deliberately awkward.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.providers.base import (
    PcmChunk,
    ProviderUnavailable,
    S2stConfig,
    TranscriptDelta,
    TranslatedUnit,
)
from app.providers.cascade_s2st import (
    RATE_ASR,
    CascadeSpeechTranslator,
    english_voice,
)


class FakeAsrStream:
    def __init__(self, hypotheses):
        self._hypotheses = hypotheses
        self.pushed: list[np.ndarray] = []
        self.closed = False

    async def push_audio(self, pcm):
        self.pushed.append(pcm)

    async def hypotheses(self):
        for item in self._hypotheses:
            yield item
            await asyncio.sleep(0)

    async def aclose(self):
        self.closed = True


class FakeAsr:
    def __init__(self, hypotheses):
        self._hypotheses = hypotheses
        self.config = None
        self.stream = None

    async def open(self, cfg):
        self.config = cfg
        self.stream = FakeAsrStream(self._hypotheses)
        return self.stream


class FakeTranslator:
    def __init__(self, mapping=None, fail=None):
        self.mapping = mapping or {}
        self.fail = fail
        self.calls: list[str] = []

    async def translate(self, unit, ctx, terms=None):
        self.calls.append(unit.text)
        if self.fail is not None:
            raise self.fail
        return TranslatedUnit(self.mapping.get(unit.text, unit.text.upper()), unit)


class FakeTts:
    def __init__(self, rate=24_000, chunks=2, fail=None):
        self.rate = rate
        self.chunks = chunks
        self.fail = fail
        self.spoken: list[str] = []

    async def synthesize(self, text, voice):
        self.spoken.append(text)
        if self.fail is not None:
            raise self.fail
        for index in range(self.chunks):
            yield PcmChunk(
                pcm=np.full(480, index + 1, dtype=np.int16), rate=self.rate
            )
            await asyncio.sleep(0)


def cfg(rate=24_000):
    return S2stConfig(source_language="ur", target_language="en", rate=rate)


async def drain(stream):
    audio, text = [], []

    async def read_audio():
        async for chunk in stream.audio_out():
            audio.append(chunk)

    async def read_text():
        async for delta in stream.transcripts():
            text.append(delta)

    await asyncio.gather(read_audio(), read_text())
    return audio, text


def final(text):
    return TranscriptDelta(text=text, is_final=True)


def interim(text):
    return TranscriptDelta(text=text, is_final=False)


# ------------------------------------------------------------------- opening


def test_the_recogniser_is_opened_in_the_source_language():
    """Urdu, at the rate the recogniser wants - not the rate the pipeline
    speaks. nova-3 is the only Deepgram model that supports Urdu at all."""
    asr = FakeAsr([])

    async def run():
        translator = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await translator.open(cfg())
        await stream.aclose()

    asyncio.run(run())

    assert asr.config.language == "ur"
    assert asr.config.rate == RATE_ASR
    assert asr.config.interim_results is True


def test_a_missing_part_is_refused_with_a_reason():
    from app.providers.base import ProviderMisconfigured

    async def run():
        translator = CascadeSpeechTranslator(None, FakeTranslator(), FakeTts())
        with pytest.raises(ProviderMisconfigured):
            await translator.open(cfg())

    asyncio.run(run())


def test_english_is_the_only_output_language_offered():
    """Honest about what it can do: the translator could write any language,
    but this path exists to send English to the client."""
    translator = CascadeSpeechTranslator(FakeAsr([]), FakeTranslator(), FakeTts())
    assert translator.supports_output("en")
    assert not translator.supports_output("ur")


# --------------------------------------------------------------------- input


def test_audio_is_resampled_for_the_recogniser():
    """The pipeline speaks 24 kHz and the recogniser was opened at 16 kHz.
    Handing it 24 kHz samples it believes are 16 kHz makes every voice sound
    slow and deep, and the transcript comes back as nonsense or nothing."""
    asr = FakeAsr([])

    async def run():
        translator = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await translator.open(cfg(rate=24_000))
        await stream.push_audio(np.zeros(2_400, dtype=np.int16))  # 100 ms
        await stream.aclose()

    asyncio.run(run())

    assert len(asr.stream.pushed) == 1
    # 100 ms at 16 kHz, give or take a sample from the resampler.
    assert abs(asr.stream.pushed[0].size - 1_600) <= 2


def test_audio_is_passed_through_when_the_rates_already_match():
    asr = FakeAsr([])

    async def run():
        translator = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await translator.open(cfg(rate=RATE_ASR))
        await stream.push_audio(np.zeros(1_600, dtype=np.int16))
        await stream.aclose()

    asyncio.run(run())
    assert asr.stream.pushed[0].size == 1_600


# -------------------------------------------------------- what gets spoken


def test_only_finished_utterances_are_translated_and_spoken():
    """Interim hypotheses grow word by word - "My", "My name", "My name is".
    Translating each one means paying three times and speaking the sentence
    three times over the top of itself."""
    asr = FakeAsr([
        interim("میرا"),
        interim("میرا نام"),
        final("میرا نام حشمت ہے"),
    ])
    translator = FakeTranslator({"میرا نام حشمت ہے": "My name is Hashmat"})
    tts = FakeTts()

    async def run():
        provider = CascadeSpeechTranslator(asr, translator, tts)
        stream = await provider.open(cfg())
        return await drain(stream)

    audio, text = asyncio.run(run())

    assert translator.calls == ["میرا نام حشمت ہے"]
    assert tts.spoken == ["My name is Hashmat"]
    assert len(audio) == 2


def test_the_source_language_never_reaches_the_transcript_window():
    """This channel means "the translation" to everything downstream: the
    outgoing pipeline labels it [you] and shows it as what the client hears.
    Interim Urdu here filled the window with fragments - "کے", "کے کے",
    "اس کے" - which read as the app translating into the wrong language when
    it had not translated anything yet. The realtime model has the same two
    channels and only ever sends its OUTPUT transcript on this one."""
    asr = FakeAsr([interim("میرا"), interim("میرا نام"), final("میرا نام حشمت ہے")])

    async def run():
        provider = CascadeSpeechTranslator(
            asr, FakeTranslator({"میرا نام حشمت ہے": "My name is Hashmat"}), FakeTts()
        )
        return await drain(await provider.open(cfg()))

    _, text = asyncio.run(run())

    assert [d.text for d in text] == ["My name is Hashmat"]
    assert all(d.is_final for d in text)


def test_an_empty_translation_is_not_spoken():
    """Rule 2 of the interpreter prompt: unintelligible input returns nothing.
    Nothing must stay nothing rather than becoming an empty utterance."""
    asr = FakeAsr([final("...")])
    tts = FakeTts()

    async def run():
        provider = CascadeSpeechTranslator(asr, FakeTranslator({"...": "   "}), tts)
        return await drain(await provider.open(cfg()))

    audio, _ = asyncio.run(run())
    assert tts.spoken == []
    assert audio == []


def test_two_utterances_are_spoken_in_order_and_never_overlap():
    """Synthesising two sentences at once interleaves them at the sink, which
    is heard as one voice talking over itself."""
    asr = FakeAsr([final("ایک"), final("دو")])
    translator = FakeTranslator({"ایک": "One", "دو": "Two"})
    tts = FakeTts()

    async def run():
        provider = CascadeSpeechTranslator(asr, translator, tts)
        return await drain(await provider.open(cfg()))

    asyncio.run(run())

    assert tts.spoken == ["One", "Two"]


def test_the_last_sentence_is_not_cut_off_when_the_recogniser_finishes():
    """The recogniser's stream ending does NOT mean the audio is finished:
    the final utterance is still being translated and synthesised. Ending the
    audio stream at that moment silently drops the last thing said."""
    asr = FakeAsr([final("آخری جملہ")])
    tts = FakeTts(chunks=3)

    async def run():
        provider = CascadeSpeechTranslator(
            asr, FakeTranslator({"آخری جملہ": "The last sentence"}), tts
        )
        return await drain(await provider.open(cfg()))

    audio, _ = asyncio.run(run())

    assert tts.spoken == ["The last sentence"]
    assert len(audio) == 3


# ------------------------------------------------------------------ failures


def test_a_dead_translator_is_reported_not_swallowed():
    """A translator that has stopped answering must not present as a
    translator that heard nothing. Silence is the failure this project has
    lost the most time to."""
    asr = FakeAsr([final("کچھ")])
    translator = FakeTranslator(fail=ProviderUnavailable("no credits remaining"))

    async def run():
        provider = CascadeSpeechTranslator(asr, translator, FakeTts())
        stream = await provider.open(cfg())
        with pytest.raises(ProviderUnavailable, match="no credits"):
            async for _ in stream.audio_out():
                pass
        await stream.aclose()

    asyncio.run(run())


def test_a_dead_voice_is_reported_not_swallowed():
    asr = FakeAsr([final("کچھ")])
    tts = FakeTts(fail=ProviderUnavailable("tts refused"))

    async def run():
        provider = CascadeSpeechTranslator(
            asr, FakeTranslator({"کچھ": "something"}), tts
        )
        stream = await provider.open(cfg())
        with pytest.raises(ProviderUnavailable, match="tts refused"):
            async for _ in stream.audio_out():
                pass
        await stream.aclose()

    asyncio.run(run())


def test_closing_closes_the_recogniser_too():
    asr = FakeAsr([interim("...")])

    async def run():
        provider = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await provider.open(cfg())
        await stream.aclose()
        await stream.aclose()  # twice is safe
        return asr.stream.closed

    assert asyncio.run(run()) is True


def test_pushing_after_close_is_ignored():
    asr = FakeAsr([])

    async def run():
        provider = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await provider.open(cfg())
        await stream.aclose()
        await stream.push_audio(np.zeros(1_600, dtype=np.int16))
        return asr.stream.pushed

    assert asyncio.run(run()) == []


# --------------------------------------------------------------------- voice


def test_the_english_voice_is_an_edge_voice_by_default():
    voice = english_voice(24_000)
    assert voice.vendor == "edge"
    assert voice.name.startswith("en-")
    assert voice.rate == 24_000


def test_the_english_voice_can_be_overridden(monkeypatch):
    monkeypatch.setenv("ENGLISH_VOICE", "en-GB-RyanNeural")
    assert english_voice(24_000).name == "en-GB-RyanNeural"


def test_both_the_original_and_the_translation_are_reported():
    """The window shows two lines per utterance - what was said and what the
    client hears - so the provider has to report both. Before this, the
    outgoing direction hard-coded an empty original and the window could only
    ever show one side."""
    asr = FakeAsr([final("آپ کیسے ہیں")])

    async def run():
        provider = CascadeSpeechTranslator(
            asr, FakeTranslator({"آپ کیسے ہیں": "How are you?"}), FakeTts()
        )
        return await drain(await provider.open(cfg()))

    _, text = asyncio.run(run())

    assert len(text) == 1
    assert text[0].text == "How are you?"
    assert text[0].source_text == "آپ کیسے ہیں"
    assert text[0].language == "en"


def test_the_outgoing_pipeline_passes_the_original_to_the_window():
    from app.controllers.outgoing import OutgoingPipeline
    from app.models.telemetry import MetricsBus, TraceFactory

    class Stream:
        async def transcripts(self):
            yield TranscriptDelta(
                text="How are you?", is_final=True, source_text="آپ کیسے ہیں"
            )

        async def audio_out(self):
            if False:
                yield None

    seen = []
    pipe = OutgoingPipeline(
        s2st=object(),
        sink=object(),
        metrics=MetricsBus(),
        traces=TraceFactory(),
        on_transcript=seen.append,
    )
    pipe._stream = Stream()
    asyncio.run(pipe._consume_text())

    assert seen[0]["source"] == "آپ کیسے ہیں"
    assert seen[0]["translated"] == "How are you?"
    assert seen[0]["direction"] == "outgoing"


def test_the_recogniser_is_kept_alive_through_a_pause():
    """Deepgram closes a stream with nothing on it for ten seconds - NET-0001,
    "did not receive audio data or a text message within the timeout window".
    Pauses longer than ten seconds are normal in a meeting, and losing the
    stream means losing the next thing said."""
    asr = FakeAsr([])
    sent = []

    async def run():
        provider = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await provider.open(cfg())
        stream._asr.keepalive = lambda: _record(sent)
        stream.KEEPALIVE_EVERY_S = 0.02
        await asyncio.sleep(0.2)
        await stream.aclose()

    async def _record(bucket):
        bucket.append(True)

    asyncio.run(run())
    assert len(sent) >= 2


def test_pushing_audio_postpones_the_keepalive():
    """A keepalive sent while audio is already flowing is pure waste."""
    asr = FakeAsr([])
    sent = []

    async def _record(bucket):
        bucket.append(True)

    async def run():
        provider = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await provider.open(cfg(rate=RATE_ASR))
        stream._asr.keepalive = lambda: _record(sent)
        stream.KEEPALIVE_EVERY_S = 0.5
        for _ in range(10):
            await stream.push_audio(np.zeros(1_600, dtype=np.int16))
            await asyncio.sleep(0.02)
        await stream.aclose()

    asyncio.run(run())
    assert sent == []


def test_a_recogniser_without_keepalive_is_not_a_problem():
    """Not every recogniser needs one; the absence must not raise."""
    asr = FakeAsr([])

    async def run():
        provider = CascadeSpeechTranslator(asr, FakeTranslator(), FakeTts())
        stream = await provider.open(cfg())
        assert not hasattr(stream._asr, "keepalive")
        await asyncio.sleep(0.05)
        await stream.aclose()

    asyncio.run(run())


def test_playback_never_overruns_the_sink():
    """The voice delivers a 3.5 second sentence in a fraction of that time and
    the sink's ring holds 500 ms, dropping the OLDEST samples when full. Write
    everything at once and the client hears a fragment while the transcript
    shows a whole sentence. Measured on the real path: 336384 samples written,
    22080 dropped."""
    from app.audio.ring import RingBuffer
    from app.controllers.outgoing import OutgoingPipeline
    from app.models.telemetry import MetricsBus, TraceFactory

    class Sink:
        rate = 48_000
        channels = 2

        def __init__(self):
            self.ring = RingBuffer(48_000)  # 500 ms of 48 kHz stereo
            self.written = 0

        def write(self, pcm):
            self.written += pcm.size
            return self.ring.write(pcm)

    sink = Sink()
    pipe = OutgoingPipeline(
        s2st=object(), sink=sink, metrics=MetricsBus(), traces=TraceFactory()
    )
    pipe._running = True

    sentence = np.ones(336_384, dtype=np.int16)

    async def device():
        while sink.written < sentence.size:
            await asyncio.sleep(0.001)
            sink.ring.read(min(4_800, sink.ring.available))

    async def run():
        await asyncio.gather(pipe._write_paced(sentence), device())

    asyncio.run(run())

    assert sink.written == sentence.size
    assert sink.ring.dropped_samples == 0
