"""The referee, and the two bugs that made it necessary.

These are the reported failures, written down as tests:

    said "I am John Wick"               heard "John Bek"
    said "I work at Macrosoft Pakistan"  heard "Microsoft"

The first is a recogniser pinned to the wrong language. The second is a
language model tidying a name it did not recognise. Both are checked here.
"""

from __future__ import annotations

import asyncio

import numpy as np

from app.providers.base import AsrConfig, AsrStream, SpeechToText, TranscriptDelta
from app.providers.dual_asr import (
    DualLanguageSpeechToText,
    arabic_ratio,
    choose,
    score,
)
from app.providers.tts import speakable


# ----------------------------------------------------------------- scoring


def test_english_wins_when_the_urdu_model_returns_latin_text():
    """The reported failure. Urdu was pinned; English was being spoken."""
    language, text, _ = choose(
        {
            "en": ("I am John Wick", 0.86),
            "ur": ("آئی ایم جان بیک", 0.91),
        }
    )
    assert language == "en"
    assert text == "I am John Wick"


def test_a_real_urdu_sentence_still_wins():
    """The fix must not simply prefer English - that is the same bug back."""
    language, text, _ = choose(
        {
            "en": ("me ra nam john hai", 0.55),
            "ur": ("میرا نام جان ہے اور میں کام کرتا ہوں", 0.88),
        }
    )
    assert language == "ur"


def test_one_confident_word_does_not_beat_a_whole_sentence():
    language, _, _ = choose(
        {
            "en": ("I work at Macrosoft Pakistan and we ship on Friday", 0.80),
            "ur": ("جی", 0.99),
        }
    )
    assert language == "en"


def test_an_urdu_result_with_no_urdu_script_is_discounted():
    assert score("ur", "hello there my friend", 0.9) < score(
        "ur", "ہیلو میرے دوست جی ہاں", 0.9
    )


def test_nothing_heard_is_not_an_utterance():
    assert choose({}) == ("", "", 0.0)
    assert choose({"en": ("", 0.99), "ur": ("   ", 0.99)}) == ("", "", 0.0)


def test_arabic_ratio_ignores_digits_and_punctuation():
    assert arabic_ratio("میرا نام, 2026!") == 1.0
    assert arabic_ratio("2026 - 15%") == 0.0


# ------------------------------------------------------------------ stream


class _FakeStream(AsrStream):
    def __init__(self, deltas):
        self._deltas = list(deltas)
        self.audio_frames = 0
        self.closed = False

    async def push_audio(self, pcm):
        self.audio_frames += 1

    async def hypotheses(self):
        for delta in self._deltas:
            await asyncio.sleep(0)
            yield delta

    async def aclose(self):
        self.closed = True


class _FakeAsr(SpeechToText):
    def __init__(self, by_language):
        self.by_language = by_language
        self.opened: list[AsrConfig] = []
        self.streams: dict[str, _FakeStream] = {}

    async def open(self, cfg: AsrConfig) -> AsrStream:
        self.opened.append(cfg)
        stream = _FakeStream(self.by_language.get(cfg.language, []))
        self.streams[cfg.language] = stream
        return stream


def _final(text, confidence):
    return TranscriptDelta(text=text, is_final=True, confidence=confidence)


def test_the_winning_transcript_is_the_one_emitted():
    asyncio.run(_the_winning_transcript_is_the_one_emitted())


async def _the_winning_transcript_is_the_one_emitted():
    asr = _FakeAsr(
        {
            "en": [_final("I work at Macrosoft Pakistan", 0.88)],
            "ur": [_final("مائیکروسافٹ پاکستان", 0.90)],
        }
    )
    dual = DualLanguageSpeechToText(asr, ("en", "ur"))
    stream = await dual.open(AsrConfig(language="auto", rate=16_000, keyterms=["Macrosoft"]))

    finals = []
    async def drain():
        async for delta in stream.hypotheses():
            if delta.is_final:
                finals.append(delta)
                break

    await asyncio.wait_for(drain(), timeout=5)
    await stream.aclose()

    assert finals[0].text == "I work at Macrosoft Pakistan"
    assert finals[0].language == "en"


def test_every_recogniser_hears_the_same_audio_and_the_same_keyterms():
    asyncio.run(_every_recogniser_hears_the_same_audio_and_the_same_keyterms())


async def _every_recogniser_hears_the_same_audio_and_the_same_keyterms():
    asr = _FakeAsr({"en": [], "ur": []})
    dual = DualLanguageSpeechToText(asr, ("en", "ur"))
    stream = await dual.open(
        AsrConfig(language="auto", rate=16_000, keyterms=["Macrosoft", "John Wick"])
    )
    await stream.push_audio(np.zeros(320, dtype=np.int16))
    await stream.push_audio(np.zeros(320, dtype=np.int16))
    await stream.aclose()

    assert [cfg.language for cfg in asr.opened] == ["en", "ur"]
    assert all(cfg.keyterms == ["Macrosoft", "John Wick"] for cfg in asr.opened)
    assert all(s.audio_frames == 2 for s in asr.streams.values())
    assert all(s.closed for s in asr.streams.values())


def test_one_language_is_a_plain_recogniser_not_a_referee():
    asyncio.run(_one_language_is_a_plain_recogniser_not_a_referee())


async def _one_language_is_a_plain_recogniser_not_a_referee():
    asr = _FakeAsr({"ur": []})
    dual = DualLanguageSpeechToText(asr, ("ur",))
    stream = await dual.open(AsrConfig(language="auto", rate=16_000))
    assert isinstance(stream, _FakeStream)
    await stream.aclose()


# --------------------------------------------------------------- the voice


def test_english_is_not_transliterated_into_urdu_script():
    """The other half of the accent complaint.

    `speakable` used to run the Urdu preparation on everything, so the English
    the client hears reached the English voice with "I", "a" and "API"
    replaced by their Urdu spellings.
    """
    said = speakable(
        "I work at Macrosoft Pakistan and I am a developer using the API",
        language="en",
    )
    assert said == (
        "I work at Macrosoft Pakistan and I am a developer using the API."
    )
    assert arabic_ratio(said) == 0.0


def test_urdu_still_gets_the_urdu_preparation():
    said = speakable("میرا نام جان ہے", language="ur")
    assert said.endswith("۔")


def test_an_unfinished_english_clause_is_left_open():
    assert speakable("but not in production", language="en", final=False).endswith(",")
