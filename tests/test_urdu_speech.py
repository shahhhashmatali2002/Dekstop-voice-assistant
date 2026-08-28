"""How Urdu text is prepared for a voice, and why.

Two user reports drive every test in this file:

    "urdu speaking like a robot"
    "some words don't pronounce accurate"

Neither was the voice's fault. The first was punctuation - a full stop after
every clause of a sentence - and the second was Latin script surviving into a
line an Urdu voice had to read aloud.
"""

import numpy as np
import pytest

from app.providers.base import VoiceId
from app.providers.tts import (
    AzureTextToSpeech,
    EdgeTextToSpeech,
    ElevenLabsTextToSpeech,
    NullTextToSpeech,
    build_ssml,
    rate_percent,
    speakable,
)
from app.providers.urdu_speech import (
    WORDS,
    close,
    ends_sentence,
    for_speech,
    pronounceable,
    say_word,
    spell_acronym,
)


# ----------------------------------------------------------------------
# pronunciation
# ----------------------------------------------------------------------


def test_an_english_word_is_said_the_way_an_urdu_speaker_says_it():
    """The translator is told to carry technical terms through unchanged
    (§8.4.3). Right for the transcript, unreadable for a Pakistani voice."""
    assert say_word("deployment") == "ڈیپلائمنٹ"
    assert say_word("YouTube") == "یوٹیوب"
    assert say_word("meeting") == "میٹنگ"


def test_case_and_surrounding_punctuation_do_not_hide_a_known_word():
    assert say_word("Deployment") == "ڈیپلائمنٹ"
    assert say_word("e-mail") == "ای میل"


def test_an_acronym_is_spelled_with_urdu_letter_names():
    """"API" read as a word is not a word. Said letter by letter it is."""
    assert spell_acronym("API") == "اے پی آئی"
    assert say_word("CEO") == "سی ای او"
    # Written with dots or without, it is the same word said aloud.
    assert say_word("A.P.I.") == "اے پی آئی"


def test_a_single_letter_is_a_letter():
    assert say_word("B") == "بی"


def test_a_loanword_plural_keeps_its_plural():
    """An Urdu speaker says "سرورز", not "سرور", for two of them."""
    assert say_word("servers") == "سرورز"
    assert say_word("tickets") == "ٹکٹز"


def test_a_word_we_do_not_know_is_left_for_the_voice():
    """Deliberate, and the most important decision in the module.

    Rule-based Latin-to-Urdu transliteration is confident about words it has
    never seen, and a wrong Urdu spelling is pronounced wrongly with total
    conviction - which is worse than the voice's own hedge. Adding the word to
    WORDS is the fix, and it is one line.
    """
    assert say_word("Kowalski") == "Kowalski"
    assert say_word("frobnicate") == "frobnicate"


def test_urdu_text_is_not_touched_at_all():
    urdu = "ہم نے کل شام یہ بات کی تھی"
    assert pronounceable(urdu) == urdu


def test_digits_are_left_alone():
    """The voice reads these correctly already; every edit is a chance to
    break something that works."""
    assert pronounceable("2026 میں 15 فیصد") == "2026 میں 15 فیصد"


def test_a_currency_symbol_moves_to_where_urdu_says_it():
    """A character-for-word map would produce "ڈالر پانچ"."""
    assert pronounceable("$5") == "5 ڈالر"
    assert pronounceable("50%") == "50 فیصد"


def test_a_mixed_sentence_comes_out_entirely_readable():
    got = pronounceable("ہم نے API کا test کیا اور YouTube پر deploy کیا")
    assert got == "ہم نے اے پی آئی کا ٹیسٹ کیا اور یوٹیوب پر ڈیپلائے کیا"
    assert not any("a" <= ch.lower() <= "z" for ch in got)


def test_the_dictionary_is_urdu_on_the_right_hand_side():
    """A Latin value would be a silent no-op, and easy to add by accident."""
    for english, urdu in WORDS.items():
        assert english == english.lower(), english
        assert not any("a" <= ch.lower() <= "z" for ch in urdu), english


# ----------------------------------------------------------------------
# the robot
# ----------------------------------------------------------------------


def test_a_mid_sentence_clause_does_not_get_a_full_stop():
    """THE robotic reading. A voice given "۔" drops its pitch and stops, so a
    sentence delivered as four clauses was read as four flat statements."""
    assert close("اور پھر ہم نے دیکھا", final=False).endswith("،")
    assert not close("اور پھر ہم نے دیکھا", final=False).endswith("۔")


def test_the_clause_that_ends_the_utterance_does_get_one():
    assert close("یہ ہو گیا", final=True) == "یہ ہو گیا۔"


def test_an_ending_that_is_already_there_is_never_doubled():
    """Caught by a real test failure: an English full stop coming back from
    the translator was followed by an Urdu one, and "بات.۔" is read as a stop,
    a pause, and another stop."""
    for ending in ("۔", "؟", "!", ".", "?", "،", ","):
        text = "کچھ بات" + ending
        assert close(text, final=True) == text
        assert close(text, final=False) == text


def test_finality_is_read_off_the_english_not_the_urdu():
    """The English carries recogniser punctuation. The Urdu often does not,
    because the translator is told not to invent an ending the speaker did not
    imply."""
    assert ends_sentence("So that is done.", "stable") is True
    assert ends_sentence("and then we", "max_words") is False
    # The recogniser or the VAD saying the utterance ended settles it.
    assert ends_sentence("and then we", "final") is True
    assert ends_sentence("and then we", "vad") is True


def test_for_speech_does_both_jobs_in_the_right_order():
    assert for_speech("ہم نے API دیکھا", final=False) == "ہم نے اے پی آئی دیکھا،"


def test_speakable_is_the_same_pipeline_with_the_decoration_stripped():
    assert speakable('"ہم نے API دیکھا"', final=True) == "ہم نے اے پی آئی دیکھا۔"


def test_nothing_in_nothing_out():
    assert for_speech("") == ""
    assert close("   ", final=False) == ""
    assert pronounceable("") == ""


# ----------------------------------------------------------------------
# asking the voice to hurry instead of stretching what it already said
# ----------------------------------------------------------------------


def test_a_speed_factor_becomes_the_percentage_both_providers_expect():
    assert rate_percent(1.08) == "+8%"
    assert rate_percent(1.0) == "+0%"
    assert rate_percent(0.92) == "-8%"


def test_the_edge_voice_combines_its_base_pace_with_the_request():
    """One number, not two adjustments: Edge applies a single rate, so asking
    for -8% and then +10% separately would mean one of them silently
    winning."""
    voice = EdgeTextToSpeech()
    assert voice.speaking_rate == "-8%"
    assert voice.request_rate == "-8%"

    voice.set_speed(1.10)
    assert voice.request_rate == "+1%"     # 0.92 * 1.10

    voice.set_speed(1.0)
    assert voice.request_rate == "-8%"


def test_the_speed_request_is_capped():
    """Past about 1.2x a voice stops sounding hurried and starts sounding
    comic. Being further behind than that is what dropping a unit is for."""
    voice = EdgeTextToSpeech()
    assert voice.set_speed(3.0) == pytest.approx(1.20)
    assert voice.set_speed(0.5) == pytest.approx(1.0)   # never slower
    assert voice.set_speed("nonsense") == pytest.approx(1.0)


def test_every_real_provider_can_be_asked_and_the_null_one_cannot():
    """The pipeline falls back to DSP time-stretching for anything that
    cannot be asked, so this is the switch that decides which path runs."""
    for provider in (
        EdgeTextToSpeech(),
        AzureTextToSpeech(api_key="k", region="r"),
        ElevenLabsTextToSpeech(api_key="k"),
    ):
        assert getattr(provider, "supports_speed", False) is True
    assert getattr(NullTextToSpeech(), "supports_speed", False) is False


def test_azure_asks_for_the_speed_in_the_ssml():
    voice = VoiceId("azure", "ur-PK-UzmaNeural")
    assert 'rate="+8%"' in build_ssml("ہیلو", voice, rate=1.08)
    assert "prosody" not in build_ssml("ہیلو", voice)


def test_the_pronunciation_pass_reaches_the_voice_through_edge():
    """End to end through the real streaming logic, fake network and decoder,
    to prove nothing between here and the request undoes it."""
    import asyncio

    sent: list[str] = []

    class FakeEdge(EdgeTextToSpeech):
        async def _stream_mp3(self, text, voice_name):
            sent.append(text)
            yield b"x" * 4000

        def _decode(self, mp3: bytes):
            return np.zeros(len(mp3), dtype=np.int16)

    async def run():
        async for _ in FakeEdge().synthesize(
            "ہم نے API دیکھا", VoiceId("edge", "ur-PK-UzmaNeural")
        ):
            pass

    asyncio.run(run())
    assert sent == ["ہم نے اے پی آئی دیکھا۔"]
