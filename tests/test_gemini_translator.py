"""Translation through Gemini's OpenAI-compatible endpoint.

The reason this exists: OpenAI has no free tier, and a day of testing on the
paid realtime model emptied the account in the middle of a call. Gemini does
have a free tier and speaks the same protocol, so the translator can run at no
cost. Whether it is reached correctly is worth pinning down, because the
failure mode - a key sent to the wrong address - looks exactly like a bad key.
"""

from __future__ import annotations

import pytest

from app.providers.factory import (
    GEMINI_BASE_URL,
    GEMINI_DEFAULT_MODEL,
    build_translator,
    usable,
)


class Providers:
    translator_model = "gpt-4o-mini"


def keys(monkeypatch, **values):
    monkeypatch.setattr(
        "app.providers.factory.get_secret", lambda name: values.get(name, "")
    )


# ------------------------------------------------------------------ the key


def test_a_placeholder_key_is_not_a_key():
    """The shipped .env contains 'your_key_here'. Treating that as configured
    sends every request to a guaranteed 401, reported as a provider outage."""
    assert not usable("your_gemini_key_here")
    assert not usable("")
    assert not usable(None)
    assert usable("AQ.reallyLooksLikeAKey")


# --------------------------------------------------------------- the choice


def test_gemini_is_tried_before_openai(monkeypatch):
    """Order is the point: the free one first, the paid one as the backstop."""
    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini", OPENAI_API_KEY="sk-openai")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    chain = build_translator(Providers())

    first, second = chain.translators
    assert first.api_key == "AQ.gemini"
    assert first.base_url == GEMINI_BASE_URL
    assert first.model == GEMINI_DEFAULT_MODEL
    assert second.api_key == "sk-openai"
    assert second.base_url is None


def test_groq_is_tried_before_gemini(monkeypatch):
    """Groq's free tier is counted in thousands of requests a day; Google's for
    2.5-flash is twenty. Cheapest and most generous first."""
    from app.providers.factory import GROQ_BASE_URL, GROQ_DEFAULT_MODEL

    keys(monkeypatch, GROQ_API_KEY="gsk_x", GEMINI_API_KEY="AQ.gemini")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    chain = build_translator(Providers())

    assert chain.translators[0].base_url == GROQ_BASE_URL
    assert chain.translators[0].model == GROQ_DEFAULT_MODEL
    assert chain.translators[1].base_url == GEMINI_BASE_URL


def test_the_default_gemini_model_is_not_the_twenty_a_day_one(monkeypatch):
    """gemini-2.5-flash allows 20 free requests PER DAY. One sentence is one
    request, so it dies inside a single call. This must never be the default
    again."""
    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert GEMINI_DEFAULT_MODEL != "gemini-2.5-flash"
    # And not the one the API now refuses outright with a 404:
    # "no longer available to new users".
    assert GEMINI_DEFAULT_MODEL != "gemini-2.5-flash-lite"
    assert build_translator(Providers()).model == GEMINI_DEFAULT_MODEL


def test_openai_is_used_when_there_is_no_gemini_key(monkeypatch):
    keys(monkeypatch, OPENAI_API_KEY="sk-openai")

    translator = build_translator(Providers())

    assert translator.api_key == "sk-openai"
    assert translator.base_url is None
    assert translator.model == "gpt-4o-mini"


def test_a_placeholder_gemini_key_falls_back_to_openai(monkeypatch):
    keys(monkeypatch, GEMINI_API_KEY="your_key_here", OPENAI_API_KEY="sk-openai")

    translator = build_translator(Providers())

    assert translator.api_key == "sk-openai"
    assert translator.base_url is None


def test_the_gemini_model_can_be_overridden(monkeypatch):
    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3-flash")

    assert build_translator(Providers()).model == "gemini-3-flash"


def test_one_key_gives_one_translator_not_a_chain(monkeypatch):
    """No point in the indirection when there is nothing to fall back to."""
    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini")
    assert build_translator(Providers()).base_url == GEMINI_BASE_URL


def test_the_openai_model_setting_does_not_leak_into_gemini(monkeypatch):
    """A GPT model name sent to Gemini is a 404 that reads as 'Gemini is
    broken'. The account's OpenAI model choice must not travel with the key."""
    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    providers = Providers()
    providers.translator_model = "gpt-4o-mini"

    assert "gpt" not in build_translator(providers).model


# ------------------------------------------------------- the shared machinery


def test_gemini_keeps_the_prompt_and_the_terminology(monkeypatch):
    """Gemini is the same class as before, not a reimplementation - so the
    interpreter prompt, the rolling context and the glossary all still apply.
    A second copy of that prompt is a second thing to keep in step."""
    from app.models.terminology import Terminology
    from app.providers.base import TranslationUnit

    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini")
    translator = build_translator(Providers())

    unit = TranslationUnit(
        text="ہم نے deployment کر دیا ہے",
        source_language="ur",
        target_language="en",
    )
    prompt = translator.build_prompt(unit, Terminology(locked=["deployment"]))

    assert "live interpreter" in prompt
    assert "Urdu" in prompt and "English" in prompt
    assert "deployment" in prompt


def test_the_endpoint_is_the_documented_one():
    """A wrong address is indistinguishable from a wrong key in the error it
    produces, so the address is pinned rather than remembered."""
    assert GEMINI_BASE_URL == (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert GEMINI_BASE_URL.endswith("/")  # the SDK joins paths onto this


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-3-flash"])
def test_gemini_model_names_survive_the_openai_alias_rewrite(monkeypatch, model):
    """LlmTranslator rewrites one legacy OpenAI model name. Gemini names must
    pass through it untouched."""
    keys(monkeypatch, GEMINI_API_KEY="AQ.gemini")
    monkeypatch.setenv("GEMINI_MODEL", model)

    assert build_translator(Providers()).model == model


def test_the_cascade_is_the_default_even_with_an_openai_key(monkeypatch):
    """The realtime model is billed per MINUTE OF AUDIO with no free tier.
    Merely having a key on the account must not turn every minute of testing
    into spend, and must not leave the outgoing direction to die mid-call when
    the balance empties - which is indistinguishable from broken software."""
    from app.providers.cascade_s2st import CascadeSpeechTranslator
    from app.providers.factory import build_s2st

    monkeypatch.setattr(
        "app.providers.factory.get_secret",
        lambda name: "sk-openai" if name == "OPENAI_API_KEY" else "",
    )
    monkeypatch.delenv("USE_REALTIME", raising=False)

    providers = {"asr": object(), "translator": object(), "tts": object()}
    assert isinstance(build_s2st(providers), CascadeSpeechTranslator)


def test_the_realtime_model_is_opt_in(monkeypatch):
    from app.providers.factory import build_s2st

    monkeypatch.setattr(
        "app.providers.factory.get_secret",
        lambda name: "sk-openai" if name == "OPENAI_API_KEY" else "",
    )
    monkeypatch.setenv("USE_REALTIME", "1")

    providers = {"asr": object(), "translator": object(), "tts": object()}
    assert type(build_s2st(providers)).__name__ == "OpenAiSpeechTranslator"


def test_asking_for_realtime_without_a_key_still_gets_the_cascade(monkeypatch):
    from app.providers.cascade_s2st import CascadeSpeechTranslator
    from app.providers.factory import build_s2st

    monkeypatch.setattr("app.providers.factory.get_secret", lambda name: "")
    monkeypatch.setenv("USE_REALTIME", "1")

    providers = {"asr": object(), "translator": object(), "tts": object()}
    assert isinstance(build_s2st(providers), CascadeSpeechTranslator)


# ======================================================================
# the repeat guard
# ======================================================================


def _guarded():
    from app.providers.llm_translator import LlmTranslator

    return LlmTranslator(api_key="test-key")


def test_a_recap_of_already_spoken_sentences_is_stripped():
    """Observed in a live call: handed rolling context and a garbled
    fragment, the model returned the last minute of the meeting re-translated
    with the fragment on the end."""
    from app.providers.llm_translator import _norm_sentence

    t = _guarded()
    t._recent_out.append(_norm_sentence("My name is Heshmat Ali."))
    t._recent_out.append(_norm_sentence("And I am from Pakistan."))
    out = t._strip_already_spoken(
        "My name is Heshmat Ali. And I am from Pakistan. Kam Sem University"
    )
    assert out == "Kam Sem University"


def test_a_genuine_single_sentence_repeat_survives():
    """A speaker repeating themselves must still be voiced - only
    multi-sentence recaps are filtered."""
    from app.providers.llm_translator import _norm_sentence

    t = _guarded()
    t._recent_out.append(_norm_sentence("My name is Heshmat Ali."))
    kept = t._strip_already_spoken("My name is Heshmat Ali.")
    assert kept == "My name is Heshmat Ali."


def test_short_recurring_sentences_are_never_stripped():
    """"okay." and "yes." recur in every meeting and belong to whoever
    says them."""
    from app.providers.llm_translator import _norm_sentence

    t = _guarded()
    t._recent_out.append(_norm_sentence("okay."))
    out = t._strip_already_spoken("Okay. Let us continue with the plan.")
    assert "Okay." in out
