"""Falling through exhausted free tiers instead of dying on one."""

from __future__ import annotations

import asyncio

import pytest

from app.providers.base import (
    ProviderRateLimited,
    ProviderUnavailable,
    RollingContext,
    TranslatedUnit,
    TranslationUnit,
)
from app.providers.chain_translator import ChainTranslator, is_quota_problem


class Stub:
    def __init__(self, model, *, fail=None, answer=None):
        self.model = model
        self.fail = fail
        self.answer = answer or model
        self.calls = 0

    async def translate(self, unit, ctx, terms=None):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return TranslatedUnit(self.answer, unit)


def unit():
    return TranslationUnit(text="کیسے ہیں", source_language="ur",
                           target_language="en")


def translate(chain):
    return asyncio.run(chain.translate(unit(), RollingContext()))


# ---------------------------------------------------------------- recognition


def test_a_spent_free_tier_is_recognised_however_it_is_worded():
    """The real messages, verbatim from tonight."""
    assert is_quota_problem(ProviderRateLimited("slow down"))
    assert is_quota_problem(Exception(
        "Error code: 429 - quota exceeded for metric "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests"
    ))
    assert is_quota_problem(Exception(
        "insufficient_quota.credit_balance_exhausted"
    ))
    assert is_quota_problem(Exception("You have no credits remaining"))
    assert is_quota_problem(Exception("RESOURCE_EXHAUSTED"))


def test_a_real_fault_is_not_a_quota_problem():
    assert not is_quota_problem(Exception("Invalid API key"))
    assert not is_quota_problem(Exception("model not found"))
    assert not is_quota_problem(Exception("connection reset by peer"))


# ------------------------------------------------------------------ behaviour


def test_the_first_translator_is_used_while_it_works():
    first, second = Stub("groq"), Stub("gemini")
    chain = ChainTranslator([first, second])

    assert translate(chain).text == "groq"
    assert translate(chain).text == "groq"
    assert second.calls == 0


def test_an_exhausted_quota_falls_through_to_the_next():
    first = Stub("gemini", fail=Exception("429 quota exceeded, limit: 20"))
    second = Stub("openai")
    chain = ChainTranslator([first, second])

    assert translate(chain).text == "openai"


def test_an_exhausted_translator_is_not_asked_again():
    """Twenty a day means it stays exhausted. Asking every sentence costs a
    round trip of latency for a guaranteed failure."""
    first = Stub("gemini", fail=Exception("429 quota exceeded"))
    second = Stub("openai")
    chain = ChainTranslator([first, second])

    translate(chain)
    translate(chain)
    translate(chain)

    assert first.calls == 1
    assert second.calls == 3


def test_a_real_fault_is_raised_not_papered_over():
    """A wrong key must not become "the other vendor says X"."""
    first = Stub("gemini", fail=ProviderUnavailable("Invalid API key"))
    second = Stub("openai")
    chain = ChainTranslator([first, second])

    with pytest.raises(ProviderUnavailable, match="Invalid API key"):
        translate(chain)
    assert second.calls == 0


def test_running_out_everywhere_says_so():
    chain = ChainTranslator([
        Stub("groq", fail=Exception("429 quota")),
        Stub("gemini", fail=Exception("429 quota")),
    ])

    with pytest.raises(ProviderUnavailable, match="out of quota"):
        translate(chain)


def test_the_switch_is_announced():
    """The user needs to know which provider is speaking for them, and that a
    free tier has run out - silently degrading is how a whole day gets spent
    looking for a fault that is a spent quota."""
    said = []
    chain = ChainTranslator(
        [Stub("gemini", fail=Exception("429 quota")), Stub("openai")],
        on_notice=said.append,
    )

    translate(chain)

    assert any("out of quota" in m for m in said)
    assert any("openai" in m for m in said)


def test_no_translators_at_all_is_reported():
    with pytest.raises(ProviderUnavailable, match="no translator"):
        translate(ChainTranslator([]))
