"""LLM translation with rolling context, register control and terminology.

Spec reference: §8.4, §7.7.

A plain machine-translation endpoint cannot be steered the way this product
needs. Three things have to be controllable:

  * CONTEXT      - "لیکن production میں نہیں" is a meaningless fragment alone;
                   after "We already tested the API" it plainly means
                   "but not in production."
  * REGISTER     - the output must sound like a competent professional, not
                   like a translation. First person as the speaker. No
                   "he says that". No hedging, no meta-commentary, no
                   apologising for ambiguity.
  * TERMINOLOGY  - "deployment" stays "deployment" (§8.4.3).

And one hard rule: NEVER add content that was not said. If a segment is
unintelligible, emit nothing rather than guess - a confident invention in
front of a client is the worst failure this component has.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re

from app.models.telemetry import now
from app.providers.base import (
    ProviderMisconfigured,
    ProviderRateLimited,
    ProviderUnavailable,
    RollingContext,
    TranslatedUnit,
    TranslationUnit,
    Translator,
)
from app.models.terminology import Terminology

LANGUAGE_NAMES = {
    "en": "English",
    "ur": "Urdu",
}

SYSTEM_PROMPT = """\
You are a live interpreter. You render {source} speech into {target}.

Rules, in priority order:

1. TRANSLATE ALL OF IT. The input may be one clause or several sentences. \
Render EVERY sentence, in the order given. Never summarise, never condense, \
never translate only the last part and never drop a sentence because it \
sounds like a greeting or an introduction. If the input has four sentences, \
your output has four sentences.
2. NEVER add, infer or embellish. Translate only what is in the input. If it \
is unintelligible or empty, return an empty string. A confident invention is \
worse than a gap.
3. Speak as the speaker, in the first person. Never "he says that" or "the \
speaker asks". You ARE the voice.
4. WRITE FOR THE EAR. This will be spoken aloud, not read. Use the words a \
Pakistani professional actually says out loud, in natural {target} word \
order - never a word-for-word mapping of the {source} sentence. Keep clauses \
short enough to say in one breath. Competent and direct: not literary, not \
casual, not stiff.
5. No meta-commentary. No notes, no brackets, no apologies for ambiguity, no \
alternative renderings. Output the translation and nothing else.
6. Numbers, dates, quantities and currency: reproduce EXACTLY. Render digits \
as digits. Never round, never approximate, never convert units or currency.
7. Names of people, products and companies stay as they are - transliterate \
them, do not translate them and do not omit them.
8. The input may begin or end mid-sentence, because it is a live stream. \
Translate the part you were given; do not complete it or add closing \
punctuation that was not implied.
{terminology}
Output: the {target} translation only."""

#: Sent as a second attempt when the first reply is far shorter than the input.
COMPLETENESS_RETRY = (
    "That reply left most of the input untranslated. Translate the ENTIRE "
    "text below - every sentence, in order, including greetings and "
    "introductions. Output only the {target} translation."
)

BREVITY_NOTE = (
    "\n7. The listener is falling behind. Prefer the most concise natural "
    "phrasing that preserves all facts."
)


def sentence_count(text: str) -> int:
    """How many sentences the model was given. One is the common case."""
    return len([
        piece for piece in re.split(r"[.!?۔؟…]+", text or "") if piece.strip()
    ])


def looks_truncated(source: str, translated: str, *, floor: float = 0.4) -> bool:
    """Is the translation missing whole sentences of the input?

    Scoped tightly, because the fix costs a round trip and a round trip is
    latency the listener hears. The failure being caught is specifically a
    MULTI-SENTENCE unit rendered as one sentence:

        "Hey, welcome back. I'm David. And I'm Alice. Hello, everyone."
     -> "ہیلو، سب کو۔"

    A single clause cannot drop a sentence, so a single clause is never
    retried however terse it comes back - and terse is often correct, since
    Urdu can say "کیوں؟" for "So why is that, then?".
    """
    if not translated.strip():
        return False  # a deliberate refusal (rule 2), not a truncation

    asked = sentence_count(source)
    if asked < 2:
        return False

    # Sentences, not words, because sentences are what went missing. Word
    # ratios are a poor proxy: Urdu can render "So why is that, then?" as
    # "کیوں؟" and be perfectly complete.
    if sentence_count(translated) * 2 <= asked:
        return True

    source_words = len(source.split())
    return source_words >= 8 and len(translated.split()) < floor * source_words


class LlmTranslator(Translator):
    """OpenAI-compatible chat completion used as a steerable translator.

    Works against any OpenAI-compatible endpoint via `base_url`, so switching
    vendor is a config change (§8.6).
    """

    name = "llm-translator"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout_s: float = 8.0,
        max_output_tokens: int = 300,
        retry_truncated: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model if model != "gpt-4.1-mini" else "gpt-4o-mini"
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.retry_truncated = retry_truncated
        self._client = None

    def _get_client(self):  # pragma: no cover - network path
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderMisconfigured("OPENAI_API_KEY is not set")
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderMisconfigured("pip install openai") from exc
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------

    def build_prompt(
        self, unit: TranslationUnit, terms: Terminology | None
    ) -> str:
        """Exposed so the prompt can be inspected and regression-tested."""
        fragment = ""
        if terms is not None:
            # Narrow to terms actually present, so a short unit does not carry
            # a 200-word glossary and the latency that comes with it.
            relevant = terms.find_in(unit.text)
            if relevant or terms.preferred:
                sub = Terminology(
                    locked=[t for t in terms.locked if t in relevant],
                    entities=[e for e in terms.entities if e in relevant],
                    preferred=terms.preferred,
                )
                block = sub.prompt_fragment()
                if block:
                    fragment = "\n" + block + "\n"
        prompt = SYSTEM_PROMPT.format(
            source=LANGUAGE_NAMES.get(unit.source_language, unit.source_language),
            target=LANGUAGE_NAMES.get(unit.target_language, unit.target_language),
            terminology=fragment,
        )
        if unit.prefer_brevity:
            prompt += BREVITY_NOTE
        return prompt

    async def translate(
        self,
        unit: TranslationUnit,
        ctx: RollingContext,
        terms: Terminology | None = None,
    ) -> TranslatedUnit:
        text = unit.text.strip()
        if not text:
            return TranslatedUnit("", unit)

        client = self._get_client()
        messages = [{"role": "system", "content": self.build_prompt(unit, terms)}]
        history = ctx.render() if ctx is not None else ""
        if history:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Earlier in this meeting, for context only - do not "
                        f"translate this:\n{history}"
                    ),
                }
            )
        messages.append({"role": "user", "content": text})

        try:  # pragma: no cover - network path
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=self.max_output_tokens,
                ),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderUnavailable(
                f"translation exceeded {self.timeout_s}s deadline"
            ) from exc
        except Exception as exc:
            detail = str(exc)
            if "rate" in detail.lower() or "429" in detail:
                raise ProviderRateLimited(detail) from exc
            raise ProviderUnavailable(detail) from exc

        out = (resp.choices[0].message.content or "").strip()

        if self.retry_truncated and looks_truncated(text, out):
            # The failure this catches, observed verbatim: five sentences in,
            # one sentence out.
            #     "Hey, welcome back. I'm David. And I'm Alice. Hello,
            #      everyone. So Alice,"
            #  -> "ہیلو، سب کو۔ تو ایلس،"
            # Told it is an interpreter working "one fragment at a time", the
            # model treats a multi-sentence unit as one fragment and renders
            # only the tail. Rule 1 now forbids it; this is the safety net,
            # because a dropped sentence is invisible to the listener - they
            # cannot know something was said.
            messages.append({"role": "assistant", "content": out})
            messages.append({
                "role": "user",
                "content": COMPLETENESS_RETRY.format(
                    target=LANGUAGE_NAMES.get(
                        unit.target_language, unit.target_language
                    )
                ) + "\n\n" + text,
            })
            with contextlib.suppress(Exception):  # pragma: no cover - network
                again = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=self.max_output_tokens,
                    ),
                    timeout=self.timeout_s,
                )
                second = (again.choices[0].message.content or "").strip()
                if second and not looks_truncated(text, second):
                    out = second

        return TranslatedUnit(text=out, source=unit, at=now())
