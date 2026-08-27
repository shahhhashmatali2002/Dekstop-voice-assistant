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
import os

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
You are a live interpreter in a professional business meeting. You render \
{source} speech into {target}, in real time, one fragment at a time.

Rules, in priority order:

1. NEVER add, infer or embellish. Translate only what is in the fragment. If \
it is unintelligible or empty, return an empty string. A confident invention \
is worse than a gap.
2. Speak as the speaker, in the first person. Never "he says that" or "the \
speaker asks". You ARE the voice.
3. Business-meeting register: competent, direct, professional. Not literary, \
not casual, not stiff.
4. No meta-commentary. No notes, no brackets, no apologies for ambiguity, no \
alternative renderings. Output the translation and nothing else.
5. Numbers, dates, quantities and currency: reproduce EXACTLY. Render digits \
as digits. Never round, never approximate, never convert units or currency.
6. The fragment may be mid-sentence, because it is a live stream. Translate \
it as a fragment; do not complete it or add closing punctuation that was not \
implied.
{terminology}
Output: the {target} translation only."""

BREVITY_NOTE = (
    "\n7. The listener is falling behind. Prefer the most concise natural "
    "phrasing that preserves all facts."
)


class LlmTranslator(Translator):
    """OpenAI-compatible chat completion used as a steerable translator.

    Works against any OpenAI-compatible endpoint via `base_url`, so switching
    vendor is a config change (§8.6).
    """

    name = "llm-translator"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        timeout_s: float = 3.0,
        max_output_tokens: int = 300,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url
        # §8.7: every request carries a deadline. A translation that has not
        # returned in 3 s is worthless - cancel it and show the gap.
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
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
        return TranslatedUnit(text=out, source=unit, at=now())
