"""Several translators, tried in order, so one exhausted quota is not the end.

Every free tier has a cap, and the caps are small enough to hit inside one
conversation. Google's free tier for gemini-2.5-flash is TWENTY requests per
day; one request is one sentence, so a two-minute call exhausts it and the
outgoing direction dies mid-sentence with a 429.

A single provider therefore cannot be the design. This holds an ordered list -
cheapest first - and moves to the next one when the current one says it is out
of quota. It remembers where it got to, so the exhausted provider is not tried
again on every sentence and the caller does not pay that round trip.

What it deliberately does NOT do is retry a real failure. A bad key, a refused
request or a network fault are not quota problems and must surface, not hide
behind a second attempt at a different vendor.
"""

from __future__ import annotations

from app.models.terminology import Terminology
from app.providers.base import (
    ProviderRateLimited,
    ProviderUnavailable,
    RollingContext,
    TranslatedUnit,
    TranslationUnit,
    Translator,
)

#: What a spent free tier looks like from an OpenAI-compatible endpoint. The
#: SDK raises its own error types, so the text is what we have to go on.
QUOTA_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "insufficient_quota",
    "credit balance",
    "no credits",
)


def is_quota_problem(exc: BaseException) -> bool:
    """Is this "come back later or pay", rather than "you are doing it wrong"?"""
    if isinstance(exc, ProviderRateLimited):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


class ChainTranslator(Translator):
    """Try each translator in turn; skip the ones that are out of quota."""

    name = "chain-translator"

    def __init__(self, translators, *, on_notice=None) -> None:
        self.translators = [t for t in translators if t is not None]
        self.on_notice = on_notice
        self._index = 0
        self._exhausted: set[int] = set()

    @property
    def current(self):
        return self.translators[self._index] if self.translators else None

    def _say(self, message: str) -> None:
        if self.on_notice:
            try:
                self.on_notice(message)
            except Exception:
                pass

    def describe(self, index: int) -> str:
        translator = self.translators[index]
        return getattr(translator, "model", None) or getattr(
            translator, "name", "translator"
        )

    async def translate(
        self,
        unit: TranslationUnit,
        ctx: RollingContext,
        terms: Terminology | None = None,
    ) -> TranslatedUnit:
        if not self.translators:
            raise ProviderUnavailable("no translator is configured")

        last: BaseException | None = None
        for offset in range(len(self.translators)):
            index = (self._index + offset) % len(self.translators)
            if index in self._exhausted:
                continue
            try:
                result = await self.translators[index].translate(unit, ctx, terms)
            except Exception as exc:
                if not is_quota_problem(exc):
                    # A real fault. Say it rather than quietly trying another
                    # vendor and reporting whatever that one thinks.
                    raise
                last = exc
                self._exhausted.add(index)
                self._say(
                    f"{self.describe(index)} is out of quota - "
                    f"switching translator"
                )
                continue

            if index != self._index:
                self._index = index
                self._say(f"translating with {self.describe(index)}")
            return result

        raise ProviderUnavailable(
            "every translator is out of quota: " + (str(last) if last else "")
        )
