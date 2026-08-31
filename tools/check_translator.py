"""Verify the translator with EXACTLY ONE request.

    python tools/check_translator.py

One request, one sentence, one printed answer. Nothing else runs - not the app,
not the microphone, not the recogniser.

It exists because free tiers are counted in requests, and some of them are
counted in tens: Google allows twenty a day for gemini-2.5-flash. Starting the
whole application to find out whether a key works can spend a day's quota
before you have said a word. This spends one.
"""

from __future__ import annotations

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from app.models.config import Config  # noqa: E402
from app.providers.base import RollingContext, TranslationUnit  # noqa: E402
from app.providers.factory import build_translator  # noqa: E402

URDU = "\u0645\u06cc\u0631\u0627 \u0646\u0627\u0645 \u062d\u0634\u0645\u062a \u0639\u0644\u06cc \u06c1\u06d4"


def describe(translator) -> None:
    chain = getattr(translator, "translators", None)
    if chain is None:
        print(f"  {translator.model}   {translator.base_url or 'api.openai.com'}")
        return
    print("  chain, in order:")
    for index, one in enumerate(chain, 1):
        print(f"    {index}. {one.model}   {one.base_url or 'api.openai.com'}")


def main() -> int:
    config = Config.load()
    translator = build_translator(config.providers, on_notice=lambda m: print(f"  {m}"))

    print("Translator:")
    describe(translator)

    unit = TranslationUnit(text=URDU, source_language="ur", target_language="en")
    print(f"\nSending ONE request.\n  Urdu:    {URDU}")

    try:
        result = asyncio.run(translator.translate(unit, RollingContext()))
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        text = str(exc).lower()
        if "quota" in text or "429" in text or "exhausted" in text:
            print("\nThat is a spent quota, not a broken key. Either wait for the")
            print("reset, set GEMINI_MODEL to a model with its own quota, or add")
            print("a GROQ_API_KEY - Groq's free tier is thousands a day.")
        elif "api key" in text or "401" in text:
            print("\nThe key was rejected. Check it in .env.")
        return 1

    english = (result.text or "").strip()
    print(f"  English: {english or '(nothing)'}")
    print()
    if not english:
        print("It answered, but with nothing. The key and quota are fine; the")
        print("prompt or the model is the problem.")
        return 1
    print("The translator works. One request spent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
