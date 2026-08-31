"""One place that turns configuration into provider objects.

Spec reference: §8.6. Nothing above this line knows a vendor name; provider
choice is configuration, and this is the only function that reads it.

It exists because there are now two entry points - meeting mode and listen
mode - and a second copy of this mapping would be a second place to forget
when a provider is added.
"""

from __future__ import annotations

import os

from app.models.config import Config, get_secret

#: Google serve Gemini behind an OpenAI-compatible surface, so the existing
#: translator talks to it unchanged - same request shape, same SDK, only the
#: address and the key differ. Worth knowing why that matters here: Gemini has
#: a genuine free tier, and OpenAI does not. A day of testing on the paid
#: realtime model emptied the account mid-call and cost hours of debugging
#: something that was never broken.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

#: Free-tier friendly, and current. Two names have already been ruled out by
#: the API itself, so both are recorded here rather than rediscovered:
#:
#:   gemini-2.5-flash       20 free requests PER DAY - one sentence is one
#:                          request, so a single short call exhausts it
#:   gemini-2.5-flash-lite  "no longer available to new users" (404)
#:
#: Override with GEMINI_MODEL when Google moves it again.
GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"

#: Groq is also OpenAI-compatible, and its free tier is counted in thousands of
#: requests a day rather than tens.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"


def usable(key: str | None) -> bool:
    """A key that is actually set, not the placeholder from the sample .env."""
    return bool(key) and not key.startswith("your_")


def build_translator(p, *, on_notice=None):
    """Every translator we have a key for, cheapest first.

    Not one provider - a chain. Every free tier has a cap small enough to hit
    inside a single conversation, so the question is never "which vendor" but
    "what happens when this one runs out". The chain moves to the next one on a
    quota error and keeps translating.

    All of them are the same class: Groq and Gemini both serve an
    OpenAI-compatible endpoint, so the interpreter prompt, the rolling context,
    the terminology injection and the truncation retry are shared rather than
    reimplemented three times.
    """
    import os

    from app.providers.chain_translator import ChainTranslator
    from app.providers.llm_translator import LlmTranslator

    candidates = []

    groq_key = get_secret("GROQ_API_KEY")
    if usable(groq_key):
        candidates.append(LlmTranslator(
            groq_key,
            model=os.environ.get("GROQ_MODEL") or GROQ_DEFAULT_MODEL,
            base_url=GROQ_BASE_URL,
        ))

    gemini_key = get_secret("GEMINI_API_KEY")
    if usable(gemini_key):
        candidates.append(LlmTranslator(
            gemini_key,
            model=os.environ.get("GEMINI_MODEL") or GEMINI_DEFAULT_MODEL,
            base_url=GEMINI_BASE_URL,
        ))

    openai_key = get_secret("OPENAI_API_KEY")
    if usable(openai_key):
        candidates.append(
            LlmTranslator(openai_key, model=p.translator_model)
        )

    if not candidates:
        # Keep the old behaviour: a translator that reports a missing key when
        # used, rather than a crash at startup.
        candidates.append(LlmTranslator("", model=p.translator_model))

    if len(candidates) == 1:
        return candidates[0]
    return ChainTranslator(candidates, on_notice=on_notice)


def build_providers(config: Config, *, include_s2st: bool = True) -> dict:
    """Construct the configured provider stack.

    `include_s2st` is False for listen mode, which has no outgoing direction:
    Urdu is not an available output language for any speech-to-speech model
    (§8.2), and a one-way English-to-Urdu tool has nothing for one to do.
    """
    from app.providers.deepgram_asr import DeepgramSpeechToText
    from app.providers.llm_translator import LlmTranslator
    from app.providers.tts import (
        AzureTextToSpeech,
        EdgeTextToSpeech,
        ElevenLabsTextToSpeech,
        NullTextToSpeech,
    )

    p = config.providers
    azure_key = get_secret("AZURE_SPEECH_KEY")
    azure_region = get_secret("AZURE_SPEECH_REGION")
    eleven_key = get_secret("ELEVENLABS_API_KEY")

    if (
        p.tts.startswith("azure")
        and azure_key
        and not azure_key.startswith("your_")
        and azure_region
        and not azure_region.startswith("your_")
    ):
        tts = AzureTextToSpeech(azure_key, azure_region)
    elif (
        p.tts.startswith("eleven")
        and eleven_key
        and not eleven_key.startswith("your_")
    ):
        tts = ElevenLabsTextToSpeech(eleven_key)
    else:
        tts = EdgeTextToSpeech()



    providers = {
        "asr": DeepgramSpeechToText(get_secret("DEEPGRAM_API_KEY")),
        "translator": build_translator(p),
        "tts": tts,
    }
    if include_s2st:
        providers["s2st"] = build_s2st(providers)
    return providers


def build_s2st(providers: dict):
    """The cascade by default. The realtime model only when asked for.

    This order is deliberate and it is about money, not quality. The realtime
    model is billed per MINUTE OF AUDIO and has no free tier: leaving it as the
    automatic choice means that merely having a key on the account turns every
    minute of testing into spend, and when the balance empties the outgoing
    direction dies mid-call looking exactly like a software fault. That is not
    a hypothetical - it cost a full day.

    The cascade is billed per SENTENCE OF TEXT, which is cheaper by orders of
    magnitude, and its recognition is already paid for. So it is the default,
    and the better-sounding path is opt-in:

        USE_REALTIME=1      in .env, for a real client call

    Both satisfy the same interface, so this is the only place that decides.
    """
    import os

    wants_realtime = (os.environ.get("USE_REALTIME") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    openai_key = get_secret("OPENAI_API_KEY")
    if wants_realtime and usable(openai_key):
        from app.providers.openai_s2st import OpenAiSpeechTranslator

        return OpenAiSpeechTranslator(openai_key)

    from app.providers.cascade_s2st import CascadeSpeechTranslator

    return CascadeSpeechTranslator(
        providers["asr"], providers["translator"], providers["tts"]
    )


def build_voice_clone(config, *, on_notice=None):
    """A converter and a learner, or (None, None) with a reason said out loud.

    Deliberately does NOT benchmark anything here. The obvious design - time a
    synthetic clip at startup and refuse if it is too slow - fails on its own
    terms: the converter trims non-speech before it does anything, so a
    synthetic clip is trimmed to nothing and the measurement crashes instead of
    returning a number. Real speech is the only valid input, and at startup
    there is none: the source has not said anything yet.

    So the judgement moves to where the evidence is. The pipeline times its
    FIRST real conversion and switches the feature off if that one cost more
    than the budget, which is both honest and only one utterance late.
    """
    from app.audio.formats import RATE_ASR
    from app.providers.base import ProviderError
    from app.providers.voice_clone import VoiceLearner, build_converter

    p = config.providers
    if not getattr(p, "clone_voice", False):
        return None, None

    def say(message: str) -> None:
        if on_notice:
            try:
                on_notice(message)
            except Exception:
                pass

    try:
        backend = getattr(p, "clone_backend", "light")
        converter = build_converter(backend)
    except ProviderError as exc:
        say(f"Speaking in the original voice is off: {exc}")
        return None, None

    say(
        "Speaking in the original speaker's voice: learning voice from audio. "
        "Audio starts in real-time."
    )
    return converter, VoiceLearner(RATE_ASR, min_seconds=2.0)

