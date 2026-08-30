"""One place that turns configuration into provider objects.

Spec reference: §8.6. Nothing above this line knows a vendor name; provider
choice is configuration, and this is the only function that reads it.

It exists because there are now two entry points - meeting mode and listen
mode - and a second copy of this mapping would be a second place to forget
when a provider is added.
"""

from __future__ import annotations

from app.models.config import Config, get_secret


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
        "translator": LlmTranslator(
            get_secret("OPENAI_API_KEY"), model=p.translator_model
        ),
        "tts": tts,
    }
    if include_s2st:
        from app.providers.openai_s2st import OpenAiSpeechTranslator

        providers["s2st"] = OpenAiSpeechTranslator(get_secret("OPENAI_API_KEY"))
    return providers


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

