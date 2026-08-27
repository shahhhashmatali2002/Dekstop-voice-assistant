"""Tests for the provider layer, VAD, terminology and the engine wiring.

None of this touches a network. The provider classes are tested for the things
that can be wrong without a socket: language guards, prompt construction,
SSML escaping, and cancellation safety.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.audio.formats import DTYPE, RATE_ASR
from app.controllers.outgoing import is_backchannel
from app.controllers.vad import (
    EnergyDetector,
    SpeechEvent,
    VoiceActivityGate,
    frames_of,
)
from app.providers.base import (
    PcmChunk,
    ProviderMisconfigured,
    RollingContext,
    S2stConfig,
    TranslationUnit,
    VoiceId,
)
from app.providers.llm_translator import LlmTranslator
from app.providers.openai_s2st import OUTPUT_LANGUAGES, OpenAiSpeechTranslator
from app.providers.tts import NullTextToSpeech, build_ssml
from app.models.terminology import Terminology


# ======================================================================
# VAD
# ======================================================================


def tone(n, amp=8000, rate=RATE_ASR, freq=200.0):
    t = np.arange(n, dtype=np.float32) / rate
    return (np.sin(2 * np.pi * freq * t) * amp).astype(DTYPE)


def silence(n):
    return np.zeros(n, dtype=DTYPE)


def test_energy_detector_separates_speech_from_silence():
    d = EnergyDetector(-45.0)
    assert d.probability(tone(320)) > 0.9
    assert d.probability(silence(320)) == 0.0


def test_energy_detector_handles_empty_frame():
    assert EnergyDetector().probability(np.empty(0, dtype=DTYPE)) == 0.0


def test_onset_requires_consecutive_speech_frames():
    """A single click must not open an utterance."""
    g = VoiceActivityGate(EnergyDetector(-45.0), onset_frames=3, frame_ms=20)
    assert g.push(tone(320)).event is SpeechEvent.NONE
    assert g.push(silence(320)).event is SpeechEvent.NONE
    assert not g.speaking


def test_onset_fires_after_enough_speech():
    g = VoiceActivityGate(EnergyDetector(-45.0), onset_frames=3, frame_ms=20)
    events = [g.push(tone(320)).event for _ in range(3)]
    assert events[-1] is SpeechEvent.ONSET
    assert g.speaking


def test_onset_carries_preroll_audio():
    """The pre-roll recovers the clipped first phoneme (§6.2)."""
    g = VoiceActivityGate(
        EnergyDetector(-45.0), onset_frames=2, preroll_ms=100, frame_ms=20
    )
    g.push(silence(320))
    g.push(tone(320))
    res = g.push(tone(320))
    assert res.event is SpeechEvent.ONSET
    assert res.audio is not None
    assert len(res.audio) > 320, "onset must include retained pre-roll"


def test_hangover_holds_the_utterance_open_through_a_short_pause():
    g = VoiceActivityGate(
        EnergyDetector(-45.0), onset_frames=1, hangover_ms=200, frame_ms=20
    )
    g.push(tone(320))
    for _ in range(5):  # 100 ms of silence, inside the hangover
        assert g.push(silence(320)).event is SpeechEvent.CONTINUE
    assert g.speaking


def test_offset_fires_once_the_hangover_elapses():
    g = VoiceActivityGate(
        EnergyDetector(-45.0), onset_frames=1, hangover_ms=100, frame_ms=20
    )
    g.push(tone(320))
    events = [g.push(silence(320)).event for _ in range(6)]
    assert SpeechEvent.OFFSET in events
    assert not g.speaking


def test_hangover_is_the_latency_knob():
    """A longer hangover must delay the offset. This is §9.3 made testable."""

    def frames_until_offset(hangover_ms):
        g = VoiceActivityGate(
            EnergyDetector(-45.0),
            onset_frames=1,
            hangover_ms=hangover_ms,
            frame_ms=20,
        )
        g.push(tone(320))
        for i in range(100):
            if g.push(silence(320)).event is SpeechEvent.OFFSET:
                return i
        return 999

    assert frames_until_offset(100) < frames_until_offset(400)


def test_reset_clears_speaking_state():
    g = VoiceActivityGate(EnergyDetector(-45.0), onset_frames=1)
    g.push(tone(320))
    g.reset()
    assert not g.speaking


def test_frames_of_discards_the_short_tail():
    frames = list(frames_of(np.zeros(1000, dtype=DTYPE), RATE_ASR, 20))
    assert len(frames) == 3  # 320 * 3 = 960; the last 40 samples are dropped
    assert all(len(f) == 320 for f in frames)


# ======================================================================
# backchannel suppression (§7.3)
# ======================================================================


@pytest.mark.parametrize("word", ["hmm", "okay", "right", "achha", "jee", "haan"])
def test_short_acknowledgements_are_backchannels(word):
    assert is_backchannel(word, 300)


def test_a_long_utterance_is_never_a_backchannel():
    assert not is_backchannel("okay", 2000)


def test_real_content_is_not_a_backchannel():
    assert not is_backchannel("okay let us deploy", 500)


def test_punctuation_does_not_defeat_detection():
    assert is_backchannel("Okay.", 300)


# ======================================================================
# terminology (§8.4.3)
# ======================================================================


def test_default_terminology_locks_technical_vocabulary():
    t = Terminology()
    assert t.is_locked("deployment")
    assert t.is_locked("PRODUCTION")  # case-insensitive


def test_entities_come_first_in_keyterms():
    """A mangled client name is worse than mangled jargon."""
    t = Terminology(entities=["Innovatix"])
    assert t.keyterms()[0] == "Innovatix"


def test_keyterms_are_capped():
    t = Terminology(locked=[f"term{i}" for i in range(500)])
    assert len(t.keyterms(limit=100)) == 100


def test_keyterms_deduplicate_case_insensitively():
    t = Terminology(locked=["API", "api"], entities=[])
    assert len([k for k in t.keyterms() if k.lower() == "api"]) == 1


def test_find_in_matches_whole_words_only():
    t = Terminology(locked=["API"])
    assert t.find_in("check the API docs") == ["API"]
    assert t.find_in("rapidly") == []


def test_prompt_fragment_mentions_locked_terms():
    frag = Terminology(locked=["staging"], entities=["Innovatix"]).prompt_fragment()
    assert "staging" in frag and "Innovatix" in frag


def test_terminology_roundtrips_through_disk(tmp_path):
    t = Terminology(entities=["Hashmat Ali"], preferred={"go live": "go live"})
    t.lock("blue-green deploy")
    p = tmp_path / "terminology.json"
    t.save(p)
    back = Terminology.load(p)
    assert "Hashmat Ali" in back.entities
    assert back.is_locked("blue-green deploy")
    assert back.preferred["go live"] == "go live"


def test_loading_a_missing_file_gives_defaults():
    assert Terminology.load("/nonexistent/x.json").is_locked("deployment")


def test_remove_clears_from_every_list():
    t = Terminology(entities=["Acme"])
    t.remove("Acme")
    assert "Acme" not in t.entities


# ======================================================================
# rolling context (§8.4.1)
# ======================================================================


def test_context_keeps_only_the_last_few_turns():
    ctx = RollingContext(max_turns=3)
    for i in range(10):
        ctx.add("client", f"line {i}")
    assert len(ctx) == 3
    assert "line 9" in ctx.render()
    assert "line 0" not in ctx.render()


def test_context_respects_the_character_budget():
    ctx = RollingContext(max_turns=100, max_chars=60)
    for i in range(20):
        ctx.add("client", "x" * 30)
    assert len(ctx.render()) <= 70


def test_context_ignores_empty_turns():
    ctx = RollingContext()
    ctx.add("client", "   ")
    assert len(ctx) == 0


def test_context_renders_oldest_first():
    ctx = RollingContext()
    ctx.add("client", "first")
    ctx.add("you", "second")
    rendered = ctx.render()
    assert rendered.index("first") < rendered.index("second")


# ======================================================================
# the S2ST language guard - the §8.2 asymmetry made enforceable
# ======================================================================


def test_urdu_is_not_an_available_output_language():
    """The constraint that forces the whole asymmetric architecture."""
    assert "ur" not in OUTPUT_LANGUAGES
    assert "en" in OUTPUT_LANGUAGES


def test_s2st_refuses_urdu_output_with_a_useful_message():
    tr = OpenAiSpeechTranslator(api_key="test-key")
    with pytest.raises(ProviderMisconfigured) as exc:
        asyncio.run(tr.open(S2stConfig(source_language="en", target_language="ur")))
    assert "cascade" in str(exc.value).lower()


def test_s2st_refuses_without_a_key():
    tr = OpenAiSpeechTranslator(api_key="")
    with pytest.raises(ProviderMisconfigured):
        asyncio.run(tr.open(S2stConfig(source_language="ur", target_language="en")))


def test_supports_output_is_case_insensitive():
    assert OpenAiSpeechTranslator(api_key="k").supports_output("EN")


# ======================================================================
# translation prompt (§8.4.2)
# ======================================================================


def prompt_for(text, terms=None, brevity=False):
    unit = TranslationUnit(text, "en", "ur", prefer_brevity=brevity)
    return LlmTranslator(api_key="k").build_prompt(unit, terms)


def test_prompt_forbids_invention():
    p = prompt_for("hello").lower()
    assert "never add" in p or "never add, infer" in p
    assert "empty string" in p


def test_prompt_demands_first_person():
    assert "first person" in prompt_for("hello").lower()


def test_prompt_protects_numbers():
    p = prompt_for("hello").lower()
    assert "never round" in p and "digits as digits" in p


def test_prompt_names_both_languages():
    p = prompt_for("hello")
    assert "English" in p and "Urdu" in p


def test_prompt_includes_only_relevant_terminology():
    terms = Terminology(locked=["staging", "Kubernetes"])
    p = prompt_for("push it to staging", terms)
    assert "staging" in p
    assert "Kubernetes" not in p, "an unused glossary is latency for nothing"


def test_brevity_note_only_when_the_listener_is_behind():
    assert "falling behind" not in prompt_for("hello")
    assert "falling behind" in prompt_for("hello", brevity=True)


def test_empty_text_needs_no_provider_call():
    tr = LlmTranslator(api_key="")
    unit = TranslationUnit("   ", "en", "ur")
    out = asyncio.run(tr.translate(unit, RollingContext(), None))
    assert out.text == ""


# ======================================================================
# TTS
# ======================================================================


def test_ssml_escapes_dangerous_characters():
    voice = VoiceId("azure", "ur-PK-UzmaNeural")
    out = build_ssml('5 < 6 & "quoted"', voice)
    assert "&lt;" in out and "&amp;" in out and "&quot;" in out
    assert "<voice" in out


def test_ssml_names_the_voice_and_locale():
    out = build_ssml("hello", VoiceId("azure", "ur-PK-AsadNeural"))
    assert "ur-PK-AsadNeural" in out and 'xml:lang="ur-PK"' in out


def test_ssml_style_is_optional():
    plain = build_ssml("x", VoiceId("azure", "v"))
    styled = build_ssml("x", VoiceId("azure", "v", style="calm"))
    assert "express-as" not in plain
    assert 'style="calm"' in styled


def test_null_tts_emits_chunks_then_a_final_marker():
    async def collect():
        return [
            c
            async for c in NullTextToSpeech(rate=24000).synthesize(
                "a reasonably long sentence to synthesise", VoiceId("null", "v")
            )
        ]

    chunks = asyncio.run(collect())
    assert len(chunks) > 1
    assert chunks[-1].final
    assert sum(c.pcm.size for c in chunks) > 0


def test_null_tts_says_nothing_for_empty_text():
    async def collect():
        return [
            c async for c in NullTextToSpeech().synthesize("  ", VoiceId("null", "v"))
        ]

    assert asyncio.run(collect()) == []


def test_pcm_chunk_reports_its_duration():
    chunk = PcmChunk(np.zeros(2400, dtype=DTYPE), 24000)
    assert chunk.duration_ms == pytest.approx(100.0)
