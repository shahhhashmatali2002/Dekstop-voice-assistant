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


def test_get_secret_loads_from_env_file(tmp_path, monkeypatch):
    from app.models.config import get_secret
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_SECRET_KEY=secret_val_123\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEST_SECRET_KEY", raising=False)
    assert get_secret("TEST_SECRET_KEY") == "secret_val_123"



# ----------------------------------------------------------------------
# dropped sentences
# ----------------------------------------------------------------------


def test_the_prompt_demands_every_sentence():
    """Observed verbatim: five sentences in, one sentence out.

        "Hey, welcome back. I'm David. And I'm Alice. Hello, everyone. So Alice,"
     -> "ہیلو، سب کو۔ تو ایلس،"

    Told it was an interpreter working "one fragment at a time", the model
    treated a multi-sentence unit as one fragment and rendered only the tail.
    A dropped sentence is invisible to the listener: they cannot know
    something was said.
    """
    from app.providers.llm_translator import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    assert "every sentence" in lowered
    assert "never summarise" in lowered or "never summarize" in lowered
    assert "only the last part" in lowered
    # And the framing that caused it is gone.
    assert "one fragment at a time" not in lowered


def test_names_are_not_dropped_either():
    from app.providers.llm_translator import SYSTEM_PROMPT

    assert "transliterate" in SYSTEM_PROMPT.lower()


def test_a_reply_that_lost_most_of_the_input_is_detected():
    from app.providers.llm_translator import looks_truncated

    english = ("Hey, welcome back. I'm David. And I'm Alice. Hello, everyone. "
               "So Alice,")
    assert looks_truncated(english, "ہیلو، سب کو۔ تو ایلس،")


def test_a_complete_reply_is_left_alone():
    from app.providers.llm_translator import looks_truncated

    english = ("Hey, welcome back. I'm David. And I'm Alice. Hello, everyone. "
               "So Alice,")
    full = ("ارے، واپس آنے پر خوش آمدید۔ میں ڈیوڈ ہوں۔ اور میں ایلس ہوں۔ "
            "ہیلو، سب کو۔ تو ایلس،")
    assert not looks_truncated(english, full)


def test_short_inputs_are_exempt():
    """"Yes." legitimately becomes one word; ratios on four words are noise."""
    from app.providers.llm_translator import looks_truncated

    assert not looks_truncated("Yes.", "جی ہاں۔")
    assert not looks_truncated("No pausing the video.", "ویڈیو نہ روکیں۔")


def test_an_empty_reply_is_not_treated_as_truncation():
    # Rule 2 lets the model return nothing rather than invent; retrying that
    # would be arguing with a deliberate refusal.
    from app.providers.llm_translator import looks_truncated

    assert not looks_truncated("Some long sentence that has many words in it", "")


def test_a_single_clause_is_never_retried():
    """The regression this prevents: every unit paying for a second round trip.

    A retry costs a full round trip, and a round trip is latency the listener
    hears. A single clause cannot have dropped a sentence, and terse is often
    correct - Urdu says "کیوں؟" for "So why is that, then?" - so the check
    only ever looks at multi-sentence input.
    """
    from app.providers.llm_translator import looks_truncated

    assert not looks_truncated("So why is that, then, in your view", "کیوں؟")
    assert not looks_truncated(
        "and how naturally you can use English in conversation",
        "اور آپ انگریزی کتنی قدرتی طور پر استعمال کر سکتے ہیں",
    )


def test_sentences_are_counted_in_both_scripts():
    from app.providers.llm_translator import sentence_count

    assert sentence_count("One. Two. Three.") == 3
    assert sentence_count("ایک۔ دو۔ تین۔") == 3
    assert sentence_count("What? Really!") == 2
    assert sentence_count("no punctuation at all") == 1
    assert sentence_count("") == 0


def test_losing_half_the_sentences_is_what_triggers_the_retry():
    from app.providers.llm_translator import looks_truncated

    four = "One thing. Two things. Three things. Four things happened today."
    assert looks_truncated(four, "چار چیزیں ہوئیں۔")
    assert not looks_truncated(
        four, "ایک چیز۔ دو چیزیں۔ تین چیزیں۔ آج چار چیزیں ہوئیں۔"
    )


# ----------------------------------------------------------------------
# what the voice is given to say
# ----------------------------------------------------------------------


def test_a_dangling_clause_is_given_an_ending():
    """A sentence with no final mark is read flat: the voice cannot know the
    thought ended, so it does not fall at the end. The translator is told not
    to invent punctuation the speaker did not imply - right for the text,
    wrong for the speech."""
    from app.providers.tts import speakable

    assert speakable("آپ کی انگریزی کا معیار").endswith("۔")


def test_punctuation_that_is_already_there_is_left_alone():
    from app.providers.tts import speakable

    for ending in ("۔", "؟", "!", ".", ","):
        text = "کچھ بات" + ending
        assert speakable(text) == text


def test_stray_marks_that_become_pauses_are_removed():
    from app.providers.tts import speakable

    assert speakable('"ہیلو"') == "ہیلو۔"
    assert speakable("(ہیلو)") == "ہیلو۔"


def test_runs_of_whitespace_are_collapsed():
    from app.providers.tts import speakable

    assert speakable("  ہیلو\n\n  دنیا  ") == "ہیلو دنیا۔"


def test_nothing_in_nothing_out():
    from app.providers.tts import speakable

    assert speakable("") == ""
    assert speakable("   ") == ""


def test_the_voice_is_asked_to_slow_down_a_little():
    from app.providers.tts import EdgeTextToSpeech

    assert EdgeTextToSpeech().speaking_rate.startswith("-")
    assert EdgeTextToSpeech(speaking_rate="+0%").speaking_rate == "+0%"


# ----------------------------------------------------------------------
# a realtime socket that fails must not look like one that is quiet
# ----------------------------------------------------------------------


def test_a_server_error_event_reaches_the_caller():
    """The bug that killed a live call. The read loop raised
    ProviderUnavailable for an error event and then discarded it, so a rejected
    session looked exactly like a working one with nothing to say: audio went
    in, nothing came out, and not one line was printed anywhere."""
    import asyncio
    import json

    from app.providers.base import ProviderUnavailable
    from app.providers.openai_s2st import OpenAiS2stStream

    class FakeSocket:
        """Sends one error event, as the API does for a bad session."""

        def __aiter__(self):
            async def gen():
                yield json.dumps({
                    "type": "error",
                    "error": {"code": "invalid_request_error",
                              "message": "unknown model 'gpt-realtime-translate'"},
                })
            return gen()

        async def send(self, _payload):
            return None

        async def close(self):
            return None

    async def run():
        stream = OpenAiS2stStream(FakeSocket(), 24_000)
        with pytest.raises(ProviderUnavailable) as caught:
            async for _chunk in stream.audio_out():
                pass
        await stream.aclose()
        return str(caught.value)

    detail = asyncio.run(run())
    assert "unknown model" in detail, detail


def test_a_rate_limit_is_reported_as_one():
    import asyncio
    import json

    from app.providers.base import ProviderRateLimited
    from app.providers.openai_s2st import OpenAiS2stStream

    class FakeSocket:
        def __aiter__(self):
            async def gen():
                yield json.dumps({
                    "type": "error",
                    "error": {"code": "rate_limit_exceeded", "message": "slow down"},
                })
            return gen()

        async def send(self, _payload):
            return None

        async def close(self):
            return None

    async def run():
        stream = OpenAiS2stStream(FakeSocket(), 24_000)
        with pytest.raises(ProviderRateLimited):
            async for _chunk in stream.audio_out():
                pass
        await stream.aclose()

    asyncio.run(run())


def test_a_clean_end_of_stream_raises_nothing():
    """Only a failure is re-raised. A socket that simply closes is the normal
    end of a session."""
    import asyncio
    import json

    from app.providers.openai_s2st import OpenAiS2stStream

    class FakeSocket:
        def __aiter__(self):
            async def gen():
                yield json.dumps({"type": "session.updated"})
            return gen()

        async def send(self, _payload):
            return None

        async def close(self):
            return None

    async def run():
        stream = OpenAiS2stStream(FakeSocket(), 24_000)
        chunks = [c async for c in stream.audio_out()]
        await stream.aclose()
        return chunks

    assert asyncio.run(run()) == []


# ----------------------------------------------------------------------
# the translation protocol, pinned end to end
# ----------------------------------------------------------------------


def _fake_socket(events):
    """A socket that replays server events and records what we send."""
    import json

    class Socket:
        def __init__(self):
            self.sent = []

        def __aiter__(self):
            async def gen():
                for event in events:
                    yield json.dumps(event)
            return gen()

        async def send(self, payload):
            self.sent.append(json.loads(payload))

        async def close(self):
            return None

    return Socket()


def test_audio_is_appended_with_the_namespaced_event_name():
    """The whole protocol on this endpoint is namespaced under "session.", and
    the voice-agent name fails as:

        Invalid value: 'inp...end'. Supported values are: 'session.update',
        'session.input_audio_buffer.append', and 'session.close'

    Every one of these names fails as SILENCE rather than as an error, which is
    why they are pinned by tests rather than trusted to a reading."""
    import asyncio

    import numpy as np

    from app.providers.openai_s2st import APPEND_AUDIO, OpenAiS2stStream

    async def run():
        socket = _fake_socket([])
        stream = OpenAiS2stStream(socket, 24_000)
        await stream.push_audio(np.zeros(240, dtype=np.int16))
        await stream.aclose()
        return socket.sent

    sent = asyncio.run(run())
    assert sent[0]["type"] == APPEND_AUDIO == "session.input_audio_buffer.append"
    assert "audio" in sent[0]


def test_the_session_closes_politely():
    import asyncio

    from app.providers.openai_s2st import CLOSE_SESSION, OpenAiS2stStream

    async def run():
        socket = _fake_socket([])
        stream = OpenAiS2stStream(socket, 24_000)
        await stream.aclose()
        return socket.sent

    assert asyncio.run(run())[-1]["type"] == CLOSE_SESSION


def test_translated_audio_arrives_on_the_documented_event():
    import asyncio
    import base64

    import numpy as np

    from app.providers.openai_s2st import OUTPUT_AUDIO_DELTA, OpenAiS2stStream

    payload = base64.b64encode(
        np.full(480, 1234, dtype=np.int16).tobytes()
    ).decode()

    async def run():
        stream = OpenAiS2stStream(
            _fake_socket([{"type": OUTPUT_AUDIO_DELTA, "delta": payload}]), 24_000
        )
        chunks = [c async for c in stream.audio_out()]
        await stream.aclose()
        return chunks

    chunks = asyncio.run(run())
    assert chunks and chunks[0].pcm.size == 480
    assert int(chunks[0].pcm[0]) == 1234


def test_the_source_transcript_is_never_shown_as_the_translation():
    """input_transcript is the Urdu that was heard; output_transcript is the
    English that was produced. Treating both as the translation puts the source
    language in the transcript window as though it were the result."""
    import asyncio

    from app.providers.openai_s2st import (
        INPUT_TRANSCRIPT_DELTA,
        OUTPUT_TRANSCRIPT_DELTA,
        OpenAiS2stStream,
    )

    async def run():
        stream = OpenAiS2stStream(
            _fake_socket([
                {"type": INPUT_TRANSCRIPT_DELTA, "delta": "میں ٹھیک ہوں"},
                {"type": OUTPUT_TRANSCRIPT_DELTA, "delta": "I am fine"},
            ]),
            24_000,
        )
        got = [t.text async for t in stream.transcripts()]
        await stream.aclose()
        return got

    assert asyncio.run(run()) == ["I am fine"]


def test_the_session_matches_the_documented_shape():
    from app.providers.base import S2stConfig
    from app.providers.openai_s2st import OpenAiSpeechTranslator

    cfg = S2stConfig(source_language="ur", target_language="en", rate=24_000)
    session = OpenAiSpeechTranslator("sk-test").session_update(cfg)["session"]

    assert session["audio"]["output"]["language"] == "en"
    assert session["audio"]["input"]["transcription"]["model"]
    assert session["audio"]["input"]["noise_reduction"]["type"] == "near_field"
    # No voice-agent fields: each one comes back as "Unknown parameter" and
    # takes the whole session with it.
    assert "type" not in session
    assert "instructions" not in session
    assert "turn_detection" not in session
    assert "modalities" not in session
