"""Tests for the pure-logic pipeline stages.

Segmentation, stability, drift and the mode machine are where the product's
behaviour actually lives, and all of it runs with no audio device, no network
and no GUI. That is the §10.4 dependency rule paying for itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.controllers.drift import DriftAction, DriftController, time_stretch
from app.controllers.segmenter import Hypothesis, Segmenter, StabilityFilter
from app.models.state import Colour, Health, Mode, ModeMachine
from app.models.telemetry import (
    Direction,
    MetricsBus,
    Percentiles,
    TraceFactory,
    UnitTrace,
)


# ======================================================================
# stability filter
# ======================================================================


def test_nothing_is_stable_until_the_window_is_full():
    s = StabilityFilter(min_hypotheses=3, stable_ms=0)
    assert s.update("I think", 0.0) == ""
    assert s.update("I think we", 0.1) == ""


def test_growing_text_only_ever_confirms_the_common_prefix():
    """While the speaker is still producing words, only the shared head is safe."""
    s = StabilityFilter(min_hypotheses=3, stable_ms=0)
    s.update("I think", 0.0)
    s.update("I think we", 0.1)
    assert s.update("I think we should", 0.2) == "I think"
    assert s.update("I think we should change", 0.3) == "I think we"


def test_the_prefix_can_grow_again_once_the_speaker_pauses():
    """The bug that hid for a while: a common prefix must not only shrink."""
    s = StabilityFilter(min_hypotheses=3, stable_ms=0)
    for i, text in enumerate(["a", "a b", "a b c"]):
        s.update(text, i * 0.1)
    assert s.update("a b c", 0.3) == "a b"
    # The recogniser now repeats itself, as it does while waiting to endpoint.
    s.update("a b c", 0.4)
    assert s.update("a b c", 0.5) == "a b c", (
        "once the text settles, the whole utterance must become confirmable"
    )


def test_prefix_needs_time_as_well_as_repetition():
    s = StabilityFilter(min_hypotheses=2, stable_ms=350)
    s.update("hello world", 0.0)
    assert s.update("hello world", 0.1) == ""  # repeated, but too soon
    assert s.update("hello world", 0.5) == "hello world"


def test_revision_resets_the_stability_timer():
    s = StabilityFilter(min_hypotheses=2, stable_ms=350)
    s.update("I think we should change", 0.0)
    s.update("I think we should change", 0.5)
    assert s.update("I think we should charge", 0.6) == "", (
        "a revision must restart the clock, not confirm immediately"
    )


def test_a_revision_only_retracts_back_to_the_disputed_word():
    s = StabilityFilter(min_hypotheses=2, stable_ms=0)
    s.update("I think we should change", 0.0)
    out = s.update("I think we should charge", 0.1)
    assert out == "I think we should", "the undisputed head stays confirmed"
    assert "cha" not in out.split()[-1], "must never confirm a half word"


def test_reset_clears_state():
    s = StabilityFilter(min_hypotheses=1, stable_ms=0)
    s.update("abc", 0.0)
    s.reset()
    assert s._prefix == ""


# ======================================================================
# segmenter
# ======================================================================


def feed(seg, texts, *, final_last=False, t0=0.0, dt=0.4):
    out = []
    for i, txt in enumerate(texts):
        is_final = final_last and i == len(texts) - 1
        out += seg.push(Hypothesis(txt, is_final=is_final, at=t0 + i * dt))
    return out


def test_final_hypothesis_emits_everything():
    seg = Segmenter()
    units = seg.push(Hypothesis("Can you finish it by Friday?", is_final=True))
    assert len(units) == 1
    assert units[0].text == "Can you finish it by Friday?"
    assert units[0].reason == "final"


def test_nothing_is_emitted_twice():
    seg = Segmenter(min_hypotheses=1, stable_ms=0)
    all_units = feed(
        seg,
        [
            "We should deploy.",
            "We should deploy. Then we",
            "We should deploy. Then we test.",
        ],
    )
    joined = " ".join(u.text for u in all_units)
    assert joined.count("We should deploy.") == 1


def test_clause_boundary_triggers_emission():
    seg = Segmenter(min_hypotheses=1, stable_ms=0, min_words=2)
    units = feed(seg, ["The build is green.", "The build is green."])
    assert any(u.reason == "stable" for u in units)
    assert units[0].text.endswith(".")


def test_urdu_danda_is_a_clause_boundary():
    """Urdu ends sentences with the danda, not a full stop."""
    seg = Segmenter(min_hypotheses=1, stable_ms=0, min_words=2)
    units = feed(seg, ["جی میں کر دوں گا۔", "جی میں کر دوں گا۔"])
    assert units and units[0].text.endswith("۔")


def test_long_run_without_punctuation_is_cut_at_max_words():
    seg = Segmenter(min_hypotheses=1, stable_ms=0, max_words=6, min_words=2)
    long = "one two three four five six seven eight nine ten"
    units = feed(seg, [long, long])
    assert units, "a long unpunctuated run must still be emitted"
    assert units[0].reason == "max_words"


def test_max_words_cut_never_splits_a_word():
    seg = Segmenter(min_hypotheses=1, stable_ms=0, max_words=4, min_words=1)
    text = "alpha bravo charlie delta echo foxtrot"
    units = feed(seg, [text, text])
    assert units
    for word in units[0].text.split():
        assert word in text.split()


def test_flush_emits_what_is_held():
    seg = Segmenter(min_hypotheses=1, stable_ms=0)
    seg.push(Hypothesis("half a sentence", at=0.0))
    seg.push(Hypothesis("half a sentence", at=0.1))
    units = seg.flush()
    assert units and units[0].reason == "vad"
    assert "half a sentence" in units[0].text


def test_flush_on_empty_is_harmless():
    assert Segmenter().flush() == []


def test_revision_of_spoken_text_is_not_re_emitted():
    """Once the Urdu audio has played, we cannot take the words back."""
    seg = Segmenter(min_hypotheses=1, stable_ms=0, min_words=2)
    feed(seg, ["We should charge.", "We should charge."])
    before = seg.emitted
    units = feed(seg, ["We should change.", "We should change."])
    # It must not re-emit the whole corrected sentence as if it were new.
    assert before
    assert all("We should charge." not in u.text for u in units)


def test_code_switched_sentence_survives_segmentation():
    seg = Segmenter(min_hypotheses=1, stable_ms=0, min_words=2)
    text = "میں نے staging پر deploy کر دیا ہے۔"
    units = feed(seg, [text, text])
    assert units
    assert "staging" in units[0].text and "deploy" in units[0].text


def test_soft_boundary_emits_a_long_enough_clause():
    """Fluent speech is commas for minutes at a time - a clause that ends at
    a comma and is long enough to stand alone must not wait for a full stop
    or the max_words axe."""
    seg = Segmenter(min_hypotheses=1, stable_ms=0, soft_min_words=4)
    text = "so today we are looking at the pipeline,"
    units = feed(seg, [text, text])
    assert units, "a stable clause ending at a comma must be emitted"
    assert units[0].reason == "soft"
    assert units[0].text.endswith(",")


def test_soft_boundary_holds_a_short_clause():
    """"yes," is worth holding for its sentence - the soft rule needs more
    words than the hard rule before it fires."""
    seg = Segmenter(min_hypotheses=1, stable_ms=0, soft_min_words=6)
    units = feed(seg, ["well yes,", "well yes,"])
    assert units == []


def test_soft_emission_is_not_repeated_by_the_final():
    """What the comma rule emitted must not be spoken again when the
    recogniser finalises the whole sentence."""
    seg = Segmenter(min_hypotheses=1, stable_ms=0, soft_min_words=4)
    clause = "so today we are looking at the pipeline,"
    feed(seg, [clause, clause])
    units = seg.push(Hypothesis(
        "so today we are looking at the pipeline, and then the tests.",
        is_final=True,
    ))
    assert len(units) == 1
    assert units[0].text.strip() == "and then the tests."


def test_unit_word_count():
    seg = Segmenter()
    u = seg.push(Hypothesis("one two three", is_final=True))[0]
    assert u.words == 3


# ======================================================================
# drift controller
# ======================================================================


def test_shallow_queue_plays_at_normal_rate():
    d = DriftController().decide(100)
    assert d.action is DriftAction.NORMAL
    assert d.rate == 1.0
    assert not d.drop


def test_moderate_queue_speeds_up_gently():
    d = DriftController().decide(900)
    assert d.action is DriftAction.SPEED_UP
    assert 1.0 < d.rate <= 1.15
    assert not d.prefer_brevity


def test_deep_queue_speeds_up_and_asks_for_brevity():
    d = DriftController().decide(2200)
    assert d.action is DriftAction.SPEED_UP_AND_SHORTEN
    assert d.prefer_brevity
    assert d.rate <= 1.25


def test_runaway_queue_drops():
    c = DriftController()
    d = c.decide(5000)
    assert d.action is DriftAction.DROP_OLDEST
    assert d.drop
    assert c.drops == 1


def test_rate_increases_monotonically_with_queue_depth():
    c = DriftController()
    rates = [c.decide(ms).rate for ms in (0, 300, 600, 1000, 1400, 2000, 2900)]
    assert rates == sorted(rates)


def test_rate_never_exceeds_the_cap():
    c = DriftController(max_rate=1.2)
    assert all(c.decide(ms).rate <= 1.2 for ms in range(0, 6000, 100))


def test_thresholds_must_be_increasing():
    with pytest.raises(ValueError):
        DriftController(comfortable_ms=2000, stretch_ms=1000, aggressive_ms=3000)


# ======================================================================
# time stretch - must preserve pitch
# ======================================================================


def _dominant_hz(x: np.ndarray, rate: int) -> float:
    w = x.astype(np.float64) * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(w))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[int(np.argmax(spec))])


def test_time_stretch_of_rate_one_is_identity():
    x = np.arange(1000, dtype=np.int16)
    assert time_stretch(x, 1.0, 16000) is x


def test_time_stretch_shortens_the_audio():
    rate = 16000
    t = np.arange(rate, dtype=np.float32) / rate
    tone = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    out = time_stretch(tone, 1.2, rate)
    assert len(out) < len(tone)
    assert len(out) == pytest.approx(len(tone) / 1.2, rel=0.15)


def test_time_stretch_preserves_pitch():
    """The whole point: speeding up must NOT make the voice a chipmunk."""
    rate, freq = 16000, 220.0
    t = np.arange(rate, dtype=np.float32) / rate
    tone = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)

    out = time_stretch(tone, 1.2, rate)

    assert abs(_dominant_hz(out, rate) - freq) < 15.0, (
        "pitch shifted - this is resampling, not time-stretching"
    )


def test_time_stretch_handles_tiny_input():
    tiny = np.ones(16, dtype=np.int16)
    assert time_stretch(tiny, 1.2, 16000).size > 0


# ======================================================================
# mode machine
# ======================================================================


def test_starts_in_passthrough_not_off():
    m = ModeMachine()
    assert m.mode is Mode.PASSTHROUGH
    assert m.colour() is Colour.GREY


def test_start_translates_both_directions():
    m = ModeMachine()
    m.start()
    assert m.mode is Mode.TRANSLATING
    assert m.colour() is Colour.GREEN
    assert m.should_translate(Direction.INCOMING)
    assert m.should_translate(Direction.OUTGOING)


def test_pause_reverts_to_raw_audio():
    m = ModeMachine()
    m.start()
    m.pause()
    assert m.mode is Mode.PAUSED
    assert m.colour() is Colour.AMBER
    assert not m.should_translate(Direction.INCOMING)
    assert not m.wire_muted(Direction.OUTGOING)


def test_stop_ends_translation_not_audio():
    m = ModeMachine()
    m.start()
    m.stop()
    assert m.mode is Mode.PASSTHROUGH
    assert m.snapshot().audio_flowing


def test_degrading_one_direction_leaves_the_other_working():
    m = ModeMachine()
    m.start()
    m.degrade(Direction.INCOMING, "ASR connection lost")
    assert m.mode is Mode.DEGRADED
    assert not m.should_translate(Direction.INCOMING)
    assert m.should_translate(Direction.OUTGOING), (
        "the consultant must still be understood"
    )


def test_degraded_direction_falls_back_to_passthrough_never_silence():
    m = ModeMachine()
    m.start()
    m.degrade(Direction.OUTGOING, "provider timeout")
    assert not m.wire_muted(Direction.OUTGOING), (
        "an unmuted wire is what keeps the client hearing something"
    )


def test_recovery_returns_to_translating():
    m = ModeMachine()
    m.start()
    m.degrade(Direction.INCOMING, "boom")
    m.recover(Direction.INCOMING)
    assert m.mode is Mode.TRANSLATING
    assert m.colour() is Colour.GREEN


def test_impaired_direction_is_amber_not_red():
    m = ModeMachine()
    m.start()
    m.impair(Direction.INCOMING, "TTS down, text only")
    assert m.colour() is Colour.AMBER
    assert m.state(Direction.INCOMING).health is Health.DEGRADED


def test_audio_flows_in_every_reachable_mode():
    """The §2.5 invariant, asserted across the whole state space."""
    m = ModeMachine()
    for action in (
        lambda: m.start(),
        lambda: m.pause(),
        lambda: m.resume(),
        lambda: m.degrade(Direction.INCOMING, "x"),
        lambda: m.degrade(Direction.OUTGOING, "y"),
        lambda: m.recover(Direction.INCOMING),
        lambda: m.stop(),
    ):
        snap = action()
        assert snap.audio_flowing


def test_observers_are_notified():
    m = ModeMachine()
    seen = []
    m.subscribe(seen.append)
    m.start()
    m.pause()
    assert [s.mode for s in seen] == [Mode.TRANSLATING, Mode.PAUSED]


def test_a_broken_observer_cannot_take_down_the_engine():
    m = ModeMachine()
    m.subscribe(lambda snap: 1 / 0)
    ok = []
    m.subscribe(ok.append)
    m.start()  # must not raise
    assert ok


# ======================================================================
# telemetry
# ======================================================================


def test_ttfta_is_playback_minus_onset():
    t = UnitTrace(1, Direction.INCOMING)
    t.t_speech_onset = 10.0
    t.t_playback_start = 11.25
    assert t.ttfta == pytest.approx(1.25)


def test_ttfta_is_none_until_playback_starts():
    t = UnitTrace(1, Direction.INCOMING)
    t.t_speech_onset = 10.0
    assert t.ttfta is None, "an unfinished unit has no TTFTA, not a TTFTA of 0"


def test_stage_durations_skip_stages_that_never_ran():
    t = UnitTrace(1, Direction.INCOMING)
    t.t_speech_onset, t.t_frame_sent = 1.0, 1.05
    stages = t.stages()
    assert stages["capture"] == pytest.approx(0.05)
    assert "translation" not in stages


def test_percentiles_of_a_known_set():
    p = Percentiles(window_s=1000)
    for v in range(1, 101):
        p.add(v / 100.0, at=0.0)
    s = p.snapshot(at=0.0)
    assert s["n"] == 100
    assert s["p50"] == pytest.approx(0.50, abs=0.02)
    assert s["p90"] == pytest.approx(0.90, abs=0.02)
    assert s["max"] == pytest.approx(1.0)


def test_percentiles_evict_outside_the_window():
    p = Percentiles(window_s=10)
    p.add(1.0, at=0.0)
    p.add(2.0, at=100.0)
    assert p.snapshot(at=100.0)["n"] == 1


def test_empty_percentiles_report_none_not_zero():
    assert Percentiles().snapshot() == {"n": 0, "p50": None, "p90": None, "max": None}


def test_metrics_bus_separates_directions():
    bus = MetricsBus()
    for d, val in ((Direction.INCOMING, 1.2), (Direction.OUTGOING, 0.8)):
        t = UnitTrace(1, d)
        t.t_speech_onset, t.t_playback_start = 100.0, 100.0 + val
        bus.record(t)
    snap = bus.snapshot()
    assert snap["incoming"]["ttfta"]["p50"] == pytest.approx(1.2)
    assert snap["outgoing"]["ttfta"]["p50"] == pytest.approx(0.8)


def test_dropped_units_are_counted_but_not_timed():
    bus = MetricsBus()
    t = UnitTrace(1, Direction.INCOMING)
    t.dropped = True
    bus.record(t)
    snap = bus.snapshot()["incoming"]
    assert snap["units"] == 1 and snap["dropped"] == 1
    assert snap["ttfta"]["n"] == 0


def test_trace_factory_ids_increase_per_direction():
    f = TraceFactory()
    assert f.new(Direction.INCOMING).unit_id == 1
    assert f.new(Direction.INCOMING).unit_id == 2
    assert f.new(Direction.OUTGOING).unit_id == 1


def test_trace_factory_stamps_onset():
    assert TraceFactory().new(Direction.INCOMING).t_speech_onset > 0


# ======================================================================
# the filter that went silent
# ======================================================================


def test_a_growing_prefix_is_confirmed_while_the_speaker_keeps_talking():
    """The bug: one line translated in 37 seconds of a podcast.

    "The prefix changed, so restart the clock" treats

        "Hey welcome back"  ->  "Hey welcome back so today"

    as a disagreement. It is not: every word in the first is still agreed,
    and has been agreed for LONGER. But hypotheses arrive every few hundred
    milliseconds and each extends the prefix, so the clock restarted on every
    one and nothing was ever confirmed while somebody kept talking.
    """
    s = StabilityFilter(min_hypotheses=3, stable_ms=300)
    words = "so today we are going to talk about listening".split()

    confirmed = ""
    for index in range(1, len(words) + 1):
        confirmed = s.update(" ".join(words[:index]), index * 0.35)

    assert confirmed, "the filter confirmed nothing across nine hypotheses"
    assert confirmed.split()[0] == "so"


def test_continuous_speech_produces_units_without_waiting_for_a_pause():
    """End to end: a monologue with no full stops and no gaps.

    Sixty-one words of it used to produce zero units - the only line that
    ever came out was the one the VAD flushed at the first pause.
    """
    words = ("Hey welcome back so today we are going to talk about something "
             "that a lot of you have been asking me about and it is one of "
             "those topics that comes up again and again whenever people talk "
             "about learning a language so what is it well it is listening").split()

    seg = Segmenter(min_hypotheses=3, stable_ms=300.0, max_words=12)
    spoken = []
    for index in range(1, len(words) + 1):
        for unit in seg.push(
            Hypothesis(" ".join(words[:index]), False, index * 0.35)
        ):
            spoken.append(unit)

    assert len(spoken) >= 4, (
        f"{len(words)} words of continuous speech produced only "
        f"{len(spoken)} units"
    )
    # And nothing is said twice.
    joined = " ".join(u.text for u in spoken).split()
    assert joined == words[:len(joined)], "a phrase was repeated or reordered"


def test_the_first_unit_arrives_within_a_few_seconds():
    words = ("Hey welcome back so today we are going to talk about listening "
             "and how it changes everything about learning a language").split()
    seg = Segmenter(min_hypotheses=3, stable_ms=300.0, max_words=12)

    first_at = None
    for index in range(1, len(words) + 1):
        at = index * 0.35
        if seg.push(Hypothesis(" ".join(words[:index]), False, at)):
            first_at = at
            break

    assert first_at is not None, "no unit was ever emitted"
    assert first_at <= 10.0, f"the first unit took {first_at:.1f}s"


def test_a_revision_still_holds_everything_back_until_it_settles():
    # The behaviour the extension fix must not break.
    s = StabilityFilter(min_hypotheses=2, stable_ms=350)
    s.update("I think we should change", 0.0)
    s.update("I think we should change", 0.5)
    assert s.update("I think we should charge", 0.6) == ""
