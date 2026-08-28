"""Pre-translated playback: the transcript, the merge, and the clock.

No network and no devices. The timing tests drive an injected clock, because
timing that can only be observed in real time cannot be tested at all.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from app.audio.ring import RingBuffer
from app.controllers import pretranslate as P
from app.providers.base import (
    PcmChunk,
    ProviderError,
    TextToSpeech,
    TranslatedUnit,
    Translator,
    VoiceId,
)


# ----------------------------------------------------------------------
# finding the video
# ----------------------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=1bqQHEbVIFc",
    "https://www.youtube.com/watch?v=1bqQHEbVIFc&t=42s",
    "https://youtu.be/1bqQHEbVIFc",
    "https://www.youtube.com/shorts/1bqQHEbVIFc",
    "https://www.youtube.com/embed/1bqQHEbVIFc",
    "1bqQHEbVIFc",
])
def test_every_shape_youtube_hands_people(url):
    assert P.video_id(url) == "1bqQHEbVIFc"


def test_something_that_is_not_a_video_says_so():
    with pytest.raises(P.PrepareError):
        P.video_id("https://example.com/watch?v=nope")
    with pytest.raises(P.PrepareError):
        P.video_id("")


# ----------------------------------------------------------------------
# captions are cut to fit a screen, not at meaning
# ----------------------------------------------------------------------


def cues(*rows):
    return [P.Cue(start, end, text) for start, end, text in rows]


def test_fragments_are_joined_into_sentences():
    """The exact shape a caption track has, taken from a real video."""
    merged = P.merge_sentences(cues(
        (0.0, 1.4, "skill. For example, your"),
        (1.4, 3.0, "grammar, your vocabulary,"),
        (3.0, 5.2, "your listening."),
        (5.2, 7.0, "For every correct answer,"),
        (7.0, 8.8, "you get one point."),
    ))
    assert [c.text for c in merged] == [
        "skill. For example, your grammar, your vocabulary, your listening.",
        "For every correct answer, you get one point.",
    ]


def test_a_merged_sentence_keeps_the_moment_it_started():
    # The whole mode depends on this: the timestamp is when the speaker BEGAN
    # saying it, not when the caption fragment happened to end.
    merged = P.merge_sentences(cues(
        (10.0, 11.0, "And at the end of the video,"),
        (11.0, 13.5, "I am going to reveal your level."),
    ))
    assert merged[0].start == 10.0
    assert merged[0].end == 13.5


def test_a_speaker_who_never_stops_is_still_cut():
    long_run = cues(*[(i * 2.0, i * 2.0 + 2.0, "and then") for i in range(20)])
    merged = P.merge_sentences(long_run, max_seconds=6.0)
    assert len(merged) > 1
    assert all((c.end - c.start) <= 8.0 for c in merged)


def test_the_tail_is_not_lost_when_a_video_ends_mid_sentence():
    merged = P.merge_sentences(cues((0.0, 2.0, "no full stop here")))
    assert [c.text for c in merged] == ["no full stop here"]


# ----------------------------------------------------------------------
# preparing
# ----------------------------------------------------------------------


class FakeTranslator(Translator):
    name = "fake"

    def __init__(self, fail_on: int | None = None) -> None:
        self.seen: list[str] = []
        self.context_sizes: list[int] = []
        self.fail_on = fail_on

    async def translate(self, unit, ctx, terms=None) -> TranslatedUnit:
        self.seen.append(unit.text)
        self.context_sizes.append(len(ctx))
        if self.fail_on is not None and len(self.seen) == self.fail_on:
            raise ProviderError("translator fell over")
        return TranslatedUnit(text="[ur] " + unit.text, source=unit)


class FakeTts(TextToSpeech):
    name = "fake"

    def __init__(self, rate: int = 24_000, seconds: float = 1.0) -> None:
        self.rate = rate
        self.seconds = seconds
        self.spoken: list[str] = []

    async def synthesize(self, text: str, voice):
        self.spoken.append(text)
        n = int(self.rate * self.seconds)
        yield PcmChunk(np.zeros(n, dtype=np.int16), self.rate)


VOICE = VoiceId(vendor="fake", name="ur-PK-UzmaNeural", rate=24_000)


def prepared(script_cues, **kw):
    return asyncio.run(P.prepare(
        script_cues,
        translator=kw.pop("translator", FakeTranslator()),
        tts=kw.pop("tts", FakeTts()),
        voice=VOICE,
        **kw,
    ))


def test_every_line_is_translated_and_synthesised_before_anything_plays():
    lines = prepared(cues((0.0, 2.0, "One."), (5.0, 7.0, "Two.")))
    assert [line.start for line in lines] == [0.0, 5.0]
    assert all(line.ready for line in lines)
    assert [line.urdu for line in lines] == ["[ur] One.", "[ur] Two."]


def test_each_sentence_is_translated_knowing_what_came_before():
    """The advantage this mode has: it is not translating a fragment blind."""
    translator = FakeTranslator()
    prepared(cues((0.0, 1.0, "First."), (2.0, 3.0, "Second."),
                  (4.0, 5.0, "Third.")), translator=translator)
    assert translator.context_sizes == [0, 1, 2]


def test_progress_is_reported_for_every_line():
    seen = []
    prepared(cues((0.0, 1.0, "One."), (2.0, 3.0, "Two.")),
             on_progress=lambda i, total, text: seen.append((i, total)))
    assert seen == [(1, 2), (2, 2)]


def test_a_provider_failure_names_the_line():
    with pytest.raises(P.PrepareError) as caught:
        prepared(cues((0.0, 1.0, "One."), (2.0, 3.0, "Two.")),
                 translator=FakeTranslator(fail_on=2))
    assert "line 2" in str(caught.value)


def test_urdu_that_outruns_its_slot_is_reported_not_truncated():
    """Urdu is often wordier. Overlapping slightly beats being cut off."""
    lines = prepared(cues((0.0, 1.0, "One."), (1.5, 2.0, "Two.")),
                     tts=FakeTts(seconds=3.0))
    late = P.overruns(lines)
    assert late, "a three-second line in a 1.5 second gap was not flagged"
    assert late[0][1] == pytest.approx(1.5, abs=0.05)


# ----------------------------------------------------------------------
# the clock
# ----------------------------------------------------------------------


class FakeSink:
    def __init__(self, rate: int = 24_000, channels: int = 1) -> None:
        self.rate = rate
        self.channels = channels
        self.ring = RingBuffer(rate * channels)
        self.written: list[np.ndarray] = []

    def write(self, pcm):
        self.written.append(pcm)
        self.ring.discard()  # a real sink is being drained by the hardware
        return 0


class FakeClock:
    """Time only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += max(0.0, seconds)


def line(start: float, seconds: float = 0.5, text: str = "x") -> P.Line:
    return P.Line(
        start=start,
        english=text,
        urdu="[ur] " + text,
        pcm=np.zeros(int(24_000 * seconds), dtype=np.int16),
        rate=24_000,
    )


def test_each_line_waits_for_its_own_timestamp():
    clock = FakeClock()
    player = P.Player(FakeSink(), [line(0.0), line(5.0), line(12.0)],
                      clock=clock, sleep=clock.sleep)
    asyncio.run(player.run())

    assert player.spoken == 3
    # 0 s, then 5 s later, then 7 s after that: the video's own spacing.
    assert clock.slept == pytest.approx([5.0, 7.0], abs=0.01)


def test_starting_mid_video_skips_what_is_already_past():
    clock = FakeClock()
    player = P.Player(FakeSink(), [line(0.0), line(10.0), line(20.0)],
                      clock=clock, sleep=clock.sleep)
    asyncio.run(player.run(position=15.0))

    # The first two belong to picture that has already gone by. Speaking them
    # against the wrong scene is worse than not speaking them.
    assert player.skipped == 2
    assert player.spoken == 1


def test_the_offset_shifts_every_line():
    clock = FakeClock()
    player = P.Player(FakeSink(), [line(10.0)], offset=-2.0,
                      clock=clock, sleep=clock.sleep)
    asyncio.run(player.run())
    assert clock.slept == pytest.approx([8.0], abs=0.01)


def test_lines_are_played_in_time_order_whatever_order_they_arrive():
    clock = FakeClock()
    heard = []
    player = P.Player(FakeSink(), [line(9.0, text="c"), line(1.0, text="a"),
                                   line(4.0, text="b")],
                      clock=clock, sleep=clock.sleep)
    asyncio.run(player.run(on_line=lambda ln: heard.append(ln.english)))
    assert heard == ["a", "b", "c"]


def test_the_audio_actually_reaches_the_sink():
    clock = FakeClock()
    sink = FakeSink()
    player = P.Player(sink, [line(0.0, seconds=0.5)], clock=clock,
                      sleep=clock.sleep)
    asyncio.run(player.run())
    assert sum(chunk.size for chunk in sink.written) == 12_000


def test_output_is_converted_to_the_device_format():
    clock = FakeClock()
    sink = FakeSink(rate=48_000, channels=2)
    player = P.Player(sink, [line(0.0, seconds=0.5)], clock=clock,
                      sleep=clock.sleep)
    asyncio.run(player.run())
    total = sum(chunk.size for chunk in sink.written)
    # 0.5 s at 48 kHz stereo.
    assert total == pytest.approx(48_000, abs=200)
    assert all(chunk.dtype == np.int16 for chunk in sink.written)
