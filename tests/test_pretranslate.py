"""Pre-translated playback: the transcript, the merge, and the clock.

No network and no devices. The timing tests drive an injected clock, because
timing that can only be observed in real time cannot be tested at all.
"""

from __future__ import annotations

import asyncio
import contextlib

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
        # Yield, or a coroutine that only ever calls this never lets any other
        # task run - and a test with a player task and a driver task deadlocks.
        await asyncio.sleep(0)


def line(start: float, seconds: float = 0.5, text: str = "x") -> P.Line:
    return P.Line(
        start=start,
        english=text,
        urdu="[ur] " + text,
        pcm=np.zeros(int(24_000 * seconds), dtype=np.int16),
        rate=24_000,
    )


def spoken_at(player, clock, **kw):
    """Positions in the video at which each line was actually spoken."""
    at: list[float] = []
    asyncio.run(player.run(on_line=lambda _ln: at.append(clock.now), **kw))
    return at


def test_each_line_is_spoken_at_its_own_timestamp():
    clock = FakeClock()
    player = P.Player(FakeSink(), [line(0.0), line(5.0), line(12.0)],
                      clock=clock, sleep=clock.sleep)
    at = spoken_at(player, clock)

    assert player.spoken == 3
    # The video's own spacing, not the wall clock's.
    assert at == pytest.approx([0.0, 5.0, 12.0], abs=0.15)


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
    assert spoken_at(player, clock) == pytest.approx([8.0], abs=0.15)


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


# ----------------------------------------------------------------------
# transcribing the audio instead of trusting YouTube's captions
# ----------------------------------------------------------------------


def test_deepgram_utterances_become_timed_sentences():
    """The shape this mode needs, straight from the pre-recorded endpoint."""
    payload = {
        "results": {
            "utterances": [
                {"start": 0.32, "end": 4.1,
                 "transcript": "What's your level of English?"},
                {"start": 4.5, "end": 9.8,
                 "transcript": "Are you a beginner, an intermediate, or advanced?"},
                {"start": 10.0, "end": 10.1, "transcript": "   "},
            ]
        }
    }
    cues = P.cues_from_deepgram(payload)
    assert [c.text for c in cues] == [
        "What's your level of English?",
        "Are you a beginner, an intermediate, or advanced?",
    ]
    assert cues[0].start == 0.32 and cues[1].start == 4.5


def test_words_are_grouped_when_utterances_are_missing():
    """Some responses carry only words. Every response carries words."""
    def word(text, start, end):
        return {"punctuated_word": text, "start": start, "end": end}

    payload = {"results": {"channels": [{"alternatives": [{"words": [
        word("Hello", 1.0, 1.4), word("there.", 1.4, 1.9),
        word("How", 2.2, 2.4), word("are", 2.4, 2.6), word("you?", 2.6, 3.0),
        word("Fine", 3.5, 3.9),  # no full stop: the tail must survive
    ]}]}]}}
    cues = P.cues_from_deepgram(payload)
    assert [c.text for c in cues] == ["Hello there.", "How are you?", "Fine"]
    assert cues[0].start == 1.0
    assert cues[1].start == 2.2
    assert cues[2].start == 3.5


def test_an_empty_response_is_an_error_not_an_empty_video():
    with pytest.raises(P.PrepareError):
        P.cues_from_deepgram({"results": {}})


def test_transcribing_without_a_key_says_so_before_uploading_anything(tmp_path):
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"not really audio")
    with pytest.raises(P.PrepareError) as caught:
        P.transcribe(audio, "")
    assert "DEEPGRAM_API_KEY" in str(caught.value)


def test_the_caption_failure_names_the_real_cause():
    """The library leads with "the video is no longer available", which is
    almost never true for a video the user is watching in another window."""
    from youtube_transcript_api._errors import TranscriptsDisabled

    advice = P._caption_advice(TranscriptsDisabled("abc"))
    assert "captions turned off" in advice
    assert "--transcribe" in advice


def test_a_bot_check_is_not_reported_as_a_missing_video():
    from youtube_transcript_api._errors import VideoUnavailable

    advice = P._caption_advice(VideoUnavailable("abc"))
    assert "bot check" in advice
    assert "--transcribe" in advice


def test_an_unknown_failure_still_points_somewhere_useful():
    advice = P._caption_advice(RuntimeError("something new broke"))
    assert "something new broke" in advice
    assert "--transcribe" in advice


def test_the_download_does_not_depend_on_a_command_being_on_PATH():
    """The bug this encodes: "pip install yt-dlp" advice AFTER installing it.

    On Windows pip puts console scripts in a per-user Scripts directory that
    is routinely not on PATH - pip warns about it and nobody reads the
    warning. So `yt-dlp` the command is missing on a machine where
    `import yt_dlp` works perfectly, and shelling out turns a working install
    into advice the user has already followed.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path("app/controllers/pretranslate.py").read_text(encoding="utf-8")
    download = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "download_audio"
    )
    body = ast.dump(download)

    # The library first, and no PATH lookup deciding whether it is available.
    assert "yt_dlp" in body
    assert "YoutubeDL" in body
    assert "shutil" not in body, "download_audio still gates on PATH"

    fallback = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_download_audio_via_command"
    )
    # Even the fallback prefers `python -m yt_dlp` over a bare command.
    assert "executable" in ast.dump(fallback)


# ----------------------------------------------------------------------
# following the video: pause, resume, and starting on the first line
# ----------------------------------------------------------------------


class FakeRing:
    """A ring that hands out whatever the test says arrived."""

    def __init__(self):
        self.arriving = 0

    def read_available(self):
        n, self.arriving = self.arriving, 0
        return np.zeros(n, dtype=np.int16)


class FakeSource:
    def __init__(self):
        self.ring = FakeRing()


def test_the_clock_only_advances_while_the_video_plays():
    clock = FakeClock()
    video = P.PausableClock(clock=clock)

    video.start()
    clock.now += 5.0
    assert video.position() == pytest.approx(5.0)

    video.pause()
    clock.now += 100.0          # you went to make tea
    assert video.position() == pytest.approx(5.0)

    video.start()
    clock.now += 2.0
    assert video.position() == pytest.approx(7.0)


def test_a_paused_video_stops_the_urdu():
    """The bug: pausing the video left the Urdu running on against a frozen
    picture, and it never recovered."""
    clock = FakeClock()
    video = P.PausableClock(clock=clock)
    video.start()

    sink = FakeSink()
    script = P.Script()
    for start in (0.0, 1.0, 2.0):
        script.add(line(start, seconds=0.1))
    script.close()

    player = P.Player(sink, [], clock=clock, sleep=clock.sleep)
    spoken: list[float] = []

    async def drive():
        task = asyncio.ensure_future(
            player.follow(video, script, on_line=lambda ln: spoken.append(ln.start))
        )
        # Let the first line through, then pause as the viewer would.
        while not spoken:
            await clock.sleep(0.05)
        video.pause()
        for _ in range(200):        # ten seconds of wall time, paused
            await clock.sleep(0.05)
        assert spoken == [0.0], f"the Urdu kept going while paused: {spoken}"
        video.start()               # resume
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(drive())
    assert spoken == [0.0, 1.0, 2.0]


def test_the_monitor_starts_the_clock_when_audio_appears():
    """No keypress: audio arriving IS the video starting."""
    source = FakeSource()
    video = P.PausableClock(clock=lambda: 0.0)
    monitor = P.SourceMonitor(source, video, interval=0.01)

    assert not video.running
    source.ring.arriving = 480
    monitor.poll() and video.start()
    assert video.running


def test_the_monitor_measures_arrival_not_loudness():
    """A paused video goes ABSENT, not quiet.

    WASAPI stops delivering callbacks for an endpoint nothing renders into,
    so no frames arrive at all. A quiet passage in a PLAYING video still
    delivers frames - they are just near zero. Measuring amplitude would
    confuse the two and pause the clock every time the speaker drew breath,
    putting it permanently behind the picture.
    """
    source = FakeSource()
    source.ring.arriving = 480          # silent frames, but they ARRIVED
    monitor = P.SourceMonitor(source, P.PausableClock(), interval=0.01)
    assert monitor.poll() == 480

    source.ring.arriving = 0            # nothing at all: paused
    assert monitor.poll() == 0


def test_the_monitor_pauses_the_clock_after_audio_stops():
    clock = FakeClock()
    source = FakeSource()
    video = P.PausableClock(clock=clock)
    changes: list[bool] = []
    monitor = P.SourceMonitor(
        source, video, idle_after=0.2, interval=0.05,
        on_change=changes.append, sleep=clock.sleep,
    )

    async def drive():
        task = asyncio.ensure_future(monitor.run())
        source.ring.arriving = 480
        await clock.sleep(0.05)
        await clock.sleep(0.05)
        assert video.running, "audio arrived and the clock did not start"

        source.ring.arriving = 0        # the viewer hit pause
        for _ in range(10):
            await clock.sleep(0.05)
        assert not video.running, "audio stopped and the clock kept running"

        monitor.stop()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=1)

    asyncio.run(drive())
    assert changes == [True, False]


def test_playback_can_begin_on_the_first_line_while_the_rest_is_prepared():
    """The other bug: a minute of preparation before a word is spoken.

    Playback only ever needs the NEXT line, and preparing one takes a second
    or two while lines are spaced seconds apart.
    """
    clock = FakeClock()
    video = P.PausableClock(clock=clock)
    video.start()
    script = P.Script()
    script.add(line(0.0, seconds=0.1))          # only the first is ready

    player = P.Player(FakeSink(), [], clock=clock, sleep=clock.sleep)
    spoken: list[float] = []

    async def drive():
        task = asyncio.ensure_future(
            player.follow(video, script, on_line=lambda ln: spoken.append(ln.start))
        )
        while not spoken:
            await clock.sleep(0.05)
        assert spoken == [0.0], "it waited for the whole script"

        script.add(line(1.0, seconds=0.1))      # prepared while playing
        script.close()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(drive())
    assert spoken == [0.0, 1.0]


def test_the_player_waits_rather_than_skipping_when_preparation_lags():
    clock = FakeClock()
    video = P.PausableClock(clock=clock)
    video.start()
    script = P.Script()

    player = P.Player(FakeSink(), [], clock=clock, sleep=clock.sleep)
    spoken: list[float] = []

    async def drive():
        task = asyncio.ensure_future(
            player.follow(video, script, on_line=lambda ln: spoken.append(ln.start))
        )
        for _ in range(20):          # a second of nothing being ready
            await clock.sleep(0.05)
        assert spoken == []
        script.add(line(0.2, seconds=0.1))
        script.close()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(drive())
    assert spoken == [0.2], "a late-prepared line was dropped"


# ----------------------------------------------------------------------
# speaking in the video's own voice
# ----------------------------------------------------------------------


class FakeConverter:
    """Records what it was asked to convert, and marks the audio it returns."""

    def __init__(self, fail: bool = False, realtime_factor: float = 0.2) -> None:
        self.calls: list[tuple[int, float]] = []
        self.fail = fail
        self.realtime_factor = realtime_factor

    def convert(self, pcm, rate, reference):
        self.calls.append((pcm.size, reference.seconds))
        if self.fail:
            raise ProviderError("the converter fell over")
        return np.full(pcm.size, 7, dtype=np.int16), rate


def test_each_line_is_converted_to_the_reference_voice():
    from app.controllers.pretranslate import prepare, Cue
    from app.providers.voice_clone import Reference

    cues = [Cue(0.0, 2.0, "Hello there."), Cue(3.0, 5.0, "And welcome back.")]
    converter = FakeConverter()
    reference = Reference(np.zeros(24_000 * 10, dtype=np.int16), 24_000)

    lines = asyncio.run(prepare(
        cues, translator=FakeTranslator(), tts=FakeTts(), voice=VOICE,
        converter=converter, reference=reference,
    ))

    assert len(lines) == 2
    assert len(converter.calls) == 2
    # Every line carries the converted audio, not the synthesised audio.
    assert all(np.all(line.pcm == 7) for line in lines)
    # The reference reached it intact - ten seconds, not an empty clip.
    assert converter.calls[0][1] == pytest.approx(10.0)


def test_the_urdu_text_and_timings_are_untouched_by_conversion():
    """Conversion changes the voice, not the words or where they land."""
    from app.controllers.pretranslate import prepare, Cue
    from app.providers.voice_clone import Reference

    cues = [Cue(4.0, 6.0, "Hello there.")]
    plain = asyncio.run(prepare(
        cues, translator=FakeTranslator(), tts=FakeTts(), voice=VOICE))
    cloned = asyncio.run(prepare(
        cues, translator=FakeTranslator(), tts=FakeTts(), voice=VOICE,
        converter=FakeConverter(),
        reference=Reference(np.zeros(24_000, dtype=np.int16), 24_000),
    ))

    assert cloned[0].urdu == plain[0].urdu
    assert cloned[0].start == plain[0].start
    assert cloned[0].pcm.size == plain[0].pcm.size


def test_nothing_is_converted_when_no_reference_was_given():
    """The feature is opt-in twice over: a converter AND a voice to imitate."""
    from app.controllers.pretranslate import prepare, Cue

    converter = FakeConverter()
    lines = asyncio.run(prepare(
        [Cue(0.0, 2.0, "Hello there.")], translator=FakeTranslator(),
        tts=FakeTts(), voice=VOICE, converter=converter, reference=None,
    ))
    assert converter.calls == []
    assert not np.all(lines[0].pcm == 7)


def test_a_converter_that_fails_says_which_line():
    from app.controllers.pretranslate import prepare, PrepareError, Cue
    from app.providers.voice_clone import Reference

    with pytest.raises(PrepareError) as caught:
        asyncio.run(prepare(
            [Cue(0.0, 2.0, "Hello there.")], translator=FakeTranslator(),
            tts=FakeTts(), voice=VOICE, converter=FakeConverter(fail=True),
            reference=Reference(np.zeros(24_000, dtype=np.int16), 24_000),
        ))
    assert "line 1" in str(caught.value)
