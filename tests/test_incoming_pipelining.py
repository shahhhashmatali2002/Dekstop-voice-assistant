"""The incoming unit worker must PIPELINE, not serialise.

The defect these tests pin down: the original worker did

    translate -> synthesise -> wait for the first audible block

one unit at a time, so during fluent speech every unit paid its
predecessor's full network cost on top of its own and the lag compounded
without bound. The fix runs translations ahead (bounded lookahead) while
speaking stays strictly ordered.
"""

from __future__ import annotations

import asyncio

import numpy as np

from app.controllers.incoming import IncomingPipeline
from app.controllers.segmenter import Unit
from app.models.telemetry import MetricsBus, TraceFactory
from app.providers.base import PcmChunk, TranslatedUnit, VoiceId


class Log:
    def __init__(self) -> None:
        self.events: list[str] = []

    def add(self, event: str) -> None:
        self.events.append(event)

    def index(self, prefix: str) -> int:
        """First event starting with `prefix` - the spoken text gains an
        Urdu continuation mark on its way to the voice, so exact matches on
        tts events would miss."""
        for i, event in enumerate(self.events):
            if event.startswith(prefix):
                return i
        raise AssertionError(f"{prefix!r} not found in {self.events}")


class SlowTranslator:
    """Records when each translation starts and ends."""

    def __init__(self, log: Log, delay: float = 0.10) -> None:
        self.log = log
        self.delay = delay

    async def translate(self, unit, ctx, terms=None):
        self.log.add(f"translate_start {unit.text}")
        await asyncio.sleep(self.delay)
        self.log.add(f"translate_end {unit.text}")
        return TranslatedUnit(text=f"ur({unit.text})", source=unit)


class SlowTts:
    supports_speed = False

    def __init__(self, log: Log, delay: float = 0.10, rate: int = 24_000) -> None:
        self.log = log
        self.delay = delay
        self.rate = rate

    async def synthesize(self, text, voice):
        self.log.add(f"tts_start {text}")
        await asyncio.sleep(self.delay)
        # One second of audio - past the preroll cushion in one chunk.
        yield PcmChunk(pcm=np.zeros(self.rate, dtype=np.int16), rate=self.rate)
        self.log.add(f"tts_end {text}")


class NullSink:
    """No ring: the play loop writes directly and 'sounds' immediately."""

    rate = 24_000
    channels = 1

    def __init__(self) -> None:
        self.written = 0

    def write(self, pcm) -> None:
        self.written += pcm.size


def _pipeline(log: Log):
    tts = SlowTts(log)
    pipe = IncomingPipeline(
        asr=None,
        translator=SlowTranslator(log),
        tts=tts,
        voice=VoiceId(vendor="null", name="test", language="ur"),
        sink=NullSink(),
        metrics=MetricsBus(),
        traces=TraceFactory(),
    )
    return pipe


def _start_workers(pipe) -> list[asyncio.Task]:
    pipe._running = True
    return [
        asyncio.create_task(pipe._play_loop()),
        asyncio.create_task(pipe._translate_worker()),
        asyncio.create_task(pipe._speak_worker()),
    ]


async def _drain(tasks: list[asyncio.Task]) -> None:
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_next_translation_starts_while_previous_unit_is_speaking():
    log = Log()

    async def run() -> None:
        pipe = _pipeline(log)
        tasks = _start_workers(pipe)
        for i in range(3):
            pipe._enqueue(Unit(f"unit{i} words here", "stable"))
        # Long enough for all three units at ~0.2 s of work each even when
        # fully serialised - the ORDER of events is what is asserted, not
        # the wall time.
        await asyncio.sleep(1.5)
        await _drain(tasks)

    asyncio.run(run())

    # Unit 1's translation must begin while unit 0 is still being
    # synthesised - i.e. before unit 0's TTS has finished. Under the old
    # serial worker it began strictly after.
    assert log.index("translate_start unit1 words here") < log.index(
        "tts_end ur(unit0 words here)"
    ), f"translations did not overlap speech: {log.events}"


def test_units_are_spoken_in_the_order_they_were_confirmed():
    log = Log()

    async def run() -> None:
        pipe = _pipeline(log)
        # Unit 0 translates SLOWLY, unit 1 instantly - the fast one must
        # still be spoken second.
        delays = {"unit0": 0.3, "unit1": 0.0}

        class UnevenTranslator(SlowTranslator):
            async def translate(self, unit, ctx, terms=None):
                self.delay = delays[unit.text.split()[0]]
                return await super().translate(unit, ctx, terms)

        pipe.translator = UnevenTranslator(log)
        tasks = _start_workers(pipe)
        pipe._enqueue(Unit("unit0 words here", "stable"))
        pipe._enqueue(Unit("unit1 words here", "stable"))
        await asyncio.sleep(1.2)
        await _drain(tasks)

    asyncio.run(run())

    assert log.index("tts_start ur(unit0 words here)") < log.index(
        "tts_start ur(unit1 words here)"
    ), f"a fast translation overtook a slow one into the mouth: {log.events}"


def test_a_failed_translation_skips_the_unit_not_the_pipeline():
    log = Log()

    async def run() -> None:
        pipe = _pipeline(log)

        class FlakyTranslator(SlowTranslator):
            async def translate(self, unit, ctx, terms=None):
                if unit.text.startswith("unit0"):
                    raise RuntimeError("boom")
                return await super().translate(unit, ctx, terms)

        pipe.translator = FlakyTranslator(log, delay=0.0)
        tasks = _start_workers(pipe)
        pipe._enqueue(Unit("unit0 words here", "stable"))
        pipe._enqueue(Unit("unit1 words here", "stable"))
        await asyncio.sleep(0.8)
        await _drain(tasks)

    asyncio.run(run())

    assert any(e.startswith("tts_start ur(unit1") for e in log.events)
    assert not any(e.startswith("tts_start ur(unit0") for e in log.events)
