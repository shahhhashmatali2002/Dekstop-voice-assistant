"""Speaking the Urdu in the original speaker's voice.

Spec reference: §8.3, and the roadmap item this implements.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is not a cloning text-to-speech model. It cannot be, and the reason is
worth writing down because it is the whole design constraint:

    Checked in August 2026. ElevenLabs supports Urdu only in Eleven v3;
    Flash v2.5 and Multilingual v2 do not have Urdu at all, and Professional
    Voice Cloning is aligned to Flash v2.5's language list - so a cloned Urdu
    voice cannot be ordered from them. XTTS-v2 and OpenVoice v2, the usual
    local cloning models, have no Urdu either. Azure's personal voice is the
    closest hosted fit and is behind an access application.

So instead of asking a cloning model to speak Urdu, this lets the Urdu voice
we already have do the speaking, and then converts the TIMBRE of that audio
onto the target speaker. Voice conversion operates on the waveform, so the
absence of Urdu from any cloning model's language list does not apply to it.

    Urdu TTS (Edge / Azure)  ->  voice conversion onto a reference clip
                                 taken from the speaker's own audio

The reference is short - one to thirty seconds is enough for a zero-shot
converter - which matters, because for a video the reference is simply a clip
of the video.

WHY IT IS OFF BY DEFAULT
------------------------
Conversion is the first thing in this project that wants a GPU. On the
measured reference hardware for Seed-VC (an RTX 3060 laptop) a chunk takes
about 150 ms, which is comfortably faster than real time; on a CPU it is
several times slower than real time, which is unusable live and merely slow
for a recording prepared in advance. So the provider exists, reports honestly
how fast it is on the machine it finds, and nothing turns it on without being
asked. `tools/clone_bench.py` is how that decision gets made with a number
instead of a hope.

CONSENT
-------
A voice is not a setting. Converting a video's narration for your own
listening is one thing; playing a converted voice to another person on a call
is another, and that needs the knowledge of whoever is being imitated. The
reference clip and the model stay on this machine - nothing here uploads a
voice anywhere.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.audio.formats import DTYPE
from app.providers.base import ProviderMisconfigured, ProviderUnavailable


def best_device(preferred: str = "") -> str:
    """"cuda" when torch reports a usable GPU, "cpu" otherwise.

    Never raises: torch is an optional install, and a machine without it must
    still be able to load this module and be told what is missing.
    """
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class Reference:
    """The voice to sound like: a short clip and its sample rate."""

    pcm: np.ndarray
    rate: int

    @property
    def seconds(self) -> float:
        return self.pcm.size / float(self.rate or 1)


class VoiceConverter(ABC):
    """Rewrites the voice of already-synthesised speech."""

    name = "voice-converter"

    #: Whether this implementation is fast enough to sit on the live path on
    #: the machine it is running on. Set after `measure`, never guessed.
    realtime_factor: float | None = None

    #: How much reference audio this backend actually wants. It differs by an
    #: order of magnitude between the two, and using the wrong figure wastes
    #: either quality or time.
    preferred_reference_s: float = 10.0

    @abstractmethod
    def convert(
        self, pcm: np.ndarray, rate: int, reference: Reference
    ) -> tuple[np.ndarray, int]:
        """Same words, same timing, different voice."""

    def measure(
        self, pcm: np.ndarray, rate: int, reference: Reference
    ) -> float:
        """Seconds of compute per second of audio, timed on REAL speech.

        Below 1.0 the converter can keep up with playback; above it, a
        recording has to be prepared before playback starts and the live path
        is out of the question. A measurement rather than a lookup, because the
        same code on a laptop CPU and a desktop GPU differ by more than an
        order of magnitude.

        It has to be given real speech, and that is not a preference. The first
        version of this timed a synthesised sine tone, which seemed tidy - no
        provider call needed - and produced, on the user's machine:

            cannot reshape tensor of 0 elements into shape [-1, 0]

        kNN-VC trims non-speech from its input before matching. A pure tone is
        entirely non-speech, so it was trimmed to nothing and the model was
        handed an empty tensor. A benchmark that cannot be run on the thing it
        benchmarks is worth less than no benchmark.
        """
        seconds = max(pcm.size / float(rate or 1), 1e-6)
        started = time.monotonic()
        self.convert(pcm, rate, reference)
        self.realtime_factor = (time.monotonic() - started) / seconds
        return self.realtime_factor


class NullVoiceConverter(VoiceConverter):
    """Hands the audio back unchanged. The default, and the fallback."""

    name = "null-converter"
    realtime_factor = 0.0

    def convert(
        self, pcm: np.ndarray, rate: int, reference: Reference
    ) -> tuple[np.ndarray, int]:
        return pcm, rate


class SeedVoiceConverter(VoiceConverter):
    """Zero-shot conversion via Seed-VC, on whatever device is available.

    Chosen over RVC because it needs no per-voice training run: a reference
    clip of a few seconds is enough, which is the only workable shape when the
    voice you want is a stranger in a video you started two minutes ago.

    The import is deliberately lazy and the failure deliberately loud. This is
    an optional extra - torch and a model download - and the application must
    start, and run, on a machine where none of it is installed.
    """

    name = "seed-vc"
    preferred_reference_s = 10.0

    def __init__(self, device: str = "", model: str = "") -> None:
        self._device = device
        self._model = model
        self._pipeline = None

    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        return best_device(self._device)

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from seed_vc import SeedVC  # type: ignore
        except ImportError as exc:
            raise ProviderMisconfigured(
                "Seed-VC is not installed, and on Python 3.13 it cannot be: it "
                "pins numpy below 2.0, which has no wheel for 3.13 and so tries "
                "to compile from source. Use the knn-vc backend instead - it "
                "needs only torch and torchaudio, both of which have wheels. "
                "See tools/clone_bench.py."
            ) from exc
        try:
            self._pipeline = (
                SeedVC(self._model, device=self.device)
                if self._model
                else SeedVC(device=self.device)
            )
        except Exception as exc:
            raise ProviderUnavailable(f"Seed-VC failed to load: {exc}") from exc
        return self._pipeline

    def convert(
        self, pcm: np.ndarray, rate: int, reference: Reference
    ) -> tuple[np.ndarray, int]:
        if pcm.size == 0:
            return pcm, rate
        if reference.pcm.size == 0:
            raise ProviderMisconfigured("no reference clip to sound like")

        pipeline = self._load()
        try:
            out, out_rate = pipeline.convert(
                source=(pcm.astype(np.float32) / 32768.0, rate),
                target=(reference.pcm.astype(np.float32) / 32768.0, reference.rate),
            )
        except Exception as exc:
            raise ProviderUnavailable(f"voice conversion failed: {exc}") from exc

        out = np.asarray(out, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out = out / peak
        return (out * 32767.0).astype(DTYPE), int(out_rate or rate)


class KnnVoiceConverter(VoiceConverter):
    """Voice conversion by nearest-neighbour feature matching (kNN-VC).

    WHY THIS IS THE DEFAULT AND SEED-VC IS NOT
    ------------------------------------------
    Seed-VC pins numpy below 2.0. There is no numpy 1.x wheel for Python 3.13,
    so pip tries to build it from source, which needs a C compiler nobody
    installed on a laptop that just wanted to watch a video in Urdu. kNN-VC
    needs torch, torchaudio and any numpy - all of which have wheels - and
    loads its weights through torch.hub.

    HOW IT WORKS, BECAUSE IT EXPLAINS THE REFERENCE LENGTH
    ------------------------------------------------------
    It converts by lookup rather than by generation. Every short frame of the
    synthesised Urdu is described as a self-supervised speech feature, the
    nearest few frames in the TARGET speaker's own audio are found, and the
    output is vocoded from those. So the words and the timing come from our
    Urdu, and every sample of the timbre comes from real recordings of the
    person.

    Two consequences follow, and both are useful here. It is inherently
    language-independent: nothing in it knows or cares that the source is Urdu
    and the reference is English, because it is matching acoustics, not
    phonemes - which is exactly the property that no cloning TTS gives us for
    Urdu. And more reference audio directly improves it, because there are more
    frames to match against. Quality stops improving beyond about five minutes.

    That is why this backend asks for minutes rather than seconds. For a video
    it costs nothing: the whole audio is already downloaded.
    """

    name = "knn-vc"
    #: Its own working rate. The player resamples per line, so returning 16 kHz
    #: is correct rather than convenient.
    RATE = 16_000
    preferred_reference_s = 150.0

    #: One loaded model per process, shared by every instance. A new
    #: converter is built each time translation starts, and reloading a
    #: gigabyte of weights per Start press was measured doing real damage:
    #: the load holds the GIL in bursts, the audio loop freezes for seconds
    #: (the stall watchdog printed 3.0s and 5.0s), and Deepgram closes both
    #: directions with NET-0001 - blamed, naturally, on the network.
    _shared_model = None
    _shared_key = None

    def __init__(self, device: str = "", topk: int = 4, prematched: bool = True):
        self._device = device
        self.topk = topk
        self.prematched = prematched
        self._model = None
        self._matching = None      # the target's feature set, built once
        self._matched_ref = None

    @property
    def device(self) -> str:
        return best_device(self._device)

    def _load(self):
        if self._model is not None:
            return self._model
        cls = type(self)
        key = (self.device, self.prematched)
        if cls._shared_model is not None and cls._shared_key == key:
            self._model = cls._shared_model
            return self._model
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError as exc:
            raise ProviderMisconfigured(
                "voice conversion needs PyTorch:\n"
                "    .venv\\Scripts\\python -m pip install torch torchaudio\n"
                "The voice model itself downloads on first use (about 1 GB). "
                "Run tools/clone_bench.py to see how fast it is here."
            ) from exc
        import torch

        try:
            self._model = torch.hub.load(
                "bshall/knn-vc", "knn_vc",
                prematched=self.prematched, trust_repo=True, pretrained=True,
                device=self.device,
            )
        except Exception as exc:
            raise ProviderUnavailable(
                f"could not load the kNN-VC model: {exc}"
            ) from exc
        cls._shared_model, cls._shared_key = self._model, key
        return self._model

    # ------------------------------------------------------------------

    def _as_tensor(self, pcm: np.ndarray, rate: int):
        """Mono float32 at 16 kHz, shaped (1, n) the way torchaudio expects."""
        import torch

        from app.audio import convert as audio_convert

        if rate != self.RATE:
            pcm = audio_convert.resample(pcm, rate, self.RATE)
        return torch.from_numpy(
            (pcm.astype(np.float32) / 32768.0).reshape(1, -1).copy()
        )

    def _matching_set(self, reference: Reference):
        """The target speaker's features, computed once per reference.

        Not per line. Building this from two minutes of audio is the expensive
        half of the whole operation, and it is identical for every line of the
        video - recomputing it six hundred times would take longer than the
        conversions themselves.
        """
        if self._matching is not None and self._matched_ref is reference:
            return self._matching
        model = self._load()
        wav = self._as_tensor(reference.pcm, reference.rate)
        self._matching = model.get_matching_set([wav])
        self._matched_ref = reference
        return self._matching

    def convert(
        self, pcm: np.ndarray, rate: int, reference: Reference
    ) -> tuple[np.ndarray, int]:
        if pcm.size == 0:
            return pcm, rate
        if reference.pcm.size == 0:
            raise ProviderMisconfigured("no reference clip to sound like")

        model = self._load()
        matching = self._matching_set(reference)
        try:
            query = model.get_features(self._as_tensor(pcm, rate))
            out = model.match(query, matching, topk=self.topk)
        except Exception as exc:
            raise ProviderUnavailable(f"voice conversion failed: {exc}") from exc

        out = np.asarray(out.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        # SEEN IN A LIVE CALL: the model returned NaNs, and NaN.astype(int16)
        # is not silence - it is whatever bit pattern the cast lands on,
        # heard as harsh digital garbage in place of the whole utterance.
        # A clone that cannot produce a number has nothing to offer over the
        # audio it was given; hand that back unchanged instead.
        if out.size == 0 or not np.all(np.isfinite(out)):
            raise ProviderUnavailable(
                "voice conversion produced non-finite audio"
            )
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out = out / peak
        return (out * 32767.0).astype(DTYPE), self.RATE


class LightVoiceMatcher(VoiceConverter):
    """A resemblance, on a CPU, in real time - not a clone.

    kNN-VC produces the speaker's actual voice and needs a GPU: measured on
    the user's laptop at 4.3 seconds of compute per second of audio, which
    would put every sentence seventeen seconds further behind. That is not a
    tuning problem, it is a factor of twelve.

    So this does the part that is cheap, and says plainly that it is not the
    same thing. Two properties carry most of what makes a voice recognisable
    at a glance, and both can be measured from a few seconds of speech and
    imposed on ours with arithmetic:

      PITCH   - the median fundamental of the speaker against that of our
                voice, applied as a shift. A woman's voice reading for a man
                is the single most obvious mismatch there is, and it is one
                number.
      COLOUR  - the long-term average spectrum: where this speaker's energy
                sits across frequency, which is most of what "bright",
                "warm", "nasal" or "chesty" mean. Imposed as a smoothed,
                clamped equaliser curve.

    What it does not do is anything about articulation, timing, or the
    idiosyncrasies that make a voice that person's rather than someone else's
    with the same pitch. It will sound like a different person of the right
    build and register reading the translation. That is worth having when the
    alternative is a stock narrator, and it is not worth calling a clone.

    Cheap enough to be uninteresting: a few FFTs per second of audio, far
    under real time on any machine that can run the rest of this application.
    """

    name = "light-match"
    preferred_reference_s = 20.0

    #: Voices this far apart are not the same person and pretending otherwise
    #: produces artefacts rather than resemblance. An octave is 2.0; this is
    #: about a fifth either way.
    MIN_RATIO = 0.7
    MAX_RATIO = 1.45

    #: How hard the equaliser is allowed to push, in dB. Beyond this it stops
    #: correcting a voice and starts inventing one, and inventing one sounds
    #: like a fault.
    MAX_TILT_DB = 9.0

    WINDOW = 1024
    HOP = 256

    def __init__(self, device: str = "") -> None:
        self._device = device      # accepted for a uniform constructor; unused
        self._target = None        # (median f0, smoothed log spectrum)
        self._for = None

    # -- measurement ---------------------------------------------------

    def _frames(self, x: np.ndarray) -> np.ndarray:
        n = self.WINDOW
        if x.size < n:
            x = np.pad(x, (0, n - x.size))
        count = 1 + (x.size - n) // self.HOP
        strides = (x.strides[0] * self.HOP, x.strides[0])
        return np.lib.stride_tricks.as_strided(x, (count, n), strides)

    def median_f0(self, pcm: np.ndarray, rate: int) -> float:
        """The speaker's median pitch, by autocorrelation. 0.0 if unvoiced.

        Deliberately crude - per-frame autocorrelation with a median over the
        voiced frames. A median is what makes it robust: half the frames of
        any real recording are noise, breath or silence, and they land all over
        the place while the voiced ones cluster.
        """
        x = pcm.astype(np.float32)
        frames = self._frames(x)
        loud = np.abs(frames).mean(axis=1)
        if not loud.size:
            return 0.0
        voiced = frames[loud > max(np.percentile(loud, 60), 1.0)]
        if not voiced.size:
            return 0.0

        lo = max(1, int(rate / 400.0))     # 400 Hz ceiling
        hi = min(self.WINDOW - 1, int(rate / 60.0))   # 60 Hz floor
        if hi <= lo:
            return 0.0

        found = []
        for frame in voiced[:200]:         # a couple of seconds is plenty
            frame = frame - frame.mean()
            corr = np.correlate(frame, frame, mode="full")[frame.size - 1:]
            if corr[0] <= 0:
                continue
            window = corr[lo:hi]
            if not window.size:
                continue
            peak = int(np.argmax(window)) + lo
            # A peak barely above the noise is not a period, it is a guess.
            if corr[peak] > 0.3 * corr[0]:
                found.append(rate / peak)
        return float(np.median(found)) if found else 0.0

    def spectrum(self, pcm: np.ndarray) -> np.ndarray:
        """Smoothed long-term average spectrum, in dB."""
        window = np.hanning(self.WINDOW).astype(np.float32)
        frames = self._frames(pcm.astype(np.float32)) * window
        magnitude = np.abs(np.fft.rfft(frames, axis=1)).mean(axis=0)
        db = 20.0 * np.log10(np.maximum(magnitude, 1e-6))
        # Smoothed hard, because the point is the shape of the voice and not
        # the words that happened to be in the sample.
        kernel = np.ones(21, dtype=np.float32) / 21.0
        return np.convolve(db, kernel, mode="same")

    def _profile(self, reference: Reference):
        if self._for is reference and self._target is not None:
            return self._target
        self._target = (
            self.median_f0(reference.pcm, reference.rate),
            self.spectrum(reference.pcm),
        )
        self._for = reference
        return self._target

    # -- application ---------------------------------------------------

    def pitch_shift(self, pcm: np.ndarray, ratio: float, rate: int) -> np.ndarray:
        """Move the pitch by `ratio`, keeping the duration.

        Resample to change pitch and duration together, then put the duration
        back with the overlap-add stretcher the drift controller already uses.
        Never a plain resample on its own: that is the chipmunk voice, and it
        is disqualifying.
        """
        from app.audio import convert as audio_convert
        from app.controllers.drift import _wsola

        if abs(ratio - 1.0) < 0.02 or pcm.size == 0:
            return pcm
        moved = audio_convert.resample(pcm, rate, max(1, int(rate / ratio)))
        if moved.size == 0:
            return pcm
        restored = _wsola(moved, 1.0 / ratio, rate).astype(DTYPE)
        # Back to exactly the length we were given. The stretcher lands within
        # a per cent or two, and a per cent per utterance is a drift that
        # accumulates against a video's clock over an hour.
        if restored.size < pcm.size:
            return np.pad(restored, (0, pcm.size - restored.size))
        return restored[: pcm.size]

    def apply_tilt(self, pcm: np.ndarray, curve_db: np.ndarray) -> np.ndarray:
        """Impose the target's spectral shape, one frame at a time."""
        gain = 10.0 ** (np.clip(curve_db, -self.MAX_TILT_DB, self.MAX_TILT_DB) / 20.0)
        window = np.hanning(self.WINDOW).astype(np.float32)
        x = pcm.astype(np.float32)
        out = np.zeros(x.size + self.WINDOW, dtype=np.float32)
        norm = np.zeros_like(out)

        for start in range(0, max(1, x.size - self.WINDOW), self.HOP):
            frame = x[start : start + self.WINDOW]
            if frame.size < self.WINDOW:
                frame = np.pad(frame, (0, self.WINDOW - frame.size))
            spectrum = np.fft.rfft(frame * window)
            shaped = np.fft.irfft(spectrum * gain, n=self.WINDOW)
            out[start : start + self.WINDOW] += shaped * window
            norm[start : start + self.WINDOW] += window ** 2

        out = out[: x.size] / np.maximum(norm[: x.size], 1e-6)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 32767.0:
            out = out * (32767.0 / peak)
        return out.astype(DTYPE)

    def convert(
        self, pcm: np.ndarray, rate: int, reference: Reference
    ) -> tuple[np.ndarray, int]:
        if pcm.size == 0:
            return pcm, rate
        if reference.pcm.size == 0:
            raise ProviderMisconfigured("no reference clip to sound like")

        target_f0, target_db = self._profile(reference)
        ours_f0 = self.median_f0(pcm, rate)

        out = pcm
        if target_f0 > 0 and ours_f0 > 0:
            ratio = float(np.clip(target_f0 / ours_f0, self.MIN_RATIO, self.MAX_RATIO))
            out = self.pitch_shift(out, ratio, rate)

        ours_db = self.spectrum(out)
        size = min(ours_db.size, target_db.size)
        out = self.apply_tilt(out, target_db[:size] - ours_db[:size])
        return out, rate


class VoiceLearner:
    """Builds a reference voice out of the audio as it plays.

    The recorded path has it easy: the whole file is on disk before anything
    starts, so a good reference clip can simply be chosen. Live there is no
    file and no future - at the moment the user presses Start we have never
    heard this speaker. So the voice is learned from the stream itself.

    Only frames the voice-activity gate called speech are kept, because a
    reference built from room tone converts every line into room tone. Silence
    is not merely useless here, it is actively harmful.

    Nothing is converted until `min_seconds` of speech has accumulated, which
    means the first half-minute or so comes out in the stock voice and then
    the speaker's own voice takes over. That is a deliberate trade: the
    alternative is either waiting in silence or converting against two seconds
    of audio, which sounds worse than not converting at all.

    The reference then IMPROVES at intervals rather than continuously. Each
    upgrade costs a full rebuild of the converter's feature set, so doing it
    per utterance would cost more than the conversions - but a reference that
    never grew would be stuck with whatever the first thirty seconds happened
    to contain.
    """

    def __init__(
        self,
        rate: int = 16_000,
        *,
        min_seconds: float = 2.0,
        cap_seconds: float = 150.0,
        upgrade_every: float = 45.0,
    ) -> None:

        self.rate = rate
        self.min_seconds = min_seconds
        self.cap_seconds = cap_seconds
        self.upgrade_every = upgrade_every
        self._heard: list[np.ndarray] = []
        self._samples = 0
        self._reference: Reference | None = None
        self._at_last_upgrade = 0.0

    @property
    def seconds(self) -> float:
        return self._samples / float(self.rate or 1)

    @property
    def ready(self) -> bool:
        return self.seconds >= self.min_seconds

    def hear(self, pcm: np.ndarray) -> None:
        """One frame the gate called speech."""
        if self._samples >= self.cap_seconds * self.rate:
            return
        if pcm.size:
            self._heard.append(pcm.copy())
            self._samples += pcm.size

    def reference(self) -> Reference | None:
        """The voice so far, or None while there is not enough of it.

        Returns the SAME object until there is enough new material to be worth
        rebuilding for - converters cache their expensive per-reference work on
        object identity, so handing back a fresh object every call would
        silently make every line cost as much as the first.
        """
        if not self.ready:
            return None
        grown = self.seconds - self._at_last_upgrade
        if self._reference is not None and grown < self.upgrade_every:
            return self._reference
        self._reference = Reference(np.concatenate(self._heard), self.rate)
        self._at_last_upgrade = self.seconds
        return self._reference

    def reset(self) -> None:
        self._heard.clear()
        self._samples = 0
        self._reference = None
        self._at_last_upgrade = 0.0


# ----------------------------------------------------------------------


BACKENDS = {
    "knn": KnnVoiceConverter,
    "light": LightVoiceMatcher,
    "seed": SeedVoiceConverter,
    "off": NullVoiceConverter,
}


def build_converter(backend: str = "knn", device: str = "") -> VoiceConverter:
    """One of the backends by name. `knn` unless there is a reason."""
    try:
        factory = BACKENDS[backend]
    except KeyError:
        raise ProviderMisconfigured(
            f"unknown voice conversion backend {backend!r}; "
            f"try {', '.join(BACKENDS)}"
        ) from None
    if factory is NullVoiceConverter:
        return factory()
    return factory(device=device)


# ----------------------------------------------------------------------


#: Where to put the line between the male and the female Urdu voice. Between
#: about 165 Hz and 180 Hz the two overlap in real speakers; 160 sits just
#: under that, so an ambiguous voice keeps the default rather than flipping
#: back and forth between two different-sounding narrators.
MALE_F0_CEILING = 160.0

VOICES_BY_SEX = {
    "male": "ur-PK-AsadNeural",
    "female": "ur-PK-UzmaNeural",
}


def voice_for_pitch(f0: float) -> str:
    """The Urdu voice closest to a speaker of this pitch, or "" if unclear.

    Worth more than any amount of signal processing. Pitch-shifting a female
    voice down an octave to stand in for a man does not produce a man, it
    produces a distorted woman - whereas simply reading the translation with
    the male voice starts an octave closer and has no artefacts at all. The
    processing is then only closing the last few per cent.
    """
    if f0 <= 0:
        return ""
    return VOICES_BY_SEX["male"] if f0 < MALE_F0_CEILING else VOICES_BY_SEX["female"]


def pick_reference(
    pcm: np.ndarray, rate: int, *, seconds: float = 10.0, window_ms: int = 200
) -> Reference:
    """The stretch of audio that best represents a speaker's voice.

    Not the first ten seconds: a video's first ten seconds are an intro sting,
    a logo, or a musical bed, and a reference clip of music produces a
    converted voice that sounds like a fault. The densest run of continuous
    speech-level energy is a crude proxy for "someone talking uninterrupted",
    and crude is enough - the converter needs a voice, not a performance.
    """
    if pcm.size == 0:
        return Reference(pcm, rate)
    want = int(seconds * rate)
    if pcm.size <= want:
        return Reference(pcm, rate)

    step = max(1, int(rate * window_ms / 1000))
    frames = pcm[: (pcm.size // step) * step].reshape(-1, step)
    energy = np.abs(frames.astype(np.float32)).mean(axis=1)
    if frames.shape[0] < 2:
        return Reference(pcm[:want].copy(), rate)

    # Ordinary speaking level, not maximum level. Ranking windows by loudness
    # picks the music sting every time, because a mastered sting is louder than
    # any voice - so the target is the MEDIAN of the non-quiet windows, and a
    # window far above it is treated as not-a-voice just as firmly as a window
    # far below it. (A test in tests/test_voice_clone.py exists precisely
    # because the loudness version of this passed a casual reading and then
    # chose four seconds of a bass drone.)
    voiced = energy[energy > np.percentile(energy, 40)]
    typical = float(np.median(voiced)) if voiced.size else float(np.median(energy))
    good = ((energy > 0.5 * typical) & (energy < 1.8 * typical)).astype(np.float32)

    span = max(1, want // step)
    if good.size < span:
        return Reference(pcm[:want].copy(), rate)
    run = np.convolve(good, np.ones(span), "valid")
    best = int(np.argmax(run)) * step
    return Reference(pcm[best : best + want].copy(), rate)
