"""Configuration and credentials.

Spec reference: §12.2, §13.5.

API keys go in Windows Credential Manager through `keyring` - never in the
database, never in a config file, never in an environment variable baked into
the installer, and never in a frozen Python bundle where they are trivially
recoverable. The settings file stores only a handle.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SERVICE = "voice-assistant"


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
    return Path(base) / "VoiceAssistant"


@dataclass
class DeviceConfig:
    """Endpoints, pinned by name for now; by endpoint ID from Phase 6."""

    microphone: str = ""
    headphones: str = ""
    virtual_mic: str = ""  # the render half we write English into
    teams_speaker: str = ""  # the render half Teams outputs to; we loopback it


@dataclass
class ProviderConfig:
    s2st: str = "openai-realtime-translate"
    asr: str = "deepgram-nova-3"
    translator: str = "llm-translator"
    tts: str = "azure-neural-tts"
    urdu_voice: str = "ur-PK-UzmaNeural"
    translator_model: str = "gpt-4o-mini"
    region: str = ""


@dataclass
class PipelineConfig:
    """The settings that dominate TTFTA and stability."""

    vad_hangover_ms: float = 350.0
    vad_preroll_ms: float = 250.0
    stability_hypotheses: int = 3
    stability_ms: float = 300.0
    max_words: int = 12
    barge_in_confirm_ms: float = 250.0
    suppress_backchannels: bool = True

    #: Synthesised audio held back before playback starts, in milliseconds.
    #: The render callback consumes the sink ring continuously, so writing a
    #: TTS chunk the instant it arrives means the first slow chunk on the
    #: network drains the ring and the listener hears a hole punched through
    #: the middle of a word. A small buffer absorbs that jitter. It is paid
    #: once per utterance, not per chunk.
    playback_preroll_ms: float = 250.0

    #: Drift control (§7.6). The defaults keep a live meeting close to real
    #: time and are deliberately aggressive: speed up past 400 ms of queue,
    #: drop past 3 s. Watching a recording wants the opposite trade - see the
    #: presets in listen.py.
    #: These were tuned when queue depth was read from the sink ring, which
    #: is a fixed half-second buffer: the signal could never report more than
    #: 500 ms however far behind the pipeline actually was, so a 3 s drop
    #: threshold was unreachable and never fired. The measurement is now the
    #: real lag, and a cascade through recognition, translation and synthesis
    #: genuinely sits three to six seconds behind - so the old numbers would
    #: drop every single unit and play nothing at all. Raised to match what
    #: they now mean.
    drift_comfortable_ms: float = 2000.0
    drift_stretch_ms: float = 6000.0
    drift_aggressive_ms: float = 12000.0
    drift_max_rate: float = 1.12
    original_mix_db: float | None = None



@dataclass
class ListenConfig:
    """Listen mode: one way, any application, no microphone.

    A separate topology from the meeting mode above, and deliberately a
    separate section: it needs one cable rather than two and no microphone at
    all, so reusing `DeviceConfig` here would mean four fields of which two
    are always blank.

    `source` is the *render* endpoint the source application is pointed at -
    "CABLE Input" - not its loopback twin. We ask Windows for the loopback of
    that endpoint at open time, because the loopback endpoint is named after
    its render device and is the thing the user cannot pick from a list of
    playback devices.

    `output` is where the Urdu goes. Blank means whatever Windows is using,
    which is the right default: it is the device the user is wearing.
    """

    source: str = ""
    output: str = ""
    #: Which fluency/lag trade to run at. Listen mode is for watching
    #: something, where nothing is waiting on you, so whole thoughts spoken a
    #: few seconds late beat clipped phrases spoken sooner. See PRESETS in
    #: app/controllers/listen.py.
    preset: str = "fluent"
    #: Start translating as soon as the window opens, rather than waiting for
    #: Start. Convenient once the routing is set up and never changes.
    autostart: bool = False


@dataclass
class UiConfig:
    bubble_x: int = -1
    bubble_y: int = -1
    bubble_monitor: str = ""
    transcript_open: bool = False
    exclude_from_capture: bool = True  # §2.4 - never default this off
    hotkey_pause: str = "Ctrl+Alt+Space"
    hotkey_start_stop: str = "Ctrl+Alt+T"
    hotkey_hide: str = "Ctrl+Alt+H"
    hotkey_repeat: str = "Ctrl+Alt+R"
    launch_at_login: bool = True  # §2.5 - the wire should be up before Teams


@dataclass
class Config:
    devices: DeviceConfig = field(default_factory=DeviceConfig)
    providers: ProviderConfig = field(default_factory=ProviderConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    listen: ListenConfig = field(default_factory=ListenConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    setup_complete: bool = False

    # ------------------------------------------------------------------

    @classmethod
    def path(cls) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else cls.path()
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A corrupt settings file must not stop the application starting -
            # passthrough matters more than preferences (§2.5).
            return cls()
        return cls(
            devices=DeviceConfig(**data.get("devices", {})),
            providers=ProviderConfig(**data.get("providers", {})),
            pipeline=PipelineConfig(**data.get("pipeline", {})),
            listen=ListenConfig(**data.get("listen", {})),
            ui=UiConfig(**data.get("ui", {})),
            setup_complete=bool(data.get("setup_complete", False)),
        )

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def terminology_path(self) -> Path:
        return config_dir() / "terminology.json"


# ----------------------------------------------------------------------


def _load_env_file() -> None:
    for base_dir in (Path.cwd(), Path(__file__).resolve().parent.parent.parent):
        env_path = base_dir / ".env"
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            except Exception:
                pass


def get_secret(name: str) -> str:
    """Read a credential. Credential Manager first, environment as fallback.

    The environment fallback exists for development and for the offline tools;
    the shipped application should always be using the credential store.
    """
    try:
        import keyring

        value = keyring.get_password(SERVICE, name)
        if value:
            return value
    except Exception:
        pass
    if name not in os.environ:
        _load_env_file()
    return os.environ.get(name, "")


def set_secret(name: str, value: str) -> bool:
    try:
        import keyring

        keyring.set_password(SERVICE, name, value)
        return True
    except Exception:
        return False


def delete_secret(name: str) -> bool:
    try:
        import keyring

        keyring.delete_password(SERVICE, name)
        return True
    except Exception:
        return False


SECRET_NAMES = (
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_REGION",
    "ELEVENLABS_API_KEY",
)


def missing_secrets(cfg: Config) -> list[str]:
    """Which credentials the configured provider stack still needs."""
    need: set[str] = set()
    p = cfg.providers
    if p.s2st.startswith("openai") or p.translator.startswith("llm"):
        need.add("OPENAI_API_KEY")
    if p.asr.startswith("deepgram"):
        need.add("DEEPGRAM_API_KEY")
    if p.tts.startswith("azure"):
        need.update({"AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION"})
    if p.tts.startswith("eleven"):
        need.add("ELEVENLABS_API_KEY")
    return sorted(n for n in need if not get_secret(n))
