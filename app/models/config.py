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
    translator_model: str = "gpt-4.1-mini"
    region: str = ""


@dataclass
class PipelineConfig:
    """The three settings that dominate TTFTA. Tune against recordings (§9.3)."""

    vad_hangover_ms: float = 400.0
    vad_preroll_ms: float = 250.0
    stability_hypotheses: int = 3
    stability_ms: float = 350.0
    max_words: int = 12
    barge_in_confirm_ms: float = 250.0
    suppress_backchannels: bool = True
    #: 0.0 = the consultant hears only Urdu, which is the brief. A small
    #: negative value mixes the original back in for turn-taking cues (§18 O4).
    original_mix_db: float | None = None


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
