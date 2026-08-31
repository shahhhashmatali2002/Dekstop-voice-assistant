"""Record a short voice sample from your microphone and clone your voice via ElevenLabs.

Usage:
    python tools/clone_my_voice.py

What it does:
1. Prompts you to speak a natural English sentence for 8 seconds.
2. Captures your voice through your configured microphone.
3. Automatically uploads the sample to ElevenLabs to create your instant voice clone.
4. Saves ELEVENLABS_VOICE_ID directly to your .env file.
5. Plays a test sentence in your new cloned voice.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from app.audio.devices import DeviceManager
from app.audio.streams import CaptureSource
from app.models.config import Config, get_secret
from app.providers.base import VoiceId
from app.providers.tts import ElevenLabsTextToSpeech

RECORD_SECONDS = 8
RATE = 24_000
CHANNELS = 1


def update_env_file(key: str, value: str) -> None:
    env_path = Path(".env")
    lines = []
    found = False
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value


def record_audio(duration: float = RECORD_SECONDS) -> np.ndarray:
    manager = DeviceManager()
    cfg = Config.load()
    mic_name = cfg.devices.microphone
    dev = manager.find(mic_name, capture=True) if mic_name else manager.default_input()
    print(f"\n[1/4] Using microphone: {dev.name}")
    print(f"      Recording {duration:.0f} seconds of speech.")
    print("      👉 Please read this sentence aloud clearly into your mic:")
    print("      \"Hello! I am testing my desktop voice assistant, and this is my real voice sample for live translation.\"\n")
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...", flush=True)
        time.sleep(1)
    print("🎙️ RECORDING NOW... Speak!", flush=True)

    source = manager.open_capture(dev, rate=RATE, channels=CHANNELS)
    source.start()
    collected: list[np.ndarray] = []
    started = time.monotonic()
    
    try:
        while time.monotonic() - started < duration:
            time.sleep(0.05)
            chunk = source.ring.read_available()
            if chunk.size > 0:
                collected.append(chunk)
    finally:
        source.close()

    if not collected:
        print("❌ No audio captured from microphone.")
        return np.empty(0, dtype=np.int16)
    
    pcm = np.concatenate(collected)
    rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2))
    db = 20.0 * np.log10(rms / 32768.0 + 1e-12)
    print(f"\n[2/4] Recording complete. Captured {pcm.size / RATE:.1f}s (peak energy {db:.1f} dB).")
    if db < -55.0:
        print("⚠️ Warning: Microphone audio was very quiet. Make sure your mic is not muted or too quiet.")
    return pcm


async def upload_clone(api_key: str, wav_path: Path, voice_name: str = "Hashmat Ali (Cloned)") -> str:
    import aiohttp

    print("\n[3/4] Uploading voice sample to ElevenLabs VoiceLab...")
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": api_key}
    
    data = aiohttp.FormData()
    data.add_field("name", voice_name)
    data.add_field("description", "Desktop Voice Assistant Instant Voice Clone")
    
    with open(wav_path, "rb") as f:
        audio_bytes = f.read()
        data.add_field(
            "files",
            audio_bytes,
            filename="sample.wav",
            content_type="audio/wav",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"ElevenLabs API returned {resp.status}: {text}")
                import json
                res = json.loads(text)
                return res.get("voice_id", "")


def main() -> int:
    api_key = get_secret("ELEVENLABS_API_KEY")
    if not api_key or api_key.startswith("your_"):
        print("❌ ELEVENLABS_API_KEY is missing or invalid in your .env file.")
        print("   Please add a valid ELEVENLABS_API_KEY to .env first.")
        return 1

    pcm = record_audio()
    if pcm.size == 0:
        return 1

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(pcm.tobytes())

        voice_id = asyncio.run(upload_clone(api_key, tmp_path))
        if not voice_id:
            print("❌ Failed to retrieve voice ID from ElevenLabs.")
            return 1

        print(f"\n[4/4] ✅ Voice successfully cloned!")
        print(f"      Voice ID: {voice_id}")
        update_env_file("ELEVENLABS_VOICE_ID", voice_id)
        update_env_file("ENGLISH_VOICE", voice_id)
        print("      Saved ELEVENLABS_VOICE_ID and ENGLISH_VOICE to .env.")
        print("\n🎉 Done! Your voice assistant is now configured to speak English in YOUR real cloned voice.")
        return 0

    except Exception as exc:
        print(f"\n❌ Error during cloning: {exc}")
        return 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
