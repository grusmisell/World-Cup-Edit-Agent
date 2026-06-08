"""ElevenLabs text-to-speech (optional, paid) — far more human than edge-tts.

When ELEVENLABS_API_KEY is set, the voiceover pipelines use this instead of
edge-tts. Pick any voice from your ElevenLabs Voice Library (licensed pro voices)
via its voice_id. Cloning a real person's voice without consent violates
ElevenLabs' ToS — stick to the library / your own voice.
"""
from __future__ import annotations

import json
import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path

from ..config import settings

API = "https://api.elevenlabs.io/v1"

# Validate TLS against certifi (the system store can be stale on this machine).
try:
    import certifi
    _SSL_CTX: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _SSL_CTX = None


# Curated premade voices that work on the FREE tier (verified) — always offered so
# the picker works even when the key lacks 'voices read' permission. Library voices
# the user adds (if the key can read them) are merged on top.
PREMADE: dict[str, str] = {
    "onwK4e9ZLuTAKqWW03F9": "Daniel — deep British, epic narrator ⭐",
    "JBFqnCBsd6RMkjVDRZzb": "George — raspy British, cinematic",
    "pNInz6obpgDQGcFmaJgB": "Adam — deep American narrator",
    "nPczCjzI2devNBz1zQrb": "Brian — deep American, calm",
    "VR6AewLTigWG4xSOukaG": "Arnold — crisp, energetic (hype)",
    "ErXwobaYiN019PkySvjV": "Antoni — warm American",
}


def is_configured() -> bool:
    return bool(settings.elevenlabs_api_key.strip())


def list_voices() -> dict[str, str]:
    """Return {voice_id: name}: the user's library voices (if the key can read them)
    plus the curated free-tier premades. Always returns the premades when configured."""
    if not is_configured():
        return {}
    voices: dict[str, str] = {}
    req = urllib.request.Request(
        f"{API}/voices", headers={"xi-api-key": settings.elevenlabs_api_key.strip()}
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
            data = json.loads(r.read().decode("utf-8"))
        voices = {v["voice_id"]: v.get("name", "") for v in data.get("voices", []) if v.get("voice_id")}
    except Exception:
        voices = {}   # key lacks 'voices read' permission — fall back to premades only
    for vid, name in PREMADE.items():
        voices.setdefault(vid, name)
    return voices


def synthesize(
    text: str,
    out_path: Path,
    *,
    voice_id: str | None = None,
    model: str | None = None,
    stability: float = 0.4,
    style: float = 0.35,
) -> Path:
    """Render `text` to an mp3 with ElevenLabs. Raises on failure so callers can
    fall back to edge-tts."""
    if not is_configured():
        raise RuntimeError("ELEVENLABS_API_KEY is not set.")
    voice_id = (voice_id or settings.elevenlabs_voice_id).strip()
    model = model or settings.elevenlabs_model
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body = json.dumps({
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": 0.8,
            "style": style,
            "use_speaker_boost": True,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/text-to-speech/{voice_id}",
        data=body,
        headers={
            "xi-api-key": settings.elevenlabs_api_key.strip(),
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as r, open(out_path, "wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore")[:200] if hasattr(exc, "read") else ""
        raise RuntimeError(f"ElevenLabs error {exc.code}: {detail}") from exc
    return out_path
