from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"  # football stock loops for news mode
MUSIC_DIR = DATA_DIR / "music"              # royalty-free music beds (edit/news mode)
IMAGES_DIR = DATA_DIR / "images"            # player still photos for image-edit mode

for _d in (DATA_DIR, UPLOADS_DIR, JOBS_DIR, BACKGROUNDS_DIR, MUSIC_DIR, IMAGES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Reuse the sibling Clipping Agent's bundled whisper.cpp + ffmpeg (~235 MB) instead
# of duplicating it on this near-full disk. Override with SHARED_BIN_DIR in .env.
_DEFAULT_SHARED_BIN = BASE_DIR.parent / "Clipping Agent" / "bin"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    whisper_model: str = "base"
    # Directory holding bin/whisper + bin/models + a staged ffmpeg.exe. Defaults to
    # the sibling Clipping Agent install so we don't re-download the 147 MB model.
    shared_bin_dir: str = str(_DEFAULT_SHARED_BIN)

    default_clip_count: int = 6
    min_clip_seconds: int = 10
    max_clip_seconds: int = 60
    default_caption_style: str = "bold_yellow"
    default_aspect: str = "9:16"
    default_reframe: str = "fill"          # football edits crop tight on the action
    default_edit_style: str = "football_edit"  # football_edit | clean
    default_niche: str = "2026 FIFA World Cup football"
    default_gen_captions: bool = True

    # News / voiceover mode: optional music bed mixed under the narration.
    default_music: str = ""           # filename in data/music/, "" = no music
    music_volume: float = 0.22        # 0..1, music gain relative to narration
    voiceover_headline: bool = True   # pin the script title at the top of the video

    # Football edit (highlights / image) mode: music bed driving the beat-sync.
    edit_music_volume: float = 0.85   # edits are music-forward (under any commentary)
    edit_zoom_punch: float = 0.07     # how hard the beat zoom pulses (0 = none)

    host: str = "127.0.0.1"
    port: int = 8001                  # 8000 is the Clipping Agent; run both at once

    # Privacy for YouTube auto-uploads: private | unlisted | public.
    youtube_privacy: str = "private"

    # Pexels (free stock video/photos) — copyright-safe football B-roll for news mode.
    pexels_api_key: str = ""

    # ElevenLabs (optional, paid) — far more human voiceover. When the key is set it
    # replaces edge-tts. Pick any voice from your ElevenLabs Voice Library (licensed
    # professional voices). Default = "Adam" (deep narrator). NOTE: cloning a real
    # person's voice without their consent violates ElevenLabs' ToS — use the library.
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"   # premade "Adam" (deep male)
    elevenlabs_model: str = "eleven_multilingual_v2"

    @property
    def shared_bin(self) -> Path:
        return Path(self.shared_bin_dir)


settings = Settings()
