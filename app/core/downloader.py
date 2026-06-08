"""Acquire a source video either from a URL (yt-dlp) or a local file."""
from __future__ import annotations

import shutil
from pathlib import Path

from . import ffmpeg_utils


def download_from_url(url: str, out_dir: Path) -> tuple[Path, str]:
    """Download `url` into `out_dir`. Returns (video_path, title)."""
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(out_dir / "source.%(ext)s")

    ydl_opts = {
        # Prefer a single mp4 up to 1080p to keep processing fast and predictable.
        "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        # yt-dlp needs ffmpeg to merge separate video/audio streams. Point it at
        # our ffmpeg (system or the imageio-bundled binary), which isn't on PATH.
        "ffmpeg_location": ffmpeg_utils.ffmpeg_dir(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title") or "Untitled"

    # yt-dlp may produce source.mp4, source.mkv, source.webm, etc.
    candidates = sorted(out_dir.glob("source.*"))
    video = next((c for c in candidates if c.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}), None)
    if video is None:
        raise FileNotFoundError("Download finished but no video file was found.")
    return video, title


def use_local_file(src: Path, out_dir: Path) -> tuple[Path, str]:
    """Copy an already-present local file into the job dir. Returns (path, title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"source{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest, src.stem
