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


_GOOD_KW = ("goals", "skills", "highlights", "best", "magic", "dribbl", "assist", "edit")
_BAD_KW = ("reaction", "react", "interview", "press", "podcast", "talk", "news ", "fifa 2", "efootball", "pes 2", "gameplay")


def _pick_video(query: str) -> dict | None:
    """Flat-search YouTube and score results to pick a real highlight clip — prefer
    goals/skills/highlights titles and a sane duration, avoid reactions/gameplay/Shorts."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as y:
            info = y.extract_info(f"ytsearch6:{query}", download=False)
    except Exception:
        return None
    entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
    if not entries:
        return None

    def score(e: dict) -> int:
        title = (e.get("title") or "").lower()
        dur = e.get("duration") or 0
        s = 0
        s += 2 * sum(kw in title for kw in _GOOD_KW)
        s -= 3 * sum(kw in title for kw in _BAD_KW)
        if 45 <= dur <= 1200:
            s += 3
        elif dur and dur < 25:      # likely a Short / intro card
            s -= 4
        elif dur and dur > 2400:    # very long comp — section may miss the action
            s -= 1
        return s

    entries.sort(key=score, reverse=True)
    return entries[0]


def download_clip(
    query: str, out_dir: Path, name: str, *, length: float = 12.0,
) -> Path | None:
    """Search YouTube for `query`, pick a good highlight video, and download just a
    ~`length`s section (past the intro) as out_dir/<name>.mp4. Uses yt-dlp's partial
    download (needs ffmpeg ON PATH, so we add it). Returns the path, or None on
    failure — callers should handle a missing clip gracefully."""
    import os
    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    # yt-dlp's range downloader (FFmpegFD) looks for ffmpeg on PATH, not just via
    # ffmpeg_location, so make sure our bundled ffmpeg is discoverable there.
    os.environ["PATH"] = ffmpeg_utils.ffmpeg_dir() + os.pathsep + os.environ.get("PATH", "")

    pick = _pick_video(query)
    if pick:
        dur = pick.get("duration") or 0
        # Start a bit in to skip intros/title cards; stay clear of the very end.
        start = min(max(12.0, dur * 0.12), max(12.0, dur - length - 2)) if dur else 15.0
        target = f"https://www.youtube.com/watch?v={pick['id']}"
    else:
        start, target = 15.0, f"ytsearch1:{query}"

    outtmpl = str(out_dir / f"{name}.%(ext)s")
    opts = {
        # Prefer a 1080p VIDEO-only stream (these clips are only used as visuals —
        # the voiceover/music drive the audio — so no merge needed, and higher res
        # means far less upscaling blur when cropped to 9:16).
        "format": (
            "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=1080]/"
            "best[height<=1080][ext=mp4]/best[height<=1080]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True, "quiet": True, "no_warnings": True, "restrictfilenames": True,
        "ffmpeg_location": ffmpeg_utils.ffmpeg_dir(),
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start, start + length)]),
        "force_keyframes_at_cuts": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(target, download=True)
    except Exception:
        return None
    cands = sorted(out_dir.glob(f"{name}.*"))
    return next((c for c in cands if c.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}), None)


def use_local_file(src: Path, out_dir: Path) -> tuple[Path, str]:
    """Copy an already-present local file into the job dir. Returns (path, title)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"source{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest, src.stem
