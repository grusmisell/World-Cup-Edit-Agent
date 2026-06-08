"""Locate and run ffmpeg / ffprobe.

System ffmpeg is preferred (it ships ffprobe too). If it is not on PATH we fall
back to the ffmpeg binary bundled with the imageio-ffmpeg package. ffprobe is not
bundled, so when we only have the imageio binary we read media info via ffmpeg's
own stderr output instead.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Reuse the ffmpeg.exe already staged in the shared bin dir (sibling Clipping
    # Agent) rather than copying the ~80 MB imageio binary again on a near-full disk.
    try:
        from ..config import settings

        staged = settings.shared_bin / "ffmpeg.exe"
        if staged.exists():
            return str(staged)
    except Exception:  # pragma: no cover
        pass
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise FFmpegError(
            "ffmpeg not found. Install it (winget install ffmpeg) or "
            "`pip install imageio-ffmpeg`."
        ) from exc


@lru_cache(maxsize=1)
def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


@lru_cache(maxsize=1)
def ffmpeg_dir() -> str:
    """A directory containing an executable named ffmpeg(.exe).

    Tools like yt-dlp locate ffmpeg by conventional name, but the imageio-bundled
    binary has a versioned filename. Stage a copy named ffmpeg.exe in bin/ so
    those tools can find it.
    """
    p = Path(ffmpeg_path())
    if p.stem.lower() == "ffmpeg":
        return str(p.parent)

    from ..config import BASE_DIR

    dest_dir = BASE_DIR / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / ("ffmpeg.exe" if p.suffix else "ffmpeg")
    if not dest.exists() or dest.stat().st_size != p.stat().st_size:
        shutil.copy2(p, dest)
    return str(dest_dir)


def run(
    args: list[str],
    *,
    capture: bool = True,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg command. `args` excludes the ffmpeg binary itself."""
    cmd = [ffmpeg_path(), "-hide_banner", "-y", *args]
    proc = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-2000:]
        raise FFmpegError(f"ffmpeg failed ({proc.returncode}):\n{tail}")
    return proc


def probe_duration(path: str | Path) -> float:
    """Return media duration in seconds."""
    path = str(path)
    fp = ffprobe_path()
    if fp:
        out = subprocess.run(
            [
                fp,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
        )
        try:
            data = json.loads(out.stdout)
            return float(data["format"]["duration"])
        except Exception:
            pass

    # Fallback: parse ffmpeg stderr ("Duration: HH:MM:SS.ms").
    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
    )
    text = proc.stderr or ""
    marker = "Duration:"
    if marker in text:
        chunk = text.split(marker, 1)[1].split(",", 1)[0].strip()
        try:
            h, m, s = chunk.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            pass
    return 0.0


def probe_resolution(path: str | Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    path = str(path)
    fp = ffprobe_path()
    if fp:
        out = subprocess.run(
            [
                fp,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
        )
        try:
            data = json.loads(out.stdout)
            stream = data["streams"][0]
            return int(stream["width"]), int(stream["height"])
        except Exception:
            pass

    proc = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", path],
        capture_output=True,
        text=True,
    )
    text = proc.stderr or ""
    import re

    m = re.search(r"Video:.*?\b(\d{2,5})x(\d{2,5})\b", text)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise FFmpegError(f"Could not determine video resolution for {path}")
