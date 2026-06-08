"""Lightweight beat / onset detection for music-synced edits.

No librosa / native ML runtime (those crash on this machine's Python 3.14). We
decode the audio to mono PCM with the bundled ffmpeg and run a small numpy
energy-flux onset detector. Good enough to drive beat-synced zoom punches and
image cuts in the football-edit render path.

Returns onset times (seconds) plus an estimated tempo period so the renderer can
pulse a zoom even when exact onsets are sparse.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg_utils

SR = 22050
HOP = 512


@dataclass
class BeatGrid:
    onsets: list[float]   # detected onset times in seconds
    period: float         # median seconds-per-beat (tempo), clamped to a sane band
    bpm: float

    @property
    def phase(self) -> float:
        """Time of the first strong beat, used to phase-align a periodic pulse."""
        return self.onsets[0] if self.onsets else 0.0


def _decode_pcm(audio_path: Path):
    """Decode `audio_path` to a mono float32 numpy array at SR. Empty on failure."""
    import numpy as np

    cmd = [
        ffmpeg_utils.ffmpeg_path(), "-v", "error", "-i", str(audio_path),
        "-ac", "1", "-ar", str(SR), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def detect(audio_path: str | Path, *, max_seconds: float | None = None) -> BeatGrid:
    """Detect onset times and tempo for `audio_path`. Always returns a usable
    BeatGrid (falls back to a 120 BPM grid if detection is weak/unavailable)."""
    import numpy as np

    fallback = BeatGrid(onsets=[], period=0.5, bpm=120.0)
    try:
        y = _decode_pcm(Path(audio_path))
    except Exception:
        return fallback
    if y.size < SR // 2:
        return fallback
    if max_seconds:
        y = y[: int(max_seconds * SR)]

    # Frame energy envelope, then half-wave-rectified flux = onset strength.
    n_frames = 1 + (len(y) - 1) // HOP
    energy = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = y[i * HOP : i * HOP + HOP]
        energy[i] = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
    flux = np.diff(energy, prepend=energy[:1])
    flux[flux < 0] = 0.0
    if flux.max() <= 1e-6:
        return fallback
    flux = flux / flux.max()

    fps = SR / HOP
    # Adaptive peak-pick: a frame is an onset if it is a local max and above a
    # moving-average threshold, with a refractory gap so we don't double-trigger.
    win = max(1, int(0.10 * fps))
    thr = np.convolve(flux, np.ones(2 * win + 1) / (2 * win + 1), mode="same") + 0.06
    min_gap = int(0.18 * fps)
    onsets: list[float] = []
    last = -min_gap
    for i in range(1, n_frames - 1):
        if (
            flux[i] > thr[i]
            and flux[i] >= flux[i - 1]
            and flux[i] > flux[i + 1]
            and i - last >= min_gap
        ):
            onsets.append(round(i / fps, 3))
            last = i

    if len(onsets) < 2:
        return fallback

    # Tempo = median inter-onset interval, clamped to a musical band (75-200 BPM).
    diffs = np.diff(onsets)
    period = float(np.median(diffs))
    period = min(0.8, max(0.3, period))
    bpm = round(60.0 / period, 1)
    return BeatGrid(onsets=onsets, period=period, bpm=bpm)


def cut_points(grid: BeatGrid, total: float, *, n: int, min_hold: float = 0.6) -> list[float]:
    """Choose `n` beat-aligned cut times spanning [0, total] for an image montage.

    Prefers real onsets at least `min_hold` apart; falls back to an even tempo grid
    if there aren't enough. Always returns a strictly increasing list ending at
    `total` with n segments (n+1 boundaries)."""
    n = max(1, n)
    pts = [0.0]
    for o in grid.onsets:
        if o <= min_hold or o >= total - 0.2:
            continue
        if o - pts[-1] >= min_hold:
            pts.append(round(o, 3))
        if len(pts) >= n:
            break
    # Top up with an even grid if we ran short of usable onsets.
    while len(pts) < n:
        frac = len(pts) / n
        pts.append(round(frac * total, 3))
        pts = sorted(set(pts))
    pts = sorted(set(p for p in pts if p < total))[:n]
    pts.append(round(total, 3))
    return pts
