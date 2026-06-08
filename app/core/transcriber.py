"""Transcribe audio locally with the bundled whisper.cpp binary.

Native Python ML runtimes do not work on this machine's Python 3.14:
ctranslate2 (faster-whisper) segfaults and torch's c10.dll fails to initialize.
whisper.cpp is a standalone native binary with no Python ABI dependency, so we
shell out to it. We extract a 16 kHz mono WAV with our bundled ffmpeg, run
whisper-cli with full-JSON output, and parse its tokens into word timestamps.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg_utils
from ..config import settings

# whisper.cpp binary + ggml model live in the shared bin dir (the sibling Clipping
# Agent install by default — see config.shared_bin_dir) so we don't duplicate the
# 147 MB model on this near-full disk.
WHISPER_DIR = settings.shared_bin / "whisper" / "Release"
WHISPER_CLI = WHISPER_DIR / "whisper-cli.exe"
MODELS_DIR = settings.shared_bin / "models"

SAMPLE_RATE = 16000


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


def _model_path(model_size: str) -> Path:
    return MODELS_DIR / f"ggml-{model_size}.bin"


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    audio_path = out_dir / "audio.wav"
    ffmpeg_utils.run(
        [
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(audio_path),
        ]
    )
    return audio_path


def _tokens_to_words(tokens: list[dict]) -> list[Word]:
    """Group whisper.cpp subword tokens into words.

    A token whose text starts with a space begins a new word; tokens without a
    leading space (subword pieces, punctuation) attach to the current word.
    Special tokens like ``[_BEG_]`` are skipped. Offsets are in milliseconds.
    """
    words: list[Word] = []
    text = ""
    start_ms: int | None = None
    end_ms: int | None = None

    def flush() -> None:
        nonlocal text, start_ms, end_ms
        t = text.strip()
        if t and start_ms is not None:
            words.append(Word(text=t, start=start_ms / 1000.0, end=(end_ms or start_ms) / 1000.0))
        text = ""
        start_ms = None
        end_ms = None

    for tok in tokens:
        raw = tok.get("text", "")
        if raw.startswith("[_"):
            continue
        off = tok.get("offsets") or {}
        frm = off.get("from")
        to = off.get("to")
        if frm is None or to is None:
            continue
        if raw.startswith(" ") and text:
            flush()
        if start_ms is None:
            start_ms = frm
        text += raw
        end_ms = to

    flush()
    return words


def transcribe(
    video_path: Path,
    out_dir: Path,
    *,
    model_size: str = "base",
    progress=None,
) -> Transcript:
    if not WHISPER_CLI.exists():
        raise RuntimeError(
            f"whisper.cpp binary not found at {WHISPER_CLI}. "
            "Re-download it (see README)."
        )
    model = _model_path(model_size)
    if not model.exists():
        raise RuntimeError(
            f"Whisper model not found: {model}. Download it from "
            f"https://huggingface.co/ggerganov/whisper.cpp (ggml-{model_size}.bin)."
        )

    audio_path = extract_audio(video_path, out_dir)
    out_base = out_dir / "whisper_out"

    cmd = [
        str(WHISPER_CLI),
        "-m",
        str(model),
        "-l",
        "auto",
        "-ojf",
        "-of",
        str(out_base),
        "-np",
        "-t",
        str(os.cpu_count() or 4),
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    json_path = out_base.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return _parse_whisper_json(data, progress=progress)


def _parse_whisper_json(data: dict, *, progress=None) -> Transcript:
    segments: list[Segment] = []
    for seg in data.get("transcription", []):
        off = seg.get("offsets") or {}
        start = float(off.get("from", 0)) / 1000.0
        end = float(off.get("to", 0)) / 1000.0
        words = _tokens_to_words(seg.get("tokens", []))
        segments.append(
            Segment(
                text=seg.get("text", "").strip(),
                start=start,
                end=end,
                words=words,
            )
        )
        if progress is not None:
            progress(end)

    language = (data.get("result") or {}).get("language", "")
    return Transcript(language=language, segments=segments)


def align_words(script_text: str, words: list[Word]) -> list[Word]:
    """When we KNOW the exact narration (e.g. a generated script), transfer the
    correct spelling onto whisper's word TIMINGS — so captions read 'Mbappe' not
    'Killian'. Aligns the script tokens to the (mis-spelled) whisper tokens with
    difflib; equal runs keep their timing 1:1, replaced runs distribute evenly.
    Falls back to the raw whisper words if anything looks off."""
    import difflib
    import re

    script_toks = (script_text or "").split()
    if not words or not script_toks:
        return words

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    w_norm = [norm(w.text) for w in words]
    s_norm = [norm(t) for t in script_toks]
    out: list[Word] = []
    sm = difflib.SequenceMatcher(a=w_norm, b=s_norm, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out.append(Word(script_toks[j1 + k], words[i1 + k].start, words[i1 + k].end))
        elif tag == "replace":
            wseg, sseg = words[i1:i2], script_toks[j1:j2]
            if not wseg or not sseg:
                continue
            start, end = wseg[0].start, wseg[-1].end
            span = max(0.04, (end - start) / len(sseg))
            for k, t in enumerate(sseg):
                out.append(Word(t, start + k * span, start + (k + 1) * span))
        elif tag == "insert":
            sseg = script_toks[j1:j2]
            anchor = words[i1].start if i1 < len(words) else (words[-1].end if words else 0.0)
            prev = out[-1].end if out else anchor
            span = max(0.05, (anchor - prev) / max(1, len(sseg)))
            for k, t in enumerate(sseg):
                out.append(Word(t, prev + k * span, prev + (k + 1) * span))
        # 'delete': whisper had extra tokens not in the script — drop them.
    return out or words


def load_transcript(out_dir: Path) -> Transcript | None:
    """Rebuild a Transcript from a previously-saved whisper_out.json (no re-run)."""
    json_path = out_dir / "whisper_out.json"
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return _parse_whisper_json(data)
