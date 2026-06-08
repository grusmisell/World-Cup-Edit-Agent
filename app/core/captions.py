"""Generate burned-in caption files (ASS) from word-level timestamps."""
from __future__ import annotations

from pathlib import Path

from .transcriber import Word

# Reference frame the style presets are tuned for; margins scale off this height.
REF_H = 1920

# ASS colours are &HAABBGGRR (alpha, blue, green, red). 00 alpha = opaque.
# Each preset fills the Style line fields that differ between looks.
STYLES: dict[str, dict] = {
    "classic": {  # white text, thick black outline, lower third
        "fontname": "Arial", "fontsize": 96, "primary": "&H00FFFFFF",
        "outline_col": "&H00000000", "back": "&H64000000", "bold": -1,
        "borderstyle": 1, "outline": 6, "shadow": 3, "alignment": 2, "marginv": 360,
    },
    "bold_yellow": {  # punchy yellow text, black outline, lower third
        "fontname": "Arial", "fontsize": 104, "primary": "&H0000FFFF",
        "outline_col": "&H00000000", "back": "&H64000000", "bold": -1,
        "borderstyle": 1, "outline": 7, "shadow": 2, "alignment": 2, "marginv": 360,
    },
    "boxed": {  # white text on a semi-transparent black box
        "fontname": "Arial", "fontsize": 92, "primary": "&H00FFFFFF",
        "outline_col": "&H00000000", "back": "&H80000000", "bold": -1,
        "borderstyle": 3, "outline": 6, "shadow": 0, "alignment": 2, "marginv": 360,
    },
    "centered": {  # white text, outline, centered in the middle of the frame
        "fontname": "Arial", "fontsize": 100, "primary": "&H00FFFFFF",
        "outline_col": "&H00000000", "back": "&H64000000", "bold": -1,
        "borderstyle": 1, "outline": 6, "shadow": 3, "alignment": 5, "marginv": 0,
    },
}
DEFAULT_STYLE = "classic"

# Top-pinned headline (the script title) used in voiceover mode for retention.
# alignment 8 = top-center; marginv here is distance from the TOP.
HEADLINE = {
    "fontname": "Arial", "fontsize": 62, "primary": "&H00FFFFFF",
    "outline_col": "&H00000000", "back": "&HA0000000", "bold": -1,
    "borderstyle": 3, "outline": 4, "shadow": 0, "alignment": 8, "marginv": 110,
}


def _fmt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("{", "(").replace("}", ")").strip()


def _style_line(p: dict, marginv: int, name: str = "Pop") -> str:
    return (
        f"Style: {name},"
        f"{p['fontname']},{p['fontsize']},{p['primary']},&H000000FF,"
        f"{p['outline_col']},{p['back']},{p['bold']},0,0,0,100,100,0,0,"
        f"{p['borderstyle']},{p['outline']},{p['shadow']},{p['alignment']},80,80,{marginv},1"
    )


def _chunk_words(
    words: list[Word],
    max_words: int = 3,
    max_chars: int = 18,
    max_gap: float = 0.7,
) -> list[tuple[float, float, str]]:
    chunks: list[tuple[float, float, str]] = []
    cur: list[Word] = []

    def flush():
        if not cur:
            return
        text = " ".join(w.text.strip() for w in cur).strip()
        chunks.append((cur[0].start, cur[-1].end, text))

    for w in words:
        if not w.text.strip():
            continue
        if cur:
            prospective = " ".join(x.text.strip() for x in cur + [w])
            gap = w.start - cur[-1].end
            if len(cur) >= max_words or len(prospective) > max_chars or gap > max_gap:
                flush()
                cur = []
        cur.append(w)
    flush()
    return chunks


def build_ass(
    words: list[Word],
    clip_start: float,
    clip_end: float,
    out_path: Path,
    *,
    style: str = DEFAULT_STYLE,
    play_w: int = 1080,
    play_h: int = 1920,
    marginv: int | None = None,
    headline: str | None = None,
) -> Path:
    """Write an ASS file with captions for the words inside [clip_start, clip_end],
    timed relative to the start of the cut clip (0-based). The ASS PlayRes matches
    the output dimensions so positions map 1:1; the vertical margin scales with
    output height so captions sit in the lower third for any aspect ratio. Pass an
    explicit `marginv` (px from the bottom) to override placement (e.g. split mode)."""
    preset = STYLES.get(style, STYLES[DEFAULT_STYLE])
    if marginv is None:
        marginv = round(preset["marginv"] * play_h / REF_H)
    inside = [
        Word(w.text, w.start - clip_start, w.end - clip_start)
        for w in words
        if w.end > clip_start and w.start < clip_end
    ]
    chunks = _chunk_words(inside)

    style_lines = [_style_line(preset, marginv)]
    headline = (headline or "").strip()
    if headline:
        head_marginv = round(HEADLINE["marginv"] * play_h / REF_H)
        style_lines.append(_style_line(HEADLINE, head_marginv, name="Head"))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(style_lines)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    if headline:
        span_end = max(0.1, clip_end - clip_start)
        lines.append(
            f"Dialogue: 0,{_fmt_time(0.0)},{_fmt_time(span_end)},Head,,60,60,0,,"
            f"{_escape(headline).upper()}"
        )
    for start, end, text in chunks:
        if end <= 0 or start >= (clip_end - clip_start):
            continue
        start = max(0.0, start)
        text = _escape(text).upper()
        lines.append(
            f"Dialogue: 0,{_fmt_time(start)},{_fmt_time(end)},Pop,,0,0,0,,{text}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# Big lower-third player/country name tag for football edits (alignment 2 = bottom).
NAME_TAG = {
    "fontname": "Arial", "fontsize": 140, "primary": "&H00FFFFFF",
    "outline_col": "&H00000000", "back": "&HB4000000", "bold": -1,
    "borderstyle": 1, "outline": 8, "shadow": 4, "alignment": 2, "marginv": 250,
}
# Small top title strip for football edits (alignment 8 = top-center).
EDIT_TITLE = {
    "fontname": "Arial", "fontsize": 58, "primary": "&H0000F0FF",  # warm yellow
    "outline_col": "&H00000000", "back": "&HA0000000", "bold": -1,
    "borderstyle": 3, "outline": 4, "shadow": 0, "alignment": 8, "marginv": 120,
}


def build_edit_ass(
    out_path: Path,
    *,
    duration: float,
    play_w: int = 1080,
    play_h: int = 1920,
    title: str = "",
    name_tag: str = "",
) -> Path:
    """Write a standalone ASS with a top title strip and a big bottom name tag,
    both shown for the whole clip — the text layer for football edits (no word
    timing needed). Returns out_path."""
    span = max(0.1, duration)
    name_mv = round(NAME_TAG["marginv"] * play_h / REF_H)
    title_mv = round(EDIT_TITLE["marginv"] * play_h / REF_H)
    styles = [
        _style_line(NAME_TAG, name_mv, name="Name"),
        _style_line(EDIT_TITLE, title_mv, name="EditTitle"),
    ]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    end = _fmt_time(span)
    if title.strip():
        lines.append(
            f"Dialogue: 0,{_fmt_time(0.0)},{end},EditTitle,,40,40,0,,{_escape(title).upper()}"
        )
    if name_tag.strip():
        lines.append(
            f"Dialogue: 0,{_fmt_time(0.0)},{end},Name,,40,40,0,,{_escape(name_tag).upper()}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
