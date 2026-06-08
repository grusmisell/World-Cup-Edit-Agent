"""Cut, reframe (face-aware) to a target aspect ratio, and burn captions onto clips."""
from __future__ import annotations

from pathlib import Path

from . import ffmpeg_utils
from . import beats as beats_mod
from .transcriber import Word
from .captions import build_ass, build_edit_ass

# Output dimensions per aspect ratio. "original" keeps the source framing.
ASPECTS: dict[str, tuple[int, int] | None] = {
    "9:16": (1080, 1920),
    "4:5": (1080, 1350),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
    "original": None,
}
DEFAULT_ASPECT = "9:16"

# How to fit the source into the target aspect:
#   fill = face-aware crop to fill (zooms in, can cut edges) — good for talking heads
#   fit  = scale whole frame to fit + blurred enlarged copy fills the rest — never crops
REFRAMES = {"fit", "fill"}
DEFAULT_REFRAME = "fit"

# Split-screen ("brainrot") layout: clip on top, looping background below.
SPLIT_W, SPLIT_H = 1080, 1920
SPLIT_TOP_H = 1152          # ~60% for the clip
SPLIT_BOT_H = SPLIT_H - SPLIT_TOP_H  # ~40% for the background

_ENCODE = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
    # Standard web/social audio: 48 kHz stereo AAC. (edge-tts narration is 24 kHz
    # mono; some browsers grey out the volume control on that, so always resample.)
    "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
    "-movflags", "+faststart",
]


def _face_center(
    video_path: Path, start: float, end: float, samples: int = 12
) -> tuple[float, float] | None:
    """Return median (x, y) of detected faces across sampled frames, or None."""
    try:
        import cv2
    except Exception:
        return None

    cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_file)
    if detector.empty():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    xs: list[float] = []
    ys: list[float] = []
    duration = max(0.1, end - start)
    try:
        for i in range(samples):
            t = start + duration * (i + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(60, 60))
            if len(faces) == 0:
                continue
            # Pick the largest face in the frame.
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            xs.append(fx + fw / 2.0)
            ys.append(fy + fh / 2.0)
    finally:
        cap.release()

    if not xs:
        return None
    xs.sort()
    ys.sort()
    return xs[len(xs) // 2], ys[len(ys) // 2]


def _compute_crop(
    w: int, h: int, cx: float, cy: float, target_ar: float
) -> tuple[int, int, int, int]:
    src_ar = w / h
    if src_ar > target_ar:
        # Source wider than target -> crop width, keep full height.
        cw = int(round(h * target_ar))
        cw -= cw % 2
        ch = h - (h % 2)
        x = int(round(cx - cw / 2))
        x = max(0, min(x, w - cw))
        y = 0
    else:
        # Source taller/narrower -> crop height, keep full width.
        ch = int(round(w / target_ar))
        ch -= ch % 2
        cw = w - (w % 2)
        y = int(round(cy - ch / 2))
        y = max(0, min(y, h - ch))
        x = 0
    return cw, ch, x, y


def _render_split(
    source: Path, out_path: Path, *, start: float, duration: float,
    words: list[Word], burn_captions: bool, caption_style: str, background: Path,
) -> Path:
    """Split-screen: clip (fit + blurred fill) on top, looping background below."""
    out_dir = out_path.parent
    background = Path(background).resolve()

    ass_name = None
    if burn_captions and words:
        ass_path = out_dir / f"{out_path.stem}.ass"
        build_ass(words, start, start + duration, ass_path, style=caption_style,
                  play_w=SPLIT_W, play_h=SPLIT_H, marginv=SPLIT_BOT_H + 60)
        ass_name = ass_path.name

    graph = (
        f"[0:v]split=2[ca][cb];"
        f"[ca]scale={SPLIT_W}:{SPLIT_TOP_H}:force_original_aspect_ratio=increase,"
        f"crop={SPLIT_W}:{SPLIT_TOP_H},gblur=sigma=20[topbg];"
        f"[cb]scale={SPLIT_W}:{SPLIT_TOP_H}:force_original_aspect_ratio=decrease[topfg];"
        f"[topbg][topfg]overlay=(W-w)/2:(H-h)/2[top];"
        f"[1:v]scale={SPLIT_W}:{SPLIT_BOT_H}:force_original_aspect_ratio=increase,"
        f"crop={SPLIT_W}:{SPLIT_BOT_H}[bot];"
        f"[top][bot]vstack=inputs=2,setsar=1"
    )
    if ass_name:
        graph += f",ass={ass_name}"
    graph += "[outv]"

    args = [
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-stream_loop", "-1", "-i", str(background),
        "-filter_complex", graph, "-map", "[outv]", "-map", "0:a?", "-shortest",
        *_ENCODE, out_path.name,
    ]
    ffmpeg_utils.run(args, cwd=out_dir)
    return out_path


def render_voiceover(
    background: Path,
    audio: Path,
    out_path: Path,
    *,
    words: list[Word],
    caption_style: str = "classic",
    headline: str | None = None,
    music: str | Path | None = None,
    music_volume: float = 0.18,
) -> Path:
    """AI-voiceover mode: full-screen looping background + VO audio + synced
    burned captions, bounded to the voiceover's length. Optionally pins a
    `headline` at the top and mixes a `music` bed (auto-ducked under the
    narration). Returns out_path."""
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    background = Path(background).resolve()
    audio = Path(audio).resolve()
    dur = max(0.5, ffmpeg_utils.probe_duration(audio))

    ass_name = None
    if words:
        ass_path = out_dir / f"{out_path.stem}.ass"
        build_ass(words, 0.0, dur, ass_path, style=caption_style,
                  play_w=SPLIT_W, play_h=SPLIT_H, headline=headline)
        ass_name = ass_path.name

    graph = (
        f"[0:v]scale={SPLIT_W}:{SPLIT_H}:force_original_aspect_ratio=increase,"
        f"crop={SPLIT_W}:{SPLIT_H},setsar=1"
    )
    if ass_name:
        graph += f",ass={ass_name}"
    graph += "[outv]"

    music_path = Path(music).resolve() if music else None
    if music_path and music_path.exists():
        vol = max(0.0, min(1.0, music_volume))
        # Narration is input 1; looping music is input 2. Duck the music under
        # the voice (sidechaincompress keyed by the narration), then mix.
        graph += (
            f";[2:a]volume={vol:.3f}[mq];"
            f"[mq][1:a]sidechaincompress=threshold=0.05:ratio=6:attack=20:release=400[mduck];"
            f"[1:a][mduck]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        args = [
            "-stream_loop", "-1", "-i", str(background),
            "-i", str(audio),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", graph, "-map", "[outv]", "-map", "[aout]",
            "-t", f"{dur:.3f}", *_ENCODE, out_path.name,
        ]
    else:
        args = [
            "-stream_loop", "-1", "-i", str(background),
            "-i", str(audio),
            "-filter_complex", graph, "-map", "[outv]", "-map", "1:a",
            "-t", f"{dur:.3f}", *_ENCODE, out_path.name,
        ]
    ffmpeg_utils.run(args, cwd=out_dir)
    return out_path


def render_clip(
    source: Path,
    out_path: Path,
    *,
    start: float,
    end: float,
    words: list[Word],
    burn_captions: bool = True,
    aspect: str = DEFAULT_ASPECT,
    reframe: str = DEFAULT_REFRAME,
    caption_style: str = "classic",
    background: str | Path | None = None,
    src_w: int,
    src_h: int,
) -> Path:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, end - start)
    # ffmpeg runs with cwd=out_dir (so the .ass file resolves by name), so the
    # source input must be absolute or it won't be found.
    source = Path(source).resolve()

    if background:
        return _render_split(
            source, out_path, start=start, duration=duration, words=words,
            burn_captions=burn_captions, caption_style=caption_style,
            background=background,
        )

    target = ASPECTS.get(aspect, ASPECTS[DEFAULT_ASPECT])
    if target is not None:
        tw, th = target
        play_w, play_h = tw, th
    else:
        tw = th = None
        play_w, play_h = src_w - (src_w % 2), src_h - (src_h % 2)

    ass_name = None
    if burn_captions and words:
        ass_path = out_dir / f"{out_path.stem}.ass"
        build_ass(words, start, end, ass_path, style=caption_style,
                  play_w=play_w, play_h=play_h)
        ass_name = ass_path.name  # referenced relative to cwd=out_dir

    vf = None
    filter_complex = None
    if target is None:
        # No reframe — keep the source framing.
        vf = f"ass={ass_name}" if ass_name else None
    elif reframe == "fill":
        # Face-aware crop to fill the target (zooms in; may cut edges).
        center = _face_center(source, start, end)
        cx, cy = center if center is not None else (src_w / 2.0, src_h / 2.0)
        cw, ch, x, y = _compute_crop(src_w, src_h, cx, cy, tw / th)
        chain = [f"crop={cw}:{ch}:{x}:{y}", f"scale={tw}:{th}", "setsar=1"]
        if ass_name:
            chain.append(f"ass={ass_name}")
        vf = ",".join(chain)
    else:
        # "fit": whole frame scaled to fit, blurred enlarged copy fills the rest.
        graph = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th},gblur=sigma=20[bg2];"
            f"[fg]scale={tw}:{th}:force_original_aspect_ratio=decrease[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,setsar=1"
        )
        if ass_name:
            graph += f",ass={ass_name}"
        filter_complex = graph + "[outv]"

    args = [
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source),
    ]
    if filter_complex:
        args += ["-filter_complex", filter_complex, "-map", "[outv]", "-map", "0:a?"]
    elif vf:
        args += ["-vf", vf]
    args += [*_ENCODE, out_path.name]

    ffmpeg_utils.run(args, cwd=out_dir)
    return out_path


# --- Football "cool edit" rendering ----------------------------------------
# A teal/orange cinematic grade + a tempo-synced zoom punch + bold name/title
# text — the CapCut-style football-edit look. Used for highlight video clips and
# for still-image montages.

FPS = 30
EDIT_W, EDIT_H = 1080, 1920

# Cinematic teal-shadow / warm-highlight grade for that "edit" colour pop.
_GRADE = (
    "eq=contrast=1.12:saturation=1.38:brightness=0.015:gamma=0.97,"
    "colorbalance=rs=-0.05:bs=0.06:rm=0.02:bm=-0.02:rh=0.07:gh=0.02:bh=-0.06,"
    "vignette=PI/5"
)


def _reframe_chain(reframe: str, w: int, h: int) -> str:
    """Filter chain that fits the source into w x h (no captions)."""
    if reframe == "fit":
        return (
            f"split=2[bg][fg];"
            f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},gblur=sigma=20[bg2];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fg2];"
            f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
        )
    # fill (default): cover + centre crop (tight on the action).
    return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"


def _zoom_punch(grid: "beats_mod.BeatGrid | None", punch: float) -> str:
    """A zoompan filter that pulses the zoom on the beat (tempo-synced)."""
    period = grid.period if grid else 0.5
    phase = grid.phase if grid else 0.0
    punch = max(0.0, min(0.25, punch))
    # Sharp peak at phase + k*period via a raised cosine; single-quoted so the
    # commas inside max()/pow() don't split the filtergraph.
    z = (
        f"z='1.0+{punch:.3f}*pow(max(0,cos(2*PI*(on/{FPS} - {phase:.3f})/{period:.3f})),3)'"
    )
    return (
        f"zoompan={z}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={EDIT_W}x{EDIT_H}:fps={FPS}"
    )


def _edit_audio_args(
    have_orig_audio: bool, music: Path | None, music_volume: float, total: float
):
    """Build the audio inputs + filter + map for an edit. Returns
    (extra_inputs, audio_filter_or_None, audio_map). Music is mixed loud over a
    ducked original commentary track (or stands alone if there's no source audio)."""
    vol = max(0.0, min(1.5, music_volume))
    if music and music.exists():
        extra = ["-stream_loop", "-1", "-i", str(music.resolve())]
        if have_orig_audio:
            af = (
                f"[0:a]volume=0.55[oa];"
                f"[1:a]volume={vol:.2f}[ma];"
                f"[oa][ma]amix=inputs=2:duration=first:normalize=0[aout]"
            )
            return extra, af, "[aout]"
        # No commentary — music only, faded.
        af = f"[1:a]volume={vol:.2f},afade=t=out:st={max(0.0, total - 0.4):.2f}:d=0.4[aout]"
        return extra, af, "[aout]"
    if have_orig_audio:
        return [], None, "0:a?"
    return [], None, None


def render_football_edit(
    source: Path,
    out_path: Path,
    *,
    start: float,
    end: float,
    src_w: int,
    src_h: int,
    reframe: str = "fill",
    name_tag: str = "",
    title: str = "",
    music: str | Path | None = None,
    music_volume: float = 0.85,
    zoom_punch: float = 0.07,
    grid: "beats_mod.BeatGrid | None" = None,
    has_audio: bool = True,
) -> Path:
    """Render a highlight segment [start, end] as a beat-synced football edit:
    9:16 reframe -> cinematic grade -> tempo-synced zoom punch -> bold name/title
    text -> music over ducked commentary. Returns out_path."""
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source).resolve()
    duration = max(0.5, end - start)
    music_path = Path(music).resolve() if music else None

    # Beat grid from the music (preferred) or the clip's own audio, for the pulse.
    if grid is None and zoom_punch > 0:
        ref = music_path if (music_path and music_path.exists()) else source
        try:
            grid = beats_mod.detect(ref, max_seconds=duration + 4)
        except Exception:
            grid = None

    ass_path = out_dir / f"{out_path.stem}.ass"
    build_edit_ass(
        ass_path, duration=duration, play_w=EDIT_W, play_h=EDIT_H,
        title=title, name_tag=name_tag,
    )

    vchain = f"[0:v]{_reframe_chain(reframe, EDIT_W, EDIT_H)},setsar=1,{_GRADE}"
    if zoom_punch > 0:
        vchain += f",{_zoom_punch(grid, zoom_punch)}"
    vchain += f",ass={ass_path.name}[outv]"

    extra_in, af, amap = _edit_audio_args(has_audio, music_path, music_volume, duration)
    graph = vchain + (";" + af if af else "")

    args = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source), *extra_in,
            "-filter_complex", graph, "-map", "[outv]"]
    if amap:
        args += ["-map", amap]
    args += ["-t", f"{duration:.3f}", *_ENCODE, out_path.name]
    ffmpeg_utils.run(args, cwd=out_dir)
    return out_path


def render_image_edit(
    images: list[Path],
    out_path: Path,
    *,
    name_tag: str = "",
    title: str = "",
    music: str | Path | None = None,
    music_volume: float = 0.9,
    seconds_per_image: float = 2.2,
    grid: "beats_mod.BeatGrid | None" = None,
) -> Path:
    """Render a still-image montage as a beat-synced football edit: each photo gets
    an alternating Ken Burns push/pull, cuts land on the beat, with the cinematic
    grade + name/title text + music. Returns out_path."""
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [Path(p).resolve() for p in images if Path(p).exists()]
    if not imgs:
        raise RuntimeError("No images provided for the image edit.")
    music_path = Path(music).resolve() if music else None

    if grid is None and music_path and music_path.exists():
        try:
            grid = beats_mod.detect(music_path)
        except Exception:
            grid = None

    rough_total = max(seconds_per_image, seconds_per_image * len(imgs))
    cuts = beats_mod.cut_points(
        grid or beats_mod.BeatGrid([], 0.5, 120.0), rough_total, n=len(imgs),
        min_hold=max(1.0, seconds_per_image * 0.7),
    )
    durs = [round(cuts[i + 1] - cuts[i], 3) for i in range(len(imgs))]
    durs = [max(0.8, d) for d in durs]
    total = round(sum(durs), 3)

    inputs: list[str] = []
    for img in imgs:
        inputs += ["-i", str(img)]

    seg_filters = []
    for i, d in enumerate(durs):
        frames = max(2, int(round(d * FPS)))
        if i % 2 == 0:  # push in
            zexpr = "z='min(zoom+0.0011,1.18)'"
        else:           # start pushed-in, pull out
            zexpr = f"z='if(eq(on,0),1.18,max(zoom-0.0011,1.0))'"
        seg_filters.append(
            f"[{i}:v]scale={EDIT_W}:{EDIT_H}:force_original_aspect_ratio=increase,"
            f"crop={EDIT_W}:{EDIT_H},setsar=1,"
            f"zoompan={zexpr}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={EDIT_W}x{EDIT_H}:fps={FPS}[v{i}]"
        )
    concat_in = "".join(f"[v{i}]" for i in range(len(imgs)))
    graph = ";".join(seg_filters)
    graph += f";{concat_in}concat=n={len(imgs)}:v=1:a=0,{_GRADE}"

    ass_path = out_dir / f"{out_path.stem}.ass"
    build_edit_ass(ass_path, duration=total, play_w=EDIT_W, play_h=EDIT_H,
                   title=title, name_tag=name_tag)
    graph += f",ass={ass_path.name}[outv]"

    args = list(inputs)
    amap = None
    if music_path and music_path.exists():
        args += ["-stream_loop", "-1", "-i", str(music_path)]
        vol = max(0.0, min(1.5, music_volume))
        graph += (
            f";[{len(imgs)}:a]volume={vol:.2f},"
            f"afade=t=out:st={max(0.0, total - 0.5):.2f}:d=0.5[aout]"
        )
        amap = "[aout]"

    args += ["-filter_complex", graph, "-map", "[outv]"]
    if amap:
        args += ["-map", amap]
    args += ["-t", f"{total:.3f}", *_ENCODE, out_path.name]
    ffmpeg_utils.run(args, cwd=out_dir)
    return out_path
