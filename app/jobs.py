"""Job orchestration: runs the World Cup edit pipelines in a background thread and
tracks progress. State is persisted to data/jobs/<id>/job.json so the UI can poll.

Modes (source_type):
  highlights — a match/highlights video (url or file) -> commentary transcript ->
               Claude picks the big moments -> football-edit or clean render.
  news       — a player/country/storyline -> Claude web-searches the latest news ->
               script -> edge-tts voice -> football B-roll -> synced captions.
  imageedit  — player still photos -> beat-synced Ken Burns montage edit.
"""
from __future__ import annotations

import json
import re
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    JOBS_DIR, UPLOADS_DIR, BACKGROUNDS_DIR, MUSIC_DIR, IMAGES_DIR, settings,
)
from .core import analyzer, clipper, downloader, transcriber, ffmpeg_utils, voiceover, football

# Pipeline stages with rough progress weights (must sum to ~1.0).
STAGES = [
    ("queued", 0.0),
    ("downloading", 0.10),
    ("scripting", 0.20),
    ("voicing", 0.40),
    ("transcribing", 0.45),
    ("analyzing", 0.55),
    ("captions", 0.68),
    ("clipping", 0.95),
    ("done", 1.0),
]


@dataclass
class ClipResult:
    index: int
    title: str
    hook: str
    score: int
    reason: str
    start: float
    end: float
    duration: float
    filename: str
    hashtags: list[str] = field(default_factory=list)
    name_tag: str = ""        # football edit on-screen label (player/country)
    subject: str = ""         # news mode: who/what the clip is about
    sources: list[str] = field(default_factory=list)  # news mode: cited source URLs
    edit_style: str = ""      # football_edit | clean | news | imageedit
    trend_score: int = 0
    matched_trend: str = ""
    trend_hashtags: list[str] = field(default_factory=list)
    captions: list[dict] = field(default_factory=list)
    mood: str = ""
    posted_to: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: str
    source_type: str          # highlights | news | imageedit
    source: str               # url / subject / "images"
    title: str = ""
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    message: str = ""
    error: str = ""
    clip_count: int = settings.default_clip_count
    burn_captions: bool = True
    aspect: str = settings.default_aspect
    reframe: str = settings.default_reframe
    edit_style: str = settings.default_edit_style   # football_edit | clean
    caption_style: str = settings.default_caption_style
    background: str = ""
    music: str = settings.default_music
    niche: str = settings.default_niche
    topic: str = ""           # news mode: subject(s), one per line
    voice: str = ""
    mood: str = ""            # "" = auto-mix, "manual", or a football.MOODS key
    news_seconds: int = 30    # news mode: target narration length per edit
    custom_script: str = ""   # news mode: user's own script(s), split on a '---' line
    images: list[str] = field(default_factory=list)  # imageedit mode: filenames
    trend_match: bool = False
    trends: list[dict] = field(default_factory=list)
    gen_captions: bool = settings.default_gen_captions
    min_clip_seconds: int = settings.min_clip_seconds
    max_clip_seconds: int = settings.max_clip_seconds
    clips: list[ClipResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def dir(self) -> Path:
        return JOBS_DIR / self.id

    def source_video_path(self) -> Path | None:
        for c in sorted(self.dir().glob("source.*")):
            if c.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}:
                return c
        return None

    def save(self) -> None:
        d = self.dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def set_stage(self, stage: str, message: str = "") -> None:
        self.stage = stage
        for name, prog in STAGES:
            if name == stage:
                self.progress = prog
                break
        if message:
            self.message = message
        self.save()


# In-memory registry (also rebuildable from disk).
_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def get_job(job_id: str) -> Job | None:
    with _lock:
        if job_id in _jobs:
            return _jobs[job_id]
    p = JOBS_DIR / job_id / "job.json"
    if p.exists():
        return _load_from_disk(p)
    return None


def list_jobs() -> list[Job]:
    jobs: dict[str, Job] = {}
    for p in JOBS_DIR.glob("*/job.json"):
        try:
            job = _load_from_disk(p)
            jobs[job.id] = job
        except Exception:
            continue
    with _lock:
        jobs.update(_jobs)
    return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)


def _load_from_disk(path: Path) -> Job:
    data = json.loads(path.read_text(encoding="utf-8"))
    clip_fields = {f.name for f in fields(ClipResult)}
    clips = [
        ClipResult(**{k: v for k, v in c.items() if k in clip_fields})
        for c in data.pop("clips", [])
    ]
    job_fields = {f.name for f in fields(Job)}
    job = Job(**{k: v for k, v in data.items() if k in job_fields})
    job.clips = clips
    return job


def create_job(
    *,
    source_type: str,
    source: str,
    clip_count: int = settings.default_clip_count,
    burn_captions: bool = True,
    aspect: str = settings.default_aspect,
    reframe: str = settings.default_reframe,
    edit_style: str = settings.default_edit_style,
    caption_style: str = settings.default_caption_style,
    background: str = "",
    music: str = settings.default_music,
    niche: str = settings.default_niche,
    trend_match: bool = False,
    gen_captions: bool = settings.default_gen_captions,
    min_seconds: int = settings.min_clip_seconds,
    max_seconds: int = settings.max_clip_seconds,
    topic: str = "",
    voice: str = "",
    mood: str = "",
    news_seconds: int = 30,
    custom_script: str = "",
    images: list[str] | None = None,
    local_path: Path | None = None,
) -> Job:
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        source_type=source_type,
        source=source,
        clip_count=clip_count,
        burn_captions=burn_captions,
        aspect=aspect,
        reframe=reframe,
        edit_style=edit_style,
        caption_style=caption_style,
        background=background,
        music=music,
        niche=niche,
        trend_match=trend_match,
        gen_captions=gen_captions,
        min_clip_seconds=min_seconds,
        max_clip_seconds=max_seconds,
        topic=topic,
        voice=voice,
        mood=mood,
        news_seconds=news_seconds,
        custom_script=custom_script,
        images=images or [],
    )
    with _lock:
        _jobs[job_id] = job
    job.save()

    t = threading.Thread(target=_run_pipeline, args=(job, local_path), daemon=True)
    t.start()
    return job


_BG_EXTS = {".mp4", ".mkv", ".webm", ".mov"}
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _all_backgrounds() -> list[Path]:
    avail = [p for p in sorted(BACKGROUNDS_DIR.glob("*")) if p.suffix.lower() in _BG_EXTS]
    if not avail:
        raise RuntimeError(
            "No background video found. Add football B-roll to data/backgrounds/ "
            "(or click 'Fetch football B-roll (Pexels)')."
        )
    return avail


def _all_music() -> list[Path]:
    return [p for p in sorted(MUSIC_DIR.glob("*")) if p.suffix.lower() in _AUDIO_EXTS]


def _mood_for_index(job: Job, i: int) -> str | None:
    """Football mood key for clip i: None = manual, a MOODS key = fixed, else rotate."""
    if job.mood == "manual":
        return None
    if job.mood and job.mood in football.MOODS:
        return job.mood
    return football.MOOD_ORDER[i % len(football.MOOD_ORDER)]


def _pick(pool: list[Path], keywords: list[str], counters: dict) -> Path | None:
    if not pool:
        return None
    matches = [p for p in pool if any(kw in p.name.lower() for kw in keywords)] or pool
    sig = "|".join(p.name for p in matches)
    idx = counters.get(sig, 0)
    counters[sig] = idx + 1
    return matches[idx % len(matches)]


def _manual_backgrounds(job: Job) -> list[Path]:
    if job.background:
        cand = BACKGROUNDS_DIR / job.background
        if cand.exists():
            return [cand]
    return _all_backgrounds()


def _resolve_music(job: Job) -> Path | None:
    if not job.music:
        return None
    cand = MUSIC_DIR / job.music
    return cand if cand.exists() else None


# --- NEWS mode -------------------------------------------------------------

def _news_subjects(job: Job, niche: str, n: int) -> list[str]:
    """Explicit subject lines first, then web-search-filled trending subjects."""
    explicit = [t.strip() for t in (job.topic or "").splitlines() if t.strip()]
    subjects = explicit[:n]
    if len(subjects) < n:
        job.set_stage("scripting", f"Finding {n - len(subjects)} trending stories...")
        subjects += football.trending_subjects(
            n - len(subjects), api_key=settings.anthropic_api_key,
            model=settings.claude_model, niche=niche, avoid=subjects,
        )
    if not subjects:
        subjects = [job.topic.strip() or niche]
    return subjects[:n]


def _run_news(job: Job, work_dir: Path, clips_dir: Path) -> None:
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    niche = job.niche or football.DEFAULT_NICHE
    all_bgs = _all_backgrounds()
    manual_bgs = _manual_backgrounds(job)
    all_music = _all_music()
    manual_music = _resolve_music(job)

    # Custom scripts (split on a line that is just '---') take priority over
    # web-searched subjects. Each chunk becomes one edit, posted verbatim.
    custom_chunks: list[str] = []
    if job.custom_script.strip():
        custom_chunks = [c.strip() for c in re.split(r"(?m)^\s*-{3,}\s*$", job.custom_script) if c.strip()]
    if custom_chunks:
        items = custom_chunks[:20]
    else:
        n = max(1, min(20, job.clip_count))
        items = _news_subjects(job, niche, n)
    total = len(items)
    counters: dict[str, int] = {}
    voice_counters: dict[str, int] = {}

    for i, item in enumerate(items):
        mood = _mood_for_index(job, i)
        if mood:
            cfg = football.MOODS[mood]
            tone = cfg["tone"]
            vlist = cfg["voices"]
            voice = vlist[voice_counters.get(mood, 0) % len(vlist)]
            voice_counters[mood] = voice_counters.get(mood, 0) + 1
            music = _pick(all_music, cfg["music_kw"], counters)
            background = _pick(all_bgs, cfg["bg_kw"], counters)
            mood_label = cfg["label"]
        else:
            tone = ""
            voice = job.voice or football.DEFAULT_VOICE
            music = manual_music
            background = manual_bgs[i % len(manual_bgs)]
            mood_label = "manual"

        job.progress = 0.10 + 0.85 * (i / total)
        if custom_chunks:
            job.set_stage("scripting", f"Edit {i + 1}/{total} ({mood_label}): preparing your script...")
            job.save()
            script = football.script_to_news(
                item, api_key=settings.anthropic_api_key,
                model=settings.claude_model, niche=niche,
            )
        else:
            job.set_stage("scripting", f"Story {i + 1}/{total} ({mood_label}): pulling latest news...")
            job.save()
            script = football.news_script(
                item, api_key=settings.anthropic_api_key,
                model=settings.claude_model, niche=niche, tone=tone,
                seconds=max(12, min(90, job.news_seconds or 30)),
            )

        voice_name = voiceover.VOICES.get(voice, voice).split(" — ")[0]
        job.set_stage("voicing", f"Story {i + 1}/{total}: voicing ({voice_name})...")
        audio_path = voiceover.synthesize(
            script.text, work_dir / f"voice_{i + 1:02d}.mp3", voice=voice
        )

        job.set_stage("transcribing", f"Story {i + 1}/{total}: timing captions...")
        transcript = transcriber.transcribe(
            audio_path, work_dir, model_size=settings.whisper_model
        )

        job.set_stage("clipping", f"Story {i + 1}/{total}: compositing...")
        filename = f"clip_{i + 1:02d}.mp4"
        clipper.render_voiceover(
            background, audio_path, clips_dir / filename,
            words=transcript.words, caption_style=job.caption_style,
            headline=script.title if settings.voiceover_headline else None,
            music=music, music_volume=settings.music_volume,
        )
        dur = ffmpeg_utils.probe_duration(clips_dir / filename)
        bits = [f"News · {script.subject}", f"mood: {mood_label}", f"voice: {voice_name}"]
        if background:
            bits.append(f"bg: {background.name}")
        job.clips.append(
            ClipResult(
                index=i + 1, title=script.title, hook="", score=0,
                reason=" · ".join(bits),
                start=0.0, end=round(dur, 2), duration=round(dur, 2), filename=filename,
                hashtags=script.hashtags, name_tag=script.name_tag,
                subject=script.subject, sources=script.sources, edit_style="news",
                captions=[{"caption": script.title, "hashtags": script.hashtags}],
                mood=mood_label,
            )
        )
        job.save()

    if not job.clips:
        raise RuntimeError("No news clips were produced.")
    job.title = job.clips[0].title if total == 1 else f"{total} World Cup news edits"
    job.status = "done"
    job.set_stage("done", f"Done. {len(job.clips)} news edit(s) ready.")


# --- IMAGE EDIT mode -------------------------------------------------------

def _job_images(job: Job) -> list[Path]:
    """Images for this edit: ones uploaded into the job dir, else named ones from
    data/images/, else every image in data/images/."""
    job_imgs = sorted(
        p for p in (job.dir() / "images").glob("*") if p.suffix.lower() in _IMG_EXTS
    ) if (job.dir() / "images").exists() else []
    if job_imgs:
        return job_imgs
    if job.images:
        named = [IMAGES_DIR / n for n in job.images]
        named = [p for p in named if p.exists()]
        if named:
            return named
    return sorted(p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in _IMG_EXTS)


def _run_imageedit(job: Job, work_dir: Path, clips_dir: Path) -> None:
    imgs = _job_images(job)
    if not imgs:
        raise RuntimeError(
            "No images found. Upload player photos, or drop them in data/images/."
        )
    music = _resolve_music(job)
    if music is None:
        allm = _all_music()
        music = allm[0] if allm else None

    name_tag = next((t.strip() for t in (job.topic or "").splitlines() if t.strip()), "")
    job.set_stage("clipping", f"Building image edit from {len(imgs)} photo(s)...")
    grid = None
    if music:
        from .core import beats as _beats  # noqa: PLC0415
        try:
            grid = _beats.detect(music)
        except Exception:
            grid = None

    filename = "clip_01.mp4"
    clipper.render_image_edit(
        imgs, clips_dir / filename,
        name_tag=name_tag, title=job.title or "", music=music,
        music_volume=settings.edit_music_volume, grid=grid,
    )
    dur = ffmpeg_utils.probe_duration(clips_dir / filename)
    job.clips.append(
        ClipResult(
            index=1, title=job.title or (name_tag or "World Cup edit"), hook="",
            score=0, reason=f"Image edit · {len(imgs)} photos"
                            + (f" · music: {music.stem}" if music else ""),
            start=0.0, end=round(dur, 2), duration=round(dur, 2), filename=filename,
            hashtags=["worldcup", "football", "edit", "fyp"],
            name_tag=name_tag, edit_style="imageedit",
            captions=[{"caption": job.title or name_tag,
                       "hashtags": ["worldcup", "football", "edit", "fyp"]}],
        )
    )
    if not job.title:
        job.title = name_tag or "World Cup image edit"
    job.status = "done"
    job.set_stage("done", "Done. Image edit ready.")


# --- HIGHLIGHTS mode -------------------------------------------------------

def _run_pipeline(job: Job, local_path: Path | None) -> None:
    try:
        job.status = "running"
        work_dir = job.dir()
        clips_dir = work_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        if job.source_type == "news":
            _run_news(job, work_dir, clips_dir)
            return
        if job.source_type == "imageedit":
            _run_imageedit(job, work_dir, clips_dir)
            return

        # highlights mode: acquire the source video.
        job.set_stage("downloading", "Fetching source video...")
        if job.source_type == "url":
            video_path, title = downloader.download_from_url(job.source, work_dir)
        else:
            if local_path is None or not local_path.exists():
                raise FileNotFoundError("Uploaded file not found.")
            video_path, title = downloader.use_local_file(local_path, work_dir)
        job.title = title
        job.save()

        src_w, src_h = ffmpeg_utils.probe_resolution(video_path)

        # Transcribe the commentary (the signal for where the big moments are).
        job.set_stage("transcribing", "Transcribing commentary (local Whisper)...")
        transcript = transcriber.transcribe(
            video_path, work_dir, model_size=settings.whisper_model
        )
        (work_dir / "transcript.txt").write_text(transcript.full_text(), encoding="utf-8")

        job.set_stage("analyzing", "Finding the biggest moments...")
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        plans = analyzer.select_clips(
            transcript, api_key=settings.anthropic_api_key,
            model=settings.claude_model, clip_count=job.clip_count,
            min_seconds=job.min_clip_seconds, max_seconds=job.max_clip_seconds,
            niche=job.niche, highlight_mode=True,
        )
        plans = [analyzer.snap_to_words(p, transcript) for p in plans]
        if not plans:
            raise RuntimeError("The analyzer returned no moments.")

        if job.gen_captions:
            job.set_stage("captions", "Writing post captions...")
            analyzer.generate_captions(
                plans, job.niche, api_key=settings.anthropic_api_key,
                model=settings.claude_model,
            )

        music_path = _resolve_music(job)
        bg_path = None
        if job.background and job.edit_style != "football_edit":
            cand = BACKGROUNDS_DIR / job.background
            if cand.exists():
                bg_path = cand

        total = len(plans)
        for i, plan in enumerate(plans):
            job.set_stage("clipping", f"Rendering edit {i + 1} of {total}: {plan.title}")
            job.progress = 0.60 + 0.35 * (i / max(1, total))
            job.save()
            filename = f"clip_{i + 1:02d}.mp4"
            out_path = clips_dir / filename

            if job.edit_style == "football_edit":
                clipper.render_football_edit(
                    video_path, out_path, start=plan.start, end=plan.end,
                    src_w=src_w, src_h=src_h, reframe=job.reframe,
                    name_tag=plan.name_tag, title=plan.title,
                    music=music_path, music_volume=settings.edit_music_volume,
                    zoom_punch=settings.edit_zoom_punch, has_audio=True,
                )
            else:
                clipper.render_clip(
                    video_path, out_path, start=plan.start, end=plan.end,
                    words=transcript.words, burn_captions=job.burn_captions,
                    aspect=job.aspect, reframe=job.reframe,
                    caption_style=job.caption_style, background=bg_path,
                    src_w=src_w, src_h=src_h,
                )
            job.clips.append(
                ClipResult(
                    index=i + 1, title=plan.title, hook=plan.hook, score=plan.score,
                    reason=plan.reason, start=round(plan.start, 2),
                    end=round(plan.end, 2), duration=round(plan.end - plan.start, 2),
                    filename=filename, hashtags=plan.hashtags, name_tag=plan.name_tag,
                    edit_style=job.edit_style, captions=plan.captions,
                )
            )
            job.save()

        job.status = "done"
        job.set_stage("done", f"Done. {len(job.clips)} edits ready.")
    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.stage = "error"
        job.error = f"{exc}"
        job.message = f"Failed: {exc}"
        traceback.print_exc()
        job.save()
