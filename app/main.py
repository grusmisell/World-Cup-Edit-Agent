"""FastAPI app: web UI + API for the clipping agent."""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import JOBS_DIR, UPLOADS_DIR, BACKGROUNDS_DIR, MUSIC_DIR, IMAGES_DIR, settings
from . import jobs as jobs_mod

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _list_backgrounds() -> list[str]:
    return sorted(
        p.name for p in BACKGROUNDS_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )


def _list_music() -> list[str]:
    return sorted(
        p.name for p in MUSIC_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    )


def _list_images() -> list[str]:
    return sorted(
        p.name for p in IMAGES_DIR.glob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )

app = FastAPI(title="World Cup Edit Agent")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.ico")
def favicon() -> Response:
    # No favicon asset; return empty to avoid a 404 on every page load.
    return Response(status_code=204)


@app.get("/api/backgrounds")
def backgrounds() -> dict:
    return {"backgrounds": _list_backgrounds()}


@app.get("/api/music")
def music() -> dict:
    return {"music": _list_music()}


@app.get("/api/images")
def images() -> dict:
    return {"images": _list_images()}


@app.get("/api/moods")
def moods() -> dict:
    from .core.football import MOODS, MOOD_ORDER

    return {"moods": {k: MOODS[k]["label"] for k in MOOD_ORDER}}


@app.get("/api/pexels/status")
def pexels_status() -> dict:
    return {"configured": bool(settings.pexels_api_key.strip())}


@app.post("/api/pexels/fetch")
def pexels_fetch(per_mood: int = Form(default=1)) -> JSONResponse:
    """Download real copyright-safe background loops from Pexels (one or more per
    mood), named so the mood-matcher picks them. Needs PEXELS_API_KEY in .env."""
    from .core import pexels

    per_mood = max(1, min(3, per_mood))
    try:
        saved = pexels.fetch_mood_packs(per_mood=per_mood)
    except pexels.PexelsError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Pexels fetch failed: {exc}")
    return JSONResponse({"saved": saved, "backgrounds": _list_backgrounds()})


@app.get("/api/voices")
def voices() -> dict:
    from .core.voiceover import VOICES, DEFAULT_VOICE

    return {"voices": VOICES, "default": DEFAULT_VOICE}


@app.get("/api/queue-status")
def queue_status(
    days: int = 7,
    tiktok: int = 3,
    instagram: int = 2,
    youtube: int = 0,
) -> dict:
    """How full the posting queue is for the next `days` days at the given
    per-platform cadence. Each clip is cross-posted to every platform, so the
    clips needed = busiest platform's posts/day x days. `have` = total finished
    clips across all jobs."""
    days = max(1, min(60, int(days)))
    per_day = {}
    for name, v in (("tiktok", tiktok), ("instagram", instagram), ("youtube", youtube)):
        v = int(v or 0)
        if v > 0:
            per_day[name] = max(1, min(12, v))

    done_jobs = [j for j in jobs_mod.list_jobs() if j.status == "done"]
    all_clips = [c for j in done_jobs for c in j.clips]
    total = len(all_clips)

    # Per platform: a clip is still "available" until it's posted TO THAT platform.
    # Generating a brand-new clip adds availability to every platform at once, so
    # the number to generate is the largest single-platform gap.
    platforms: dict[str, dict] = {}
    generate = 0
    covered_days = None
    for p, pd in per_day.items():
        posted_p = sum(1 for c in all_clips if p in (getattr(c, "posted_to", None) or []))
        available = total - posted_p
        needed = pd * days
        deficit = max(0, needed - available)
        cov = available // pd
        platforms[p] = {
            "needed": needed, "available": available, "deficit": deficit,
            "covered_days": cov, "posted": posted_p,
        }
        generate = max(generate, deficit)
        covered_days = cov if covered_days is None else min(covered_days, cov)

    return {
        "days": days,
        "per_day": per_day,
        "total": total,
        "platforms": platforms,
        "generate": generate,
        "covered_days": covered_days or 0,
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "anthropic_key_set": bool(settings.anthropic_api_key),
        "whisper_model": settings.whisper_model,
        "defaults": {
            "clip_count": settings.default_clip_count,
            "min_clip_seconds": settings.min_clip_seconds,
            "max_clip_seconds": settings.max_clip_seconds,
        },
    }


def _job_to_dict(job: jobs_mod.Job) -> dict:
    d = asdict(job)
    return d


@app.post("/api/jobs")
async def create_job(
    mode: str = Form(default="highlights"),   # highlights | news | imageedit
    url: str | None = Form(default=None),
    clip_count: int = Form(default=settings.default_clip_count),
    burn_captions: bool = Form(default=True),
    aspect: str = Form(default=settings.default_aspect),
    reframe: str = Form(default=settings.default_reframe),
    edit_style: str = Form(default=settings.default_edit_style),
    caption_style: str = Form(default=settings.default_caption_style),
    background: str = Form(default=""),
    music: str = Form(default=settings.default_music),
    niche: str = Form(default=settings.default_niche),
    trend_match: bool = Form(default=False),
    gen_captions: bool = Form(default=settings.default_gen_captions),
    min_seconds: int = Form(default=settings.min_clip_seconds),
    max_seconds: int = Form(default=settings.max_clip_seconds),
    topic: str = Form(default=""),
    voice: str = Form(default=""),
    mood: str = Form(default=""),
    news_seconds: int = Form(default=30),
    custom_script: str = Form(default=""),
    title: str = Form(default=""),
    file: UploadFile | None = File(default=None),
    images: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    from .core.captions import STYLES
    from .core.clipper import ASPECTS, REFRAMES
    from .core.football import MOODS, DEFAULT_NICHE, VOICES

    clip_count = max(1, min(20, int(clip_count)))
    if caption_style not in STYLES:
        caption_style = settings.default_caption_style
    if aspect not in ASPECTS:
        aspect = settings.default_aspect
    if reframe not in REFRAMES:
        reframe = settings.default_reframe
    if edit_style not in ("football_edit", "clean"):
        edit_style = settings.default_edit_style
    background = Path(background).name if background else ""
    if background and background not in _list_backgrounds():
        background = ""
    music = Path(music).name if music else ""
    if music and music not in _list_music():
        music = ""
    niche = (niche or "").strip()[:80] or DEFAULT_NICHE
    min_seconds = max(5, min(170, int(min_seconds)))
    max_seconds = max(min_seconds + 5, min(180, int(max_seconds)))
    if voice not in VOICES:
        voice = ""
    if mood not in MOODS and mood != "manual":
        mood = ""  # auto-mix

    # NEWS mode: subject(s) -> web-search news -> voiceover edit.
    if mode == "news":
        first = next((t.strip() for t in (topic or "").splitlines() if t.strip()), "")
        job = jobs_mod.create_job(
            source_type="news",
            source=(first or niche),
            clip_count=clip_count,
            burn_captions=burn_captions,
            caption_style=caption_style,
            background=background,
            music=music,
            niche=niche,
            topic=(topic or "").strip()[:2000],
            voice=voice,
            mood=mood,
            news_seconds=max(12, min(90, int(news_seconds))),
            custom_script=(custom_script or "").strip()[:12000],
        )
        return JSONResponse({"id": job.id})

    # IMAGE EDIT mode: uploaded (or library) photos -> beat-synced montage.
    if mode == "imageedit":
        names: list[str] = []
        for up in images or []:
            if not up or not up.filename:
                continue
            ext = Path(up.filename).suffix.lower()
            if ext not in _IMAGE_EXTS:
                continue
            dest = IMAGES_DIR / f"{Path(up.filename).stem}_{_rand()}{ext}"
            with dest.open("wb") as out:
                shutil.copyfileobj(up.file, out)
            names.append(dest.name)
        if not names and not _list_images():
            raise HTTPException(400, "Upload at least one image (or add some to data/images/).")
        job = jobs_mod.create_job(
            source_type="imageedit",
            source="images",
            music=music,
            niche=niche,
            topic=(topic or "").strip()[:200],   # used as the name tag
            title=(title or "").strip()[:80],
            images=names,
        )
        return JSONResponse({"id": job.id})

    # COUNTDOWN mode: a ranked "Top N" voiceover edit over uploaded player images.
    if mode == "countdown":
        names: list[str] = []
        for up in images or []:
            if not up or not up.filename:
                continue
            ext = Path(up.filename).suffix.lower()
            if ext not in _IMAGE_EXTS:
                continue
            dest = IMAGES_DIR / f"{Path(up.filename).stem}_{_rand()}{ext}"
            with dest.open("wb") as out:
                shutil.copyfileobj(up.file, out)
            names.append(dest.name)
        if not names and not _list_images():
            raise HTTPException(400, "Upload player/team images (in countdown order) for a countdown.")
        job = jobs_mod.create_job(
            source_type="countdown",
            source=(topic or niche),
            clip_count=max(3, min(10, int(clip_count))),   # number of ranked items
            music=music,
            niche=niche,
            topic=(topic or "").strip()[:300],
            voice=voice,
            images=names,
        )
        return JSONResponse({"id": job.id})

    # HIGHLIGHTS mode: a match/highlights video (file or URL) -> football edits.
    if file is not None and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXT:
            raise HTTPException(400, f"Unsupported file type: {ext}")
        dest = UPLOADS_DIR / f"{Path(file.filename).stem}_{_rand()}{ext}"
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        job = jobs_mod.create_job(
            source_type="file", source=file.filename, clip_count=clip_count,
            burn_captions=burn_captions, aspect=aspect, reframe=reframe,
            edit_style=edit_style, caption_style=caption_style, background=background,
            music=music, niche=niche, trend_match=trend_match,
            gen_captions=gen_captions, min_seconds=min_seconds, max_seconds=max_seconds,
            local_path=dest,
        )
    elif url:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(400, "Provide a valid http(s) URL.")
        job = jobs_mod.create_job(
            source_type="url", source=url, clip_count=clip_count,
            burn_captions=burn_captions, aspect=aspect, reframe=reframe,
            edit_style=edit_style, caption_style=caption_style, background=background,
            music=music, niche=niche, trend_match=trend_match,
            gen_captions=gen_captions, min_seconds=min_seconds, max_seconds=max_seconds,
        )
    else:
        raise HTTPException(400, "Provide a highlights video URL or upload a file.")

    return JSONResponse({"id": job.id})


def _rand() -> str:
    import uuid

    return uuid.uuid4().hex[:8]


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    return JSONResponse([_job_to_dict(j) for j in jobs_mod.list_jobs()])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JSONResponse(_job_to_dict(job))


@app.get("/api/jobs/{job_id}/clips/{filename}")
def get_clip(job_id: str, filename: str, download: bool = False) -> FileResponse:
    safe = Path(filename).name
    path = JOBS_DIR / job_id / "clips" / safe
    if not path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=safe if download else None,
    )


def _slug(text: str, fallback: str = "clips") -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", (text or "").strip()).strip("-").lower()
    return (s or fallback)[:60]


@app.get("/api/jobs/{job_id}/download-all")
def download_all(job_id: str) -> Response:
    """Bundle every clip in the job into a single zip for one-click download."""
    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    clips_dir = JOBS_DIR / job_id / "clips"
    files = sorted(p for p in clips_dir.glob("*.mp4") if p.is_file())
    if not files:
        raise HTTPException(404, "No clips to download")

    buf = io.BytesIO()
    # mp4 is already compressed — store (no deflate) to keep it fast and light.
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    name = _slug(job.title, fallback=job_id)
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}-clips.zip"'},
    )


@app.post("/api/jobs/{job_id}/clips/{filename}/retrim")
def retrim_clip(
    job_id: str,
    filename: str,
    start: float = Form(...),
    end: float = Form(...),
) -> JSONResponse:
    from .core import transcriber, clipper, ffmpeg_utils

    safe = Path(filename).name
    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    clip = next((c for c in job.clips if c.filename == safe), None)
    if clip is None:
        raise HTTPException(404, "Clip not found")
    src = job.source_video_path()
    if not src or not src.exists():
        raise HTTPException(400, "Source video for this job is no longer available.")
    if end <= start:
        raise HTTPException(400, "End time must be after start time.")

    dur = ffmpeg_utils.probe_duration(src)
    start = max(0.0, start)
    if dur:
        end = min(end, dur)

    transcript = transcriber.load_transcript(job.dir())
    words = transcript.words if transcript else []
    w, h = ffmpeg_utils.probe_resolution(src)
    out_path = job.dir() / "clips" / safe
    bg = BACKGROUNDS_DIR / job.background if job.background else None
    if bg and not bg.exists():
        bg = None
    try:
        clipper.render_clip(
            src, out_path, start=start, end=end, words=words,
            burn_captions=job.burn_captions, aspect=job.aspect,
            reframe=job.reframe, caption_style=job.caption_style,
            background=bg, src_w=w, src_h=h,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Re-trim failed: {exc}")

    clip.start = round(start, 2)
    clip.end = round(end, 2)
    clip.duration = round(end - start, 2)
    job.save()
    return JSONResponse({"start": clip.start, "end": clip.end, "duration": clip.duration})


@app.post("/api/jobs/{job_id}/clips/{filename}/reexport")
def reexport_clip(
    job_id: str,
    filename: str,
    aspect: str = Form(...),
) -> JSONResponse:
    """Render an extra-aspect version of a clip (e.g. a 1:1 for the feed next to
    the 9:16) for cross-posting. Saves a sidecar file you can download."""
    from .core import transcriber, clipper, ffmpeg_utils
    from .core.clipper import ASPECTS

    safe = Path(filename).name
    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    clip = next((c for c in job.clips if c.filename == safe), None)
    if clip is None:
        raise HTTPException(404, "Clip not found")
    if aspect not in ASPECTS:
        raise HTTPException(400, f"Unknown aspect: {aspect}")
    src = job.source_video_path()
    if not src or not src.exists():
        raise HTTPException(
            400,
            "Aspect re-export needs the original source video. Voiceover clips are 9:16 only.",
        )

    transcript = transcriber.load_transcript(job.dir())
    words = transcript.words if transcript else []
    w, h = ffmpeg_utils.probe_resolution(src)
    suffix = aspect.replace(":", "x")  # 1:1 -> 1x1
    new_name = f"{Path(safe).stem}_{suffix}.mp4"
    out_path = job.dir() / "clips" / new_name
    try:
        # Plain reframe in the requested aspect (no split-screen background, which
        # is fixed to 9:16).
        clipper.render_clip(
            src, out_path, start=clip.start, end=clip.end, words=words,
            burn_captions=job.burn_captions, aspect=aspect,
            reframe=job.reframe, caption_style=job.caption_style,
            background=None, src_w=w, src_h=h,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Re-export failed: {exc}")

    return JSONResponse({
        "filename": new_name,
        "aspect": aspect,
        "url": f"/api/jobs/{job_id}/clips/{new_name}?download=true",
    })


@app.post("/api/jobs/{job_id}/clips/{filename}/posted")
def set_posted(
    job_id: str, filename: str,
    platform: str = Form(...), posted: bool = Form(default=True),
) -> JSONResponse:
    """Mark a clip posted (or not) for a specific platform, so the weekly queue
    stops counting it for that platform."""
    from .core import scheduler

    safe = Path(filename).name
    if platform not in scheduler.BEST_TIMES:
        raise HTTPException(400, f"Unknown platform: {platform}")
    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    clip = next((c for c in job.clips if c.filename == safe), None)
    if clip is None:
        raise HTTPException(404, "Clip not found")
    pset = set(clip.posted_to or [])
    pset.add(platform) if posted else pset.discard(platform)
    clip.posted_to = sorted(pset)
    job.save()
    return JSONResponse({"filename": safe, "posted_to": clip.posted_to})


@app.post("/api/jobs/{job_id}/mark-all-posted")
def mark_all_posted(
    job_id: str,
    platform: str = Form(default="all"),
    posted: bool = Form(default=True),
) -> JSONResponse:
    """Mark every clip in a job posted (or not) for one platform, or all of them
    (`platform=all`) — for after you've posted a whole batch."""
    from .core import scheduler

    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    targets = list(scheduler.BEST_TIMES) if platform == "all" else [platform]
    targets = [t for t in targets if t in scheduler.BEST_TIMES]
    if not targets:
        raise HTTPException(400, f"Unknown platform: {platform}")
    for c in job.clips:
        pset = set(c.posted_to or [])
        for t in targets:
            pset.add(t) if posted else pset.discard(t)
        c.posted_to = sorted(pset)
    job.save()
    return JSONResponse({"count": len(job.clips), "platforms": targets, "posted": bool(posted)})


@app.get("/api/youtube/status")
def youtube_status() -> dict:
    from .core import youtube

    return {"configured": youtube.is_configured(), "authorized": youtube.is_authorized()}


@app.post("/api/jobs/{job_id}/clips/{filename}/youtube")
def publish_youtube(job_id: str, filename: str) -> JSONResponse:
    from .core import youtube

    safe = Path(filename).name
    path = JOBS_DIR / job_id / "clips" / safe
    if not path.exists():
        raise HTTPException(404, "Clip not found")

    job = jobs_mod.get_job(job_id)
    clip = next((c for c in job.clips if c.filename == safe), None) if job else None
    title = clip.title if clip else safe
    hook = clip.hook if clip else ""
    hashtags = clip.hashtags if clip else []

    try:
        result = youtube.upload(
            path,
            title=title,
            hook=hook,
            hashtags=hashtags,
            privacy=settings.youtube_privacy,
        )
    except youtube.YouTubeNotConfigured as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"YouTube upload failed: {exc}")

    # A published clip is "used" on YouTube — auto-mark it so the queue stops
    # counting it for that platform (no need to tap the YT chip by hand).
    posted_to = None
    if clip is not None and "youtube" not in (clip.posted_to or []):
        clip.posted_to = sorted(set(clip.posted_to or []) | {"youtube"})
        job.save()
        posted_to = clip.posted_to

    return JSONResponse(
        {"url": result["url"], "privacy": settings.youtube_privacy, "posted_to": posted_to}
    )


def _parse_start(start_date: str):
    from datetime import datetime, timedelta

    s = (start_date or "").strip()
    if s:
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass
    return datetime.now() + timedelta(days=1)


def _clip_dicts(job) -> list[dict]:
    return [
        {"index": c.index, "filename": c.filename, "title": c.title,
         "captions": c.captions, "hashtags": c.hashtags}
        for c in job.clips
    ]


def _per_platform_caps(plats: list[str], tiktok: int, instagram: int, youtube: int) -> dict:
    """Per-platform posts/day from the form, falling back to the recommended
    default for any platform left at 0."""
    from .core import scheduler

    raw = {"tiktok": tiktok, "instagram": instagram, "youtube": youtube}
    caps = {}
    for p in plats:
        v = int(raw.get(p) or 0)
        caps[p] = max(1, min(12, v)) if v > 0 else scheduler.RECOMMENDED_PER_DAY.get(p, 1)
    return caps


@app.post("/api/jobs/{job_id}/schedule")
def make_schedule(
    job_id: str,
    platforms: str = Form(default="youtube,tiktok,instagram"),
    tiktok_per_day: int = Form(default=0),
    instagram_per_day: int = Form(default=0),
    youtube_per_day: int = Form(default=0),
    start_date: str = Form(default=""),
) -> JSONResponse:
    from .core import scheduler

    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    plats = [p.strip() for p in platforms.split(",") if p.strip() in scheduler.BEST_TIMES]
    if not plats:
        raise HTTPException(400, "Pick at least one platform.")
    caps = _per_platform_caps(plats, tiktok_per_day, instagram_per_day, youtube_per_day)
    sched = scheduler.build_schedule(
        _clip_dicts(job), platforms=plats,
        per_platform=caps, start=_parse_start(start_date),
    )
    return JSONResponse({"schedule": sched, "per_day": caps})


def _export_schedule_folders(job, schedule: list[dict]) -> dict:
    """Lay the scheduled clips out into per-platform / per-day folders so each
    day's posts for each platform are grouped together. Uses hard links (no extra
    disk; falls back to copy) and writes a captions.txt per folder."""
    import os
    from datetime import datetime

    clips_dir = job.dir() / "clips"
    dest = job.dir() / "posts"
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    counts: dict[str, int] = {}
    # manifest: platform -> date -> list of source clip filenames in that folder,
    # so a single click can mark exactly that day's folder posted to that platform.
    manifest: dict[str, dict[str, list[str]]] = {}
    for post in schedule:
        src = clips_dir / post["clip_filename"]
        if not src.exists():
            continue
        when = datetime.fromisoformat(post["when"])
        date = when.strftime("%Y-%m-%d")
        folder = dest / post["platform"] / date
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{when.strftime('%H%M')}_clip{int(post['clip_index']):02d}_{_slug(post['title'])[:30]}.mp4"
        target = folder / name
        if not target.exists():
            try:
                os.link(src, target)          # hard link — no extra disk
            except OSError:
                shutil.copy2(src, target)     # cross-device / FS fallback
        with (folder / "captions.txt").open("a", encoding="utf-8") as f:
            f.write(f"[{when.strftime('%a %b %d, %H:%M')}]  {name}\n{post['caption']}\n\n")
        counts[post["platform"]] = counts.get(post["platform"], 0) + 1
        manifest.setdefault(post["platform"], {}).setdefault(date, []).append(post["clip_filename"])

    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    # Flatten into UI-friendly folder rows (one per platform/day) for one-click marking.
    folders = [
        {"platform": plat, "date": date, "clips": sorted(set(files)), "count": len(set(files))}
        for plat, dates in manifest.items()
        for date, files in dates.items()
    ]
    folders.sort(key=lambda f: (f["date"], f["platform"]))
    return {"path": str(dest), "counts": counts, "folders": folders}


@app.post("/api/jobs/{job_id}/export-folders")
def export_folders(
    job_id: str,
    platforms: str = Form(default="tiktok,instagram"),
    tiktok_per_day: int = Form(default=0),
    instagram_per_day: int = Form(default=0),
    youtube_per_day: int = Form(default=0),
    start_date: str = Form(default=""),
) -> JSONResponse:
    from .core import scheduler

    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    plats = [p.strip() for p in platforms.split(",") if p.strip() in scheduler.BEST_TIMES]
    if not plats:
        raise HTTPException(400, "Pick at least one platform.")
    caps = _per_platform_caps(plats, tiktok_per_day, instagram_per_day, youtube_per_day)
    sched = scheduler.build_schedule(
        _clip_dicts(job), platforms=plats,
        per_platform=caps, start=_parse_start(start_date),
    )
    result = _export_schedule_folders(job, sched)
    result["per_day"] = caps
    return JSONResponse(result)


@app.post("/api/jobs/{job_id}/mark-folder-posted")
def mark_folder_posted(
    job_id: str,
    platform: str = Form(...),
    date: str = Form(...),
    posted: bool = Form(default=True),
) -> JSONResponse:
    """One-click: mark every clip in an exported platform/day folder posted (or not)
    to that platform — saves tapping each clip's chip after posting a day's batch.
    Reads the manifest written by the last export-folders run."""
    from .core import scheduler

    if platform not in scheduler.BEST_TIMES:
        raise HTTPException(400, f"Unknown platform: {platform}")
    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    manifest_path = job.dir() / "posts" / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(400, "No exported folders yet — export to platform folders first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filenames = set((manifest.get(platform) or {}).get(date) or [])
    if not filenames:
        raise HTTPException(404, "No folder found for that platform and date.")

    updated = []
    for c in job.clips:
        if c.filename not in filenames:
            continue
        pset = set(c.posted_to or [])
        pset.add(platform) if posted else pset.discard(platform)
        c.posted_to = sorted(pset)
        updated.append({"filename": c.filename, "posted_to": c.posted_to})
    job.save()
    return JSONResponse(
        {"platform": platform, "date": date, "posted": bool(posted), "updated": updated}
    )


@app.post("/api/jobs/{job_id}/schedule-youtube")
def schedule_youtube(
    job_id: str,
    posts_per_day: int = Form(default=2),
    start_date: str = Form(default=""),
) -> JSONResponse:
    from datetime import datetime, timezone
    from .core import scheduler, youtube

    job = jobs_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    posts_per_day = max(1, min(20, int(posts_per_day)))
    sched = scheduler.build_schedule(
        _clip_dicts(job), platforms=["youtube"],
        posts_per_day=posts_per_day, start=_parse_start(start_date),
    )

    results = []
    for post in sched:
        path = job.dir() / "clips" / post["clip_filename"]
        if not path.exists():
            continue
        clip = next((c for c in job.clips if c.filename == post["clip_filename"]), None)
        publish_at = datetime.fromisoformat(post["when"]).astimezone(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        try:
            res = youtube.upload(
                path, title=clip.title if clip else post["title"],
                hook=clip.hook if clip else "",
                hashtags=clip.hashtags if clip else [],
                publish_at=publish_at,
            )
            results.append({"when": post["when"], "url": res["url"], "ok": True})
            if clip is not None:
                clip.posted_to = sorted(set(clip.posted_to or []) | {"youtube"})
        except youtube.YouTubeNotConfigured as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            results.append({"when": post["when"], "error": str(exc), "ok": False})

    job.save()
    return JSONResponse({"results": results})
