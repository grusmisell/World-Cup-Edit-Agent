"""Plan an optimal posting schedule across platforms (the volume / numbers game).

Cross-posts a job's clips to each selected platform and spreads them over a
calendar, dropping each post into a rough best-time-to-post window for that
platform so posts land in high-engagement slots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

# Rough best-time-to-post local hours per platform (24h), ordered by preference.
# Industry rules of thumb, not gospel — a sensible default to spread posts out.
BEST_TIMES: dict[str, list[int]] = {
    "youtube": [12, 15, 17, 20],
    "tiktok": [7, 12, 19, 22],
    "instagram": [11, 13, 19, 21],
}
PLATFORMS = list(BEST_TIMES.keys())

# Recommended algorithm-safe MAX posts/day per platform (2026 rules of thumb):
# TikTok tolerates the most volume; Instagram Reels reach drops past ~2-3/day;
# YouTube Shorts ~1/day is sustainable. Used as defaults to "maximize output"
# without over-posting into reduced reach.
RECOMMENDED_PER_DAY: dict[str, int] = {
    "tiktok": 3,
    "instagram": 2,
    "youtube": 1,
}


@dataclass
class ScheduledPost:
    when: str  # ISO local datetime
    platform: str
    clip_index: int
    clip_filename: str
    title: str
    caption: str


def _caption_for(clip: dict, variant: int) -> str:
    """Pick a caption for this post, rotating variants so cross-posts differ."""
    caps = clip.get("captions") or []
    if caps:
        v = caps[variant % len(caps)]
        text = (v.get("caption") or "").strip()
        tags = [str(t).lstrip("#") for t in (v.get("hashtags") or [])]
    else:
        text = clip.get("title", "")
        tags = [str(t).lstrip("#") for t in (clip.get("hashtags") or [])]
    tagline = " ".join("#" + t for t in tags if t)
    return (text + ("\n\n" + tagline if tagline else "")).strip()


def _daily_caps(
    platforms: list[str],
    per_platform: dict[str, int] | None,
    posts_per_day: int | None,
) -> dict[str, int]:
    """Resolve a posts/day cap for each platform: explicit per-platform value,
    else a single global value, else the recommended default for that platform."""
    caps: dict[str, int] = {}
    for p in platforms:
        if per_platform and per_platform.get(p):
            caps[p] = max(1, min(12, int(per_platform[p])))
        elif posts_per_day:
            caps[p] = max(1, min(12, int(posts_per_day)))
        else:
            caps[p] = RECOMMENDED_PER_DAY.get(p, 1)
    return caps


def build_schedule(
    clips: list[dict],
    *,
    platforms: list[str],
    posts_per_day: int | None = None,
    per_platform: dict[str, int] | None = None,
    start: datetime,
) -> list[dict]:
    """Return a chronological posting plan.

    Each clip is cross-posted to every selected platform. Each platform fills its
    OWN daily quota independently (so TikTok can run more posts/day than Instagram)
    at its best-time slots. The cap per platform comes from `per_platform`, else
    `posts_per_day` for all, else RECOMMENDED_PER_DAY. `clips` are dicts with
    index/filename/title/captions/hashtags. Returns ScheduledPost dicts sorted by time.
    """
    platforms = [p for p in platforms if p in BEST_TIMES]
    if not platforms or not clips:
        return []
    caps = _daily_caps(platforms, per_platform, posts_per_day)
    base = start.replace(hour=0, minute=0, second=0, microsecond=0)

    out: list[ScheduledPost] = []
    # Schedule each platform independently at its own daily cadence. The caption
    # variant differs per platform so cross-posts of the same clip aren't identical.
    for variant, platform in enumerate(platforms):
        cap = caps[platform]
        hours = BEST_TIMES[platform]
        for i, clip in enumerate(clips):
            day, slot = divmod(i, cap)
            hour = hours[slot % len(hours)]
            minute = (slot // len(hours)) * 20  # extra posts in same slot: +20 min
            when = base + timedelta(days=day, hours=hour, minutes=minute)
            out.append(
                ScheduledPost(
                    when=when.isoformat(timespec="minutes"),
                    platform=platform,
                    clip_index=int(clip.get("index", 0)),
                    clip_filename=clip.get("filename", ""),
                    title=clip.get("title", ""),
                    caption=_caption_for(clip, variant),
                )
            )

    out.sort(key=lambda p: p.when)
    return [asdict(p) for p in out]
