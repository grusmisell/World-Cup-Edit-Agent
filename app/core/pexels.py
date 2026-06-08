"""Fetch real, copyright-safe vertical background loops from Pexels (free stock
video). Saved into data/backgrounds/ with mood-keyword filenames so the
voiceover mood-matcher (voiceover.MOODS bg_kw) picks them automatically.

Pexels content is free for commercial use, no attribution required
(https://www.pexels.com/license/). Needs a free API key in settings.pexels_api_key
(https://www.pexels.com/api/).
"""
from __future__ import annotations

import urllib.request
import urllib.parse
import json
import shutil
import ssl
from pathlib import Path

from ..config import settings, BACKGROUNDS_DIR

API = "https://api.pexels.com/videos/search"

# Validate TLS against certifi's CA bundle (the system store can be stale and
# reject Pexels' cert as "expired"). certifi ships with anthropic/httpx.
try:
    import certifi
    _SSL_CTX: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
except Exception:  # pragma: no cover
    _SSL_CTX = None

# Pexels (Cloudflare) 403s the default "Python-urllib" UA, so send a real one.
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClippingAgent/1.0"

# query terms per football mood -> picked to match football.MOODS[mood]["bg_kw"] so
# the downloaded file's name contains a keyword that mood looks for. Pexels stock is
# copyright-safe (generic stadiums/crowds/pitches — NOT real match footage).
MOOD_QUERIES = {
    "hype":     [("soccer stadium crowd", "stadium"), ("football fans celebrate", "crowd")],
    "analysis": [("soccer pitch aerial", "pitch"), ("stadium drone", "drone")],
    "drama":    [("soccer stadium lights night", "stadium"), ("football fans", "fans")],
}

# Cap downloaded file resolution to keep disk use sane (machine runs near-full).
# Prefer a portrait file with height >= MIN_H but the SMALLEST such (avoid 4K).
MIN_H = 1200
MAX_H = 1920


class PexelsError(RuntimeError):
    pass


def _require_key() -> str:
    key = settings.pexels_api_key.strip()
    if not key:
        raise PexelsError(
            "No Pexels API key. Get a free one at https://www.pexels.com/api/ "
            "and add PEXELS_API_KEY=... to .env."
        )
    return key


def search(query: str, *, per_page: int = 15) -> list[dict]:
    """Search Pexels for stock videos. Returns the raw 'videos' list. No
    orientation filter — football B-roll (stadiums/crowds) is mostly landscape and
    gets centre-cropped to 9:16 at render, so landscape sources are fine."""
    key = _require_key()
    params = urllib.parse.urlencode(
        {"query": query, "size": "medium", "per_page": per_page}
    )
    req = urllib.request.Request(
        f"{API}?{params}", headers={"Authorization": key, "User-Agent": _UA}
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        raise PexelsError(f"Pexels API error {exc.code}: {exc.reason}") from exc
    return data.get("videos", [])


def _best_portrait_file(video: dict) -> dict | None:
    """Pick a sensibly-sized file (height in [MIN_H, MAX_H]) regardless of
    orientation; fall back to the largest under MAX_H, then the smallest available.
    HD landscape (1920x1080) is ideal — it crops cleanly to a vertical slice."""
    files = [
        f for f in video.get("video_files", [])
        if f.get("height") and f.get("width") and f.get("link")
    ]
    if not files:
        return None
    in_band = sorted([f for f in files if MIN_H <= f["height"] <= MAX_H], key=lambda f: f["height"])
    if in_band:
        return in_band[0]
    under = sorted([f for f in files if f["height"] <= MAX_H], key=lambda f: -f["height"])
    return under[0] if under else min(files, key=lambda f: f["height"])


def download_one(query: str, dest_name: str, *, skip_ids: set[int] | None = None) -> Path | None:
    """Search `query`, download the first usable portrait clip to
    data/backgrounds/<dest_name>.mp4. Returns the path, or None if nothing fit."""
    skip_ids = skip_ids or set()
    for video in search(query):
        if video.get("id") in skip_ids:
            continue
        f = _best_portrait_file(video)
        if not f:
            continue
        dest = BACKGROUNDS_DIR / f"{dest_name}.mp4"
        dreq = urllib.request.Request(f["link"], headers={"User-Agent": _UA})
        with urllib.request.urlopen(dreq, timeout=120, context=_SSL_CTX) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        skip_ids.add(video.get("id"))
        return dest
    return None


def fetch_mood_packs(per_mood: int = 1) -> list[str]:
    """Download `per_mood` background(s) for each mood, named with that mood's
    keyword so mood-matching picks them. Returns the saved filenames."""
    _require_key()
    saved: list[str] = []
    seen: set[int] = set()
    for mood, queries in MOOD_QUERIES.items():
        n = 0
        for query, kw in queries:
            if n >= per_mood:
                break
            suffix = "" if n == 0 else f"-{n+1}"
            name = f"pexels-{mood}-{kw}{suffix}"
            try:
                path = download_one(query, name, skip_ids=seen)
            except PexelsError:
                raise
            except Exception:  # noqa: BLE001 -- one bad clip shouldn't abort the rest
                path = None
            if path:
                saved.append(path.name)
                n += 1
    return saved
