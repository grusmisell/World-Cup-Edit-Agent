"""World Cup / football content generation.

Two jobs:
1. NEWS scripts — Claude uses web search to pull the latest real news about a
   player, country, or storyline, then writes a hook-first short-form narration.
2. SUBJECT discovery — Claude (web search) surfaces what's trending in the World
   Cup right now so a batch can auto-fill its topics.

Voices/moods are tuned for sports content (hyped energetic delivery by default).
The narration is voiced by edge-tts (see voiceover.synthesize) and timed by
whisper.cpp for synced captions, exactly like the voiceover pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from .analyzer import _extract_json, _clean_tags
from .voiceover import synthesize, VOICES, DEFAULT_VOICE  # re-exported for jobs.py

DEFAULT_NICHE = "2026 FIFA World Cup football"

# Sports moods: hyped is the default football-edit voice; analysis is calmer.
MOODS: dict[str, dict] = {
    "hype": {
        "label": "Hype",
        "tone": "Explosive, fast-paced hype-man energy: short exclamatory lines, "
                "build to a peak, like a viral football edit voiceover.",
        "voices": ["en-US-BrianNeural", "en-GB-RyanNeural", "en-US-GuyNeural",
                   "en-AU-WilliamNeural"],
        "music_kw": ["trap", "drill", "hype", "epic", "phonk", "edm", "trailer"],
        "bg_kw": ["stadium", "crowd", "football", "soccer", "fireworks", "action"],
    },
    "analysis": {
        "label": "Analysis",
        "tone": "Sharp, confident pundit delivery: clear, punchy analysis with "
                "authority, like a tactics breakdown.",
        "voices": ["en-GB-RyanNeural", "en-US-AndrewNeural", "en-US-ChristopherNeural",
                   "en-GB-SoniaNeural"],
        "music_kw": ["ambient", "cinematic", "lofi", "documentary", "underscore"],
        "bg_kw": ["stadium", "pitch", "tactics", "city", "drone", "aerial"],
    },
    "drama": {
        "label": "Drama",
        "tone": "Cinematic, emotional storytelling: weighty, dramatic cadence that "
                "makes a moment feel legendary.",
        "voices": ["en-US-GuyNeural", "en-AU-WilliamNeural", "en-US-EricNeural"],
        "music_kw": ["cinematic", "epic", "emotional", "orchestral", "dramatic"],
        "bg_kw": ["stadium", "trophy", "slowmo", "cinematic", "fans", "tears"],
    },
}
MOOD_ORDER = ["hype", "analysis", "drama"]


@dataclass
class NewsScript:
    title: str
    text: str
    hashtags: list[str]
    subject: str          # the player/country/storyline this is about
    name_tag: str         # short on-screen label (player surname / country)
    sources: list[str]    # source URLs Claude cited (best-effort)


def _client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def news_script(
    subject: str,
    *,
    api_key: str,
    model: str,
    niche: str = DEFAULT_NICHE,
    seconds: int = 32,
    tone: str = "",
) -> NewsScript:
    """Web-search the latest real news on `subject` and write a short-form script.

    `subject` is a player ("Kylian Mbappe"), country ("Argentina"), or storyline
    ("USA World Cup squad"). If blank, Claude picks the single hottest current
    World Cup story. Uses Claude's server-side web_search so facts are current."""
    words = max(45, int(seconds * 2.6))
    subject_line = (
        f"Subject: {subject.strip()}."
        if subject and subject.strip()
        else "Pick the single hottest current World Cup story right now."
    )
    tone_line = f"- DELIVERY TONE: {tone.strip()}\n" if tone and tone.strip() else ""
    prompt = (
        f"You write faceless short-form sports-news voiceover scripts for a {niche} "
        f"channel (TikTok / Reels / Shorts). {subject_line}\n\n"
        f"FIRST use web search to find the LATEST real news (this week) about the "
        f"subject — transfers, injuries, results, squad news, controversies, form. "
        f"Only state things supported by what you found; never invent stats or quotes.\n\n"
        f"Then write a ~{seconds}-second narration. HARD LIMIT: {words} words MAX "
        f"(spoken at ~2.6 words/sec — going over makes the video too long). Count your "
        f"words and stay under the cap.\n"
        f"- Open with a 3-second HOOK (a bold, true claim or a burning question).\n"
        f"{tone_line}"
        f"- Punchy, conversational, retention-optimized. Short sentences. No filler.\n"
        f"- Pay off the hook with the real news; end on a loop-y or provocative line.\n"
        f"- 'script' is plain narration ONLY — no scene directions, emojis, or markdown.\n\n"
        f'Return ONLY JSON: {{"title": "<punchy title, max 60 chars>", '
        f'"subject": "<the player/country/storyline>", '
        f'"name_tag": "<2-18 char on-screen label, e.g. a surname or country>", '
        f'"script": "<the narration text>", '
        f'"hashtags": ["<4-6 hashtags, no # symbol, lowercase, no spaces>"], '
        f'"sources": ["<source url>"]}}. No text outside the JSON.'
    )

    client = _client(api_key)
    resp = client.messages.create(
        model=model, max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    data = _extract_json(raw)
    text = str(data.get("script", "")).strip()
    if not text:
        raise RuntimeError("The news script generator returned an empty script.")
    subj = str(data.get("subject", subject or "World Cup")).strip()
    return NewsScript(
        title=str(data.get("title", subj))[:80],
        text=text,
        hashtags=_clean_tags(data.get("hashtags")),
        subject=subj[:120],
        name_tag=str(data.get("name_tag", subj.split()[-1] if subj else "")).strip()[:18],
        sources=[str(s).strip() for s in (data.get("sources") or []) if str(s).strip()][:6],
    )


def script_to_news(
    text: str,
    *,
    api_key: str,
    model: str,
    niche: str = DEFAULT_NICHE,
) -> NewsScript:
    """Wrap a user-supplied finished script as a NewsScript, deriving post metadata
    (title / on-screen name tag / hashtags) with one cheap Claude call (NO web
    search — the script is taken as-is). Robust: falls back to sensible defaults if
    the metadata call fails, so the script always renders."""
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Empty custom script.")
    prompt = (
        f"Here is a finished short-form football voiceover script for a {niche} "
        f"channel. Do NOT rewrite it — just write post metadata for it.\n\n"
        f"SCRIPT:\n{text}\n\n"
        f'Return ONLY JSON: {{"title": "<punchy title, max 60 chars>", '
        f'"name_tag": "<2-18 char on-screen label, e.g. a surname or country>", '
        f'"hashtags": ["<4-6 hashtags, no # symbol, lowercase, no spaces>"]}}. '
        f"No text outside the JSON."
    )
    data: dict = {}
    try:
        resp = _client(api_key).messages.create(
            model=model, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(raw)
    except Exception:
        data = {}
    title = str(data.get("title") or text[:48]).strip()[:80]
    name_tag = str(data.get("name_tag") or "").strip()[:18]
    hashtags = _clean_tags(data.get("hashtags")) or ["worldcup2026", "football", "soccer", "fyp"]
    return NewsScript(
        title=title, text=text, hashtags=hashtags,
        subject=title, name_tag=name_tag, sources=[],
    )


@dataclass
class CountdownScript:
    title: str
    intro: str
    items: list[dict]   # each: {rank:int, name:str, tag:str, blurb:str}


def countdown_script(
    topic: str,
    count: int,
    *,
    api_key: str,
    model: str,
    niche: str = DEFAULT_NICHE,
    web_search: bool = True,
) -> CountdownScript:
    """Write a ranked 'Top N' countdown (like the Makaloozas edits): a hook intro
    plus one entry per rank (label + a tight spoken blurb), ordered from rank N
    down to 1. Uses web search for current accuracy when enabled."""
    count = max(3, min(10, count))
    topic = (topic or f"top {count} teams at the 2026 World Cup").strip()
    search_line = "Use web search for current, accurate info. " if web_search else ""
    prompt = (
        f"Write a TOP {count} countdown for a {niche} short-form video about: {topic}. "
        f"{search_line}Rank from {count} (shown first) DOWN to 1 (the best, shown last). "
        f"Give a SHORT one-line intro hook (max 12 words), then for each rank a short "
        f"on-screen label and a PUNCHY spoken blurb (STRICTLY max 14 words) on why it ranks "
        f"there. Conversational, hype, no filler, no emojis. Keep it tight so the whole video "
        f"is ~45 seconds.\n\n"
        f'Return ONLY JSON: {{"title": "<max 60 chars>", "intro": "<1-sentence hook>", '
        f'"items": [{{"rank": <int>, "name": "<team or player>", '
        f'"tag": "<2-18 char on-screen label>", "blurb": "<why it ranks here>", '
        f'"query": "<a YouTube search to find a short skills/goals highlight clip of '
        f'this player or team, e.g. \'Kylian Mbappe goals skills\'>"}}]}} '
        f"with items ordered rank {count} first down to rank 1 last. No text outside the JSON."
    )
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}] if web_search else []
    resp = _client(api_key).messages.create(
        model=model, max_tokens=2500,
        messages=[{"role": "user", "content": prompt}], tools=tools,
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    data = _extract_json(raw)
    items: list[dict] = []
    for it in data.get("items", []):
        try:
            rank = int(it.get("rank"))
        except (TypeError, ValueError):
            continue
        name = str(it.get("name", "")).strip()[:60]
        if not name:
            continue
        items.append({
            "rank": rank, "name": name,
            "tag": (str(it.get("tag") or name).strip()[:18]).upper(),
            "blurb": str(it.get("blurb", "")).strip()[:300],
            "query": str(it.get("query") or f"{name} football skills goals").strip()[:120],
        })
    items.sort(key=lambda x: -x["rank"])   # countdown: high rank first
    if not items:
        raise RuntimeError("The countdown generator returned no items.")
    return CountdownScript(
        title=str(data.get("title", topic))[:80],
        intro=str(data.get("intro", "")).strip() or f"The top {count}.",
        items=items,
    )


def trending_subjects(
    count: int,
    *,
    api_key: str,
    model: str,
    niche: str = DEFAULT_NICHE,
    avoid: list[str] | None = None,
) -> list[str]:
    """Web-search for `count` trending World Cup subjects (players/countries/
    storylines) to auto-fill a news batch. Best-effort: returns [] on failure."""
    if count <= 0:
        return []
    avoid = [a for a in (avoid or []) if a]
    avoid_line = ("Do NOT repeat these: " + "; ".join(avoid) + ".\n") if avoid else ""
    prompt = (
        f"Use web search to find the {count} most talked-about, currently-trending "
        f"stories in {niche} this week — specific players, national teams, results, "
        f"transfers, injuries, or controversies that are blowing up on social right now. "
        f"{avoid_line}Each must be a specific, scroll-stopping angle, not generic.\n\n"
        f'Return ONLY JSON: {{"subjects": ["<specific subject/angle>", "..."]}}. '
        f"No text outside the JSON."
    )
    try:
        client = _client(api_key)
        resp = client.messages.create(
            model=model, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(raw)
    except Exception:
        return []
    out = [str(t).strip() for t in data.get("subjects", []) if str(t).strip()]
    return out[:count]
