"""AI voiceover generation: Claude writes a short-form script, edge-tts voices it.

Powers the faceless "doomscroll" mode — AI narration on a topic over a looping
background. The narration audio is later transcribed (whisper.cpp) for synced
captions and composited with a background by clipper.render_voiceover().
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analyzer import _extract_json, _clean_tags

# Free Microsoft neural voices (no API key) — friendly label per voice id.
VOICES = {
    # US — male
    "en-US-AndrewNeural": "Andrew — US, warm male",
    "en-US-BrianNeural": "Brian — US, casual male",
    "en-US-GuyNeural": "Guy — US, deep male",
    "en-US-ChristopherNeural": "Christopher — US, male",
    "en-US-EricNeural": "Eric — US, male",
    "en-US-RogerNeural": "Roger — US, male",
    "en-US-SteffanNeural": "Steffan — US, male",
    # US — female
    "en-US-AriaNeural": "Aria — US, female",
    "en-US-JennyNeural": "Jenny — US, female",
    "en-US-EmmaNeural": "Emma — US, casual female",
    "en-US-AvaNeural": "Ava — US, female",
    "en-US-MichelleNeural": "Michelle — US, female",
    # UK
    "en-GB-RyanNeural": "Ryan — UK, male",
    "en-GB-SoniaNeural": "Sonia — UK, female",
    "en-GB-LibbyNeural": "Libby — UK, female",
    # Australia
    "en-AU-WilliamNeural": "William — AU, male",
    "en-AU-NatashaNeural": "Natasha — AU, female",
}
DEFAULT_VOICE = "en-US-AndrewNeural"
DEFAULT_NICHE = "psychology and human behavior"

# A "mood" ties the script's delivery tone, the TTS voice, the music bed, and the
# background video together so they match (no upbeat track under a monotone read).
# music_kw / bg_kw match against filenames in data/music & data/backgrounds
# (case-insensitive substring); if nothing matches, all files are eligible.
MOODS: dict[str, dict] = {
    "energetic": {
        "label": "Energetic",
        "tone": "High-energy and exciting: punchy, fast-paced delivery with short, "
                "exclamatory sentences that build momentum and hype.",
        "voices": ["en-US-BrianNeural", "en-US-AriaNeural", "en-US-EmmaNeural",
                   "en-US-GuyNeural"],
        "music_kw": ["upbeat", "pop", "energetic", "happy", "dance", "edm", "hype"],
        "bg_kw": ["gameplay", "subway", "minecraft", "parkour", "race", "action",
                  "sport", "satisfying"],
    },
    "chill": {
        "label": "Chill",
        "tone": "Calm, warm and conversational: a relaxed, reflective pace with "
                "smooth, easygoing sentences.",
        "voices": ["en-US-AndrewNeural", "en-US-JennyNeural", "en-US-MichelleNeural",
                   "en-GB-SoniaNeural"],
        "music_kw": ["lofi", "lo-fi", "chill", "relax", "mellow", "study", "calm"],
        "bg_kw": ["rain", "clouds", "lofi", "aesthetic", "cozy", "nature", "ocean",
                  "sunset", "forest"],
    },
    "cinematic": {
        "label": "Cinematic",
        "tone": "Dramatic and contemplative with a weighty, awe-inspiring cadence: "
                "vivid, evocative sentences that feel grand and reflective.",
        "voices": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural",
                   "en-AU-WilliamNeural"],
        "music_kw": ["ambient", "cinematic", "epic", "emotional", "film",
                     "orchestral", "dramatic"],
        "bg_kw": ["space", "cosmos", "cinematic", "drone", "aerial", "mountain",
                  "city", "timelapse", "abstract"],
    },
}
# Order used when rotating moods across a batch in "auto" mode.
MOOD_ORDER = ["energetic", "chill", "cinematic"]


@dataclass
class Script:
    title: str
    text: str
    hashtags: list[str]


def generate_script(
    topic: str,
    niche: str = DEFAULT_NICHE,
    *,
    api_key: str,
    model: str,
    seconds: int = 35,
    tone: str = "",
) -> Script:
    """Write a hook-first short-form voiceover script. If `topic` is empty, Claude
    picks a high-engagement, currently-relevant topic within the niche. `tone`
    steers the delivery style so the script matches the chosen mood."""
    import anthropic

    niche = (niche or DEFAULT_NICHE).strip()
    words = max(40, int(seconds * 2.5))
    topic_line = (
        f"Topic: {topic.strip()}."
        if topic and topic.strip()
        else f"Pick ONE highly engaging, currently-relevant topic within {niche}."
    )
    tone_line = f"- DELIVERY TONE: {tone.strip()}\n" if tone and tone.strip() else ""
    prompt = (
        f"You write faceless short-form video voiceover scripts for a {niche} channel "
        f"(TikTok / Reels / Shorts). {topic_line}\n\n"
        f"Write a ~{seconds}-second narration (~{words} words):\n"
        f"- Open with a 3-second HOOK (bold claim, question, or surprising fact).\n"
        f"{tone_line}"
        f"- Punchy, conversational, retention-optimized. Short sentences. No filler.\n"
        f"- Pay off the hook; end on a loop-y or thought-provoking line.\n"
        f"- 'script' is plain narration ONLY — no scene directions, emojis, or markdown.\n\n"
        f'Return ONLY JSON: {{"title": "<punchy title, max 60 chars>", '
        f'"script": "<the narration text>", "hashtags": ["<3-5 niche hashtags, no # symbol, '
        f'lowercase, no spaces>"]}}. No text outside the JSON.'
    )

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    data = _extract_json(raw)
    text = str(data.get("script", "")).strip()
    if not text:
        raise RuntimeError("The script generator returned an empty script.")
    return Script(
        title=str(data.get("title", topic or niche))[:80],
        text=text,
        hashtags=_clean_tags(data.get("hashtags")),
    )


def generate_topics(
    niche: str,
    count: int,
    *,
    seed: str = "",
    avoid: list[str] | None = None,
    api_key: str,
    model: str,
) -> list[str]:
    """Ask Claude for `count` distinct, scroll-stopping topic ideas in the niche.
    Best-effort: returns [] on failure."""
    if count <= 0:
        return []
    import anthropic

    niche = (niche or DEFAULT_NICHE).strip()
    avoid = [a for a in (avoid or []) if a]
    seed_line = f"Focus on this theme/angle: {seed.strip()}.\n" if seed and seed.strip() else ""
    avoid_line = ("Do NOT repeat or overlap these: " + "; ".join(avoid) + ".\n") if avoid else ""
    prompt = (
        f"List {count} DISTINCT, highly engaging short-form video topics for a {niche} "
        f"channel. Each must be a specific, scroll-stopping angle (a surprising claim, "
        f"counterintuitive fact, or 'why X' question) — not generic. {seed_line}{avoid_line}"
        f'Return ONLY JSON: {{"topics": ["...", "..."]}}. No text outside the JSON.'
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(raw)
    except Exception:
        return []
    out = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]
    return out[:count]


def synthesize(text: str, out_path: Path, *, voice: str = DEFAULT_VOICE) -> Path:
    """Render `text` to an mp3 voiceover with edge-tts (free neural voices)."""
    import asyncio
    import edge_tts

    if voice not in VOICES:
        voice = DEFAULT_VOICE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def _go():
        await edge_tts.Communicate(text, voice).save(str(out_path))

    asyncio.run(_go())
    return out_path
