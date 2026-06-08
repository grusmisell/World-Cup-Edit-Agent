"""Use Claude to select the most viral / engaging clip-worthy moments."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .transcriber import Transcript


@dataclass
class ClipPlan:
    start: float
    end: float
    title: str
    hook: str
    score: int
    reason: str
    hashtags: list[str] = field(default_factory=list)
    name_tag: str = ""        # football edit: on-screen player/country label
    trend_score: int = 0
    matched_trend: str = ""
    trend_hashtags: list[str] = field(default_factory=list)
    captions: list[dict] = field(default_factory=list)


SYSTEM_PROMPT = """You are an elite short-form video editor for TikTok, Instagram Reels, and YouTube Shorts. You watch long-form content (podcasts, interviews, talks, streams) and extract the moments most likely to go viral.
{niche_clause}
A great clip:
- Is a complete, self-contained thought that makes sense without the rest of the video.
- Opens with a strong hook in the first 3 seconds (a bold claim, question, surprising fact, or emotional beat).
- Has a clear payoff, insight, story, or punchline.
- Stands on its own; no "as I said earlier" dependence on missing context.

You will be given a timestamped transcript. Choose the best clips. Each clip MUST be between {min_s} and {max_s} seconds long. Clips must NOT overlap each other. Align start/end to natural sentence boundaries from the transcript.

Return ONLY a JSON object of the form:
{{"clips": [{{"start": <seconds, number>, "end": <seconds, number>, "title": "<punchy title, max 60 chars>", "hook": "<the opening line/hook>", "score": <0-100 virality score>, "reason": "<one sentence why this will perform>", "hashtags": ["<3-5 relevant hashtags without the # symbol, lowercase, no spaces>"]}}]}}

Order clips from highest to lowest score. Do not include any text outside the JSON object."""


HIGHLIGHTS_PROMPT = """You are an elite football (soccer) highlights editor making viral short-form edits for TikTok, Instagram Reels, and YouTube Shorts from match footage and highlight reels.
{niche_clause}
You are given the timestamped COMMENTARY transcript of a football video. The commentary is your signal for where the big moments are: goals, near-misses, incredible saves, skill moves, red cards, penalties, last-minute drama, and crowd-erupting moments. Excited, loud, drawn-out commentary ("GOOOAL", "what a strike", "unbelievable") marks the peak — start the clip a couple of seconds BEFORE the build-up so the moment lands, and end shortly after the celebration.

Pick the best self-contained moments. Each clip MUST be between {min_s} and {max_s} seconds long and must NOT overlap another. Favor goals and jaw-dropping plays. Write titles like a hype football edit ("MBAPPE IS UNREAL 🔥") and a short on-screen "name_tag" (the player's surname or the country) for the edit.

Return ONLY a JSON object of the form:
{{"clips": [{{"start": <seconds, number>, "end": <seconds, number>, "title": "<hype title, max 60 chars>", "hook": "<what happens in the moment>", "name_tag": "<player surname or country, 2-18 chars>", "score": <0-100 virality score>, "reason": "<one sentence why this moment slaps>", "hashtags": ["<4-6 football hashtags without the # symbol, lowercase, no spaces>"]}}]}}

Order clips from highest to lowest score. Do not include any text outside the JSON object."""


def _niche_clause(niche: str) -> str:
    niche = (niche or "").strip()
    if not niche:
        return ""
    return (
        f"\nThe creator's niche is **{niche}**. Strongly prefer the moments most "
        f"relevant and valuable to a {niche} audience, and tailor every title, hook, "
        f"and hashtag to that audience. Only include an off-niche moment if it is "
        f"exceptionally viral on its own.\n"
    )


def _build_transcript_text(transcript: Transcript) -> str:
    lines = []
    for seg in transcript.segments:
        lines.append(f"[{seg.start:.1f} - {seg.end:.1f}] {seg.text.strip()}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:500]}")
    return json.loads(text[start : end + 1])


def select_clips(
    transcript: Transcript,
    *,
    api_key: str,
    model: str,
    clip_count: int,
    min_seconds: int,
    max_seconds: int,
    niche: str = "",
    highlight_mode: bool = False,
) -> list[ClipPlan]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    base_prompt = HIGHLIGHTS_PROMPT if highlight_mode else SYSTEM_PROMPT
    system = base_prompt.format(
        min_s=min_seconds, max_s=max_seconds, niche_clause=_niche_clause(niche)
    )
    transcript_text = _build_transcript_text(transcript)

    niche_line = f"Creator niche: {niche.strip()}.\n" if niche and niche.strip() else ""
    kind = "football moments" if highlight_mode else "clips"
    user = (
        f"Select the {clip_count} best {kind} from this transcript. "
        f"Each clip between {min_seconds} and {max_seconds} seconds.\n"
        f"{niche_line}\n"
        f"TRANSCRIPT:\n{transcript_text}"
    )

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    data = _extract_json(raw)

    plans: list[ClipPlan] = []
    for c in data.get("clips", []):
        try:
            start = float(c["start"])
            end = float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        raw_tags = c.get("hashtags") or []
        hashtags = [
            str(t).lstrip("#").strip().replace(" ", "")
            for t in raw_tags
            if str(t).strip()
        ][:5]
        plans.append(
            ClipPlan(
                start=start,
                end=end,
                title=str(c.get("title", "Clip"))[:80],
                hook=str(c.get("hook", "")),
                score=int(c.get("score", 0)),
                reason=str(c.get("reason", "")),
                hashtags=hashtags,
                name_tag=str(c.get("name_tag", ""))[:18],
            )
        )
    return plans


def snap_to_words(plan: ClipPlan, transcript: Transcript, pad: float = 0.15) -> ClipPlan:
    """Nudge start/end to the nearest word boundaries so we don't cut mid-word."""
    words = transcript.words
    if not words:
        return plan

    starts = [w.start for w in words]
    ends = [w.end for w in words]

    start = min(starts, key=lambda s: abs(s - plan.start))
    end = max((e for e in ends if e <= plan.end + 0.5), default=plan.end)

    plan.start = max(0.0, start - pad)
    plan.end = end + pad
    return plan


def _clean_tags(raw) -> list[str]:
    return [
        str(t).lstrip("#").strip().replace(" ", "")
        for t in (raw or [])
        if str(t).strip()
    ][:5]


def trend_match(
    plans: list[ClipPlan],
    niche: str,
    *,
    api_key: str,
    model: str,
) -> list[dict]:
    """Find what's trending now in the niche (via web search) and map each clip to
    the best-fitting trend. Mutates each plan's trend_score / matched_trend /
    trend_hashtags in place and returns the list of discovered trends.

    Returns [] if there's nothing to do; never raises on a bad model response.
    """
    niche = (niche or "").strip()
    if not plans or not niche:
        return []

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    clips_desc = "\n".join(
        f"{i}. {p.title} — hook: {p.hook}" for i, p in enumerate(plans)
    )
    prompt = (
        f"Use web search to find what is trending RIGHT NOW (this week) in **{niche}** "
        f"short-form video on TikTok, Instagram Reels, and YouTube Shorts: trending "
        f"topics, angles, formats, and hashtags. Then map each clip below to the single "
        f"best-fitting current trend, with a 0-100 trend-fit score and 3-5 trend-optimized "
        f"hashtags (no # symbol, lowercase, no spaces).\n\n"
        f"CLIPS:\n{clips_desc}\n\n"
        f'Return ONLY a JSON object: {{"trends": [{{"trend": "<short name>", "why": '
        f'"<one line>", "hashtags": ["..."]}}], "matches": [{{"clip": <index number>, '
        f'"trend": "<short trend name>", "score": <0-100>, "hashtags": ["..."]}}]}}. '
        f"Do not include any text outside the JSON object."
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 4}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(raw)
    except Exception:
        return []

    for m in data.get("matches", []):
        i = m.get("clip")
        if isinstance(i, bool) or not isinstance(i, int) or not (0 <= i < len(plans)):
            continue
        plans[i].matched_trend = str(m.get("trend", ""))[:80]
        try:
            plans[i].trend_score = max(0, min(100, int(m.get("score", 0))))
        except (TypeError, ValueError):
            plans[i].trend_score = 0
        plans[i].trend_hashtags = _clean_tags(m.get("hashtags"))

    trends = []
    for t in data.get("trends", []):
        trends.append({
            "trend": str(t.get("trend", ""))[:120],
            "why": str(t.get("why", ""))[:200],
            "hashtags": _clean_tags(t.get("hashtags")),
        })
    return trends


def generate_captions(
    plans: list[ClipPlan],
    niche: str,
    *,
    api_key: str,
    model: str,
    variants: int = 3,
) -> None:
    """Write several distinct, post-ready captions (with hashtags) per clip so the
    same clip can be posted repeatedly without looking duplicated. Mutates each
    plan's `captions` in place; best-effort (leaves them empty on failure)."""
    if not plans:
        return

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    niche_line = f"Creator niche: {niche.strip()}.\n" if niche and niche.strip() else ""
    clips_desc = "\n".join(
        f"{i}. title: {p.title} | hook: {p.hook}"
        + (f" | on-trend: {p.matched_trend}" if p.matched_trend else "")
        for i, p in enumerate(plans)
    )
    prompt = (
        f"For each clip below, write {variants} DISTINCT social post captions — the "
        f"description text posted under a short-form video (NOT on-screen text). Each "
        f"caption is 1-2 punchy sentences that hook the viewer and invite engagement "
        f"(a question, bold statement, or call to action), plus 3-6 fitting hashtags. "
        f"Vary the angle and wording across the {variants} variants so the same clip can "
        f"be posted multiple times without looking duplicated.\n{niche_line}\n"
        f"CLIPS:\n{clips_desc}\n\n"
        f'Return ONLY JSON: {{"captions": [{{"clip": <index number>, "variants": '
        f'[{{"caption": "<text>", "hashtags": ["..."]}}]}}]}}. No text outside the JSON.'
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(b.text for b in resp.content if b.type == "text")
        data = _extract_json(raw)
    except Exception:
        return

    for item in data.get("captions", []):
        i = item.get("clip")
        if isinstance(i, bool) or not isinstance(i, int) or not (0 <= i < len(plans)):
            continue
        out = []
        for v in item.get("variants", []):
            cap = str(v.get("caption", "")).strip()
            if cap:
                out.append({"caption": cap[:500], "hashtags": _clean_tags(v.get("hashtags"))})
        if out:
            plans[i].captions = out
