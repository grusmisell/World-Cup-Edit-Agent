# ⚽ World Cup Edit Agent

Turn World Cup football into viral, vertical, short-form edits — automatically.
A sibling of the Clipping Agent, tuned for football content, with three modes:

- **🎬 Highlights edit** — paste a highlights/match video (URL or file). Local
  Whisper transcribes the commentary, Claude finds the goals & big moments, and
  each is rendered as a **beat-synced football edit** (cinematic teal/orange
  grade + zoom punches on the beat + big player-name text) or a clean captioned
  clip.
- **📰 News edit** — type players / countries / storylines (or let it auto-pick
  what's trending). Claude **web-searches the latest real news**, writes a
  hook-first script, voices it with free neural TTS, and composites it over
  football B-roll with synced captions.
- **🖼️ Photo edit** — upload player photos; they're cut to the beat with Ken
  Burns motion + the cinematic grade over your music.

**Shared engine:** local whisper.cpp transcription, bundled ffmpeg, edge-tts
voices, Claude analysis/web-search, a numpy beat detector, per-platform posting
chips, and YouTube auto-upload — reused from the Clipping Agent. The only paid
dependency is the Anthropic API.

> **Copyright note:** posting raw broadcast match footage (e.g. pulled via a
> YouTube URL) risks copyright strikes/demonetization on TikTok & YouTube — FIFA
> and broadcasters claim it. The **transformative** modes (heavy edit style,
> your own commentary, news + B-roll, photo edits) are what hold up long-term.

---

## Quick start (Windows)

1. **Install Python deps** (one time):
   ```
   pip install -r requirements.txt
   ```
   Transcription, the whisper model, and ffmpeg are **reused from the sibling
   Clipping Agent install** (`..\Clipping Agent\bin`) — nothing extra to
   download. Override the location with `SHARED_BIN_DIR` in `.env`.

2. **Add your API key:** `.env` already has `ANTHROPIC_API_KEY` and
   `PEXELS_API_KEY` (copy from `.env.example` on a fresh clone).

3. **Run it:** double-click `start.bat`, or:
   ```
   python run.py
   ```
   Your browser opens to http://127.0.0.1:8001 (port 8001 so it runs alongside
   the Clipping Agent on 8000).

4. **First-time assets:** click **⬇ Fetch football B-roll (Pexels)** (news mode)
   to grab copyright-safe stadium/crowd loops into `data/backgrounds/`, and drop
   a royalty-free music bed into `data/music/` for the beat-synced edits.

---

## How it works

| Stage | Tool | Notes |
|-------|------|-------|
| Download | `yt-dlp` | YouTube + most video sites. Or upload a local file. |
| Transcribe | `whisper.cpp` | Bundled native binary. Runs locally on CPU. Word-level timestamps. |
| Pick clips | Claude | Scores moments for virality, returns titles + hooks. |
| Cut + reframe | `ffmpeg` + OpenCV | Face-aware crop to your chosen aspect (9:16 / 4:5 / 1:1 / 16:9 / original). |
| Captions | `ffmpeg` (ASS) | Word-timed, burned-in, viral phrase-at-a-time style. |

ffmpeg is auto-located: it uses a system ffmpeg if present, otherwise the binary
bundled with the `imageio-ffmpeg` pip package (installed automatically).

---

## Sharing clips

Every clip card has a **Share** row:

- **Copy caption** — copies the title + suggested hashtags to your clipboard.
- **TikTok / YouTube / Instagram** — copies the caption and opens that platform's
  upload page in a new tab. Download the clip, drop it in, paste the caption.

### Optional: one-click "Publish to YouTube"

If you set up a Google OAuth client, a red **⤴ Publish to YouTube** button appears
on each clip that uploads it directly as a Short — plus a **Publish all to YouTube**
button in the results header to upload every clip in one go. One-time setup:

1. Go to https://console.cloud.google.com/ and create (or pick) a project.
2. **APIs & Services → Library →** enable **YouTube Data API v3**.
3. **APIs & Services → OAuth consent screen →** configure it (External is fine);
   add your own Google account under **Test users**.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID →**
   application type **Desktop app**. Download the JSON.
5. Save that file as **`bin/youtube_client_secret.json`** in this project.
6. Restart the app. The first time you click *Publish to YouTube*, a browser opens
   to authorize your channel (token is cached in `bin/youtube_token.json`).

Uploads default to **private** (safe for testing). Change with `YOUTUBE_PRIVACY`
in `.env` (`private` | `unlisted` | `public`). Note: YouTube's API quota allows
~6 uploads/day on a fresh project.

---

## Settings (`.env`)

- `ANTHROPIC_API_KEY` — required.
- `CLAUDE_MODEL` — model used to select clips.
- `WHISPER_MODEL` — `tiny`/`base`/`small`/`medium`/`large`. `base` is a good
  CPU default; larger models are more accurate but slower. The matching
  `ggml-<size>.bin` must exist in `bin/models/` (`base` ships with the project).
- `DEFAULT_CLIP_COUNT`, `MIN_CLIP_SECONDS`, `MAX_CLIP_SECONDS` — clip tuning.

---

## Project layout

```
app/
  main.py            FastAPI app + API routes
  jobs.py            Background pipeline orchestration + progress tracking
  config.py          Settings (.env)
  core/
    downloader.py    yt-dlp / local file
    transcriber.py   whisper.cpp (bundled binary), word timestamps
    analyzer.py      Claude picks viral moments
    clipper.py       cut + face-aware 9:16 reframe + caption burn
    captions.py      ASS subtitle generation
    ffmpeg_utils.py  ffmpeg/ffprobe location + helpers
  static/index.html  Web UI
data/
  uploads/           Uploaded/downloaded source videos
  jobs/<id>/         Per-job working dir, job.json state, clips/
  backgrounds/       Loop videos for split-screen / voiceover backgrounds
  music/             Royalty-free tracks for the voiceover music bed
```

---

## Voiceover moods

In AI Voiceover mode, a **Mood** ties the script's delivery tone, the TTS voice,
the music bed, and the background video together so they match (no upbeat track
under a monotone read):

- **Auto** (default) rotates Energetic / Chill / Cinematic across the batch and
  picks a matching voice, music, and background for each clip.
- **Energetic / Chill / Cinematic** lock the whole batch to one mood.
- **Manual** uses the voice & music pickers instead.

Music and backgrounds are matched to a mood by **filename keyword**, so name your
files accordingly — e.g. `upbeat`/`pop` (energetic), `lofi`/`chill` (chill),
`ambient`/`cinematic` (cinematic). If nothing matches, all files are eligible and
simply rotate across the batch.

### Free stock backgrounds (Pexels)

Instead of supplying your own loops, pull real, copyright-safe vertical footage
from [Pexels](https://www.pexels.com/license/) (free for commercial use, no
attribution):

1. Get a free API key at <https://www.pexels.com/api/>.
2. Add `PEXELS_API_KEY=your_key` to `.env`.
3. Click **"⬇ Fetch stock backgrounds (Pexels)"** under the Background dropdown
   (voiceover tab), or `POST /api/pexels/fetch` (`per_mood=1..3`).

Downloads land in `data/backgrounds/` named `pexels-<mood>-<keyword>.mp4` so the
mood-matcher picks them automatically. Resolution is capped (~1080–1920 tall) to
keep disk use reasonable.

---

## Roadmap (Phase 2 — for the commercial version)

- **Auto-posting** to TikTok / Instagram Reels / YouTube Shorts (needs each
  platform's API + app review).
- **Multi-tenant accounts + billing** (turn this into a hosted SaaS).
- **Brand kits**: per-client caption fonts/colors, logo watermark, intro/outro.
- **B-roll & zoom effects**, multi-speaker tracking, emoji auto-placement.
- **GPU transcription** + a job queue (Celery/RQ) for concurrent client jobs.
```
