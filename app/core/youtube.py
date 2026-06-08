"""Upload clips to YouTube (as Shorts) via the YouTube Data API v3.

Setup (one time): create a Google Cloud project, enable "YouTube Data API v3",
create an OAuth client of type "Desktop app", and download its JSON to
bin/youtube_client_secret.json. The first upload opens a browser for consent and
caches the token in bin/youtube_token.json. See the README for full steps.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import BASE_DIR

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET_FILE = BASE_DIR / "bin" / "youtube_client_secret.json"
TOKEN_FILE = BASE_DIR / "bin" / "youtube_token.json"


class YouTubeNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    """True if the OAuth client secret has been provided."""
    return CLIENT_SECRET_FILE.exists()


def is_authorized() -> bool:
    """True if we already have a cached user token."""
    return TOKEN_FILE.exists()


def _load_credentials():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow

    if not is_configured():
        raise YouTubeNotConfigured(
            f"Missing {CLIENT_SECRET_FILE.name}. Create a Google Cloud OAuth "
            "desktop client and save its JSON there (see README)."
        )

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        # First-time consent: opens a browser, captures the redirect locally.
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
        creds = flow.run_local_server(port=0)

    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    return creds


def build_description(title: str, hook: str, hashtags: list[str]) -> str:
    parts = [title.strip()]
    if hook and hook.strip():
        parts.append(hook.strip())
    tagline = " ".join("#" + t for t in hashtags) if hashtags else ""
    tagline = (tagline + " #Shorts").strip()
    parts.append(tagline)
    return "\n\n".join(p for p in parts if p)


def upload(
    video_path: str | Path,
    *,
    title: str,
    hook: str = "",
    hashtags: list[str] | None = None,
    privacy: str = "private",
    publish_at: str | None = None,
) -> dict:
    """Upload a video and return {'id', 'url'}. Raises on failure.

    If `publish_at` (RFC3339 UTC, e.g. 2026-06-01T17:00:00Z) is given, the video
    is uploaded private and YouTube auto-publishes it publicly at that time.
    """
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    hashtags = hashtags or []
    creds = _load_credentials()
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)

    # YouTube titles are capped at 100 chars and may not contain < or >.
    safe_title = title.replace("<", "").replace(">", "")[:100]
    body = {
        "snippet": {
            "title": safe_title,
            "description": build_description(title, hook, hashtags),
            "tags": hashtags[:15],
            "categoryId": "22",  # People & Blogs
        },
        "status": {
            "privacyStatus": privacy if privacy in {"public", "unlisted", "private"} else "private",
            "selfDeclaredMadeForKids": False,
        },
    }
    # Scheduled publish: must be uploaded private; YouTube flips it public at publishAt.
    if publish_at:
        body["status"]["privacyStatus"] = "private"
        body["status"]["publishAt"] = publish_at

    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()
    vid = response["id"]
    return {"id": vid, "url": f"https://youtube.com/shorts/{vid}", "publish_at": publish_at}
