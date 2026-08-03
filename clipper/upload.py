"""Upload finished clips to YouTube via the YouTube Data API v3.

Handles the one-time OAuth consent flow (opens a browser, caches the
resulting token per channel so subsequent runs don't need it again),
uploads clip.mp4 with the metadata package.py already wrote, sets the
thumbnail, and records the resulting video id/URL in an upload.json
sidecar so a clip is never silently uploaded twice.

Uploads default to private (per-channel config) -- nothing goes public
without the user reviewing and publishing it manually in YouTube Studio.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clipper.config import ChannelConfig, Settings

TOKEN_DIR = Path(".youtube_tokens")


class UploadError(Exception):
    """Raised when authentication or the upload itself fails."""


def _token_path(channel: ChannelConfig) -> Path:
    return TOKEN_DIR / f"{channel.name}.json"


def get_authenticated_service(channel: ChannelConfig, settings: Settings):
    """Return an authenticated YouTube Data API client, running the
    one-time OAuth consent flow in a browser if no cached (or refreshable)
    token exists for this channel.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise UploadError(
            "google-api-python-client / google-auth-oauthlib not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    client_secrets_path = settings.youtube.client_secrets_path
    if not client_secrets_path.exists():
        raise UploadError(
            f"YouTube OAuth client secrets not found at {client_secrets_path}. "
            "Create an OAuth client (type: Desktop app) in Google Cloud Console, "
            "enable the YouTube Data API v3 for that project, and save the "
            "downloaded JSON at that path. See README.md."
        )

    token_path = _token_path(channel)
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), settings.youtube.scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_path), settings.youtube.scopes
            )
            creds = flow.run_local_server(port=0)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)


def build_video_body(meta: dict, channel: ChannelConfig, privacy_status: Optional[str] = None) -> dict:
    tags = [h.lstrip("#") for h in meta.get("hashtags", [])]
    return {
        "snippet": {
            "title": meta["title"],
            "description": meta["description"],
            "tags": tags,
            "categoryId": channel.upload.category_id,
        },
        "status": {
            "privacyStatus": privacy_status or channel.upload.privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }


def upload_clip(
    clip_dir: Path,
    channel: ChannelConfig,
    settings: Settings,
    privacy_status: Optional[str] = None,
    youtube_client=None,
) -> dict:
    """Upload <clip_dir>/clip.mp4 with <clip_dir>/meta.json's metadata and
    <clip_dir>/thumb.jpg as the thumbnail (if present). Writes
    <clip_dir>/upload.json. Returns the upload metadata dict.
    """
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    clip_dir = Path(clip_dir)
    clip_path = clip_dir / "clip.mp4"
    meta_path = clip_dir / "meta.json"
    thumb_path = clip_dir / "thumb.jpg"

    if not clip_path.exists():
        raise UploadError(f"clip.mp4 not found: {clip_path}")
    if not meta_path.exists():
        raise UploadError(f"meta.json not found: {meta_path}. Run `clipper package` first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    body = build_video_body(meta, channel, privacy_status=privacy_status)

    youtube = youtube_client or get_authenticated_service(channel, settings)

    media = MediaFileUpload(str(clip_path), chunksize=-1, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    try:
        response = None
        while response is None:
            _, response = request.next_chunk()
    except HttpError as exc:
        raise UploadError(f"YouTube upload failed: {exc}") from exc

    video_id = response["id"]

    if thumb_path.exists():
        try:
            youtube.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(thumb_path))
            ).execute()
        except HttpError as exc:
            # Non-fatal: the video itself uploaded fine, just no custom thumbnail.
            pass

    upload_meta = {
        "video_id": video_id,
        "url": f"https://youtube.com/watch?v={video_id}",
        "privacy_status": body["status"]["privacyStatus"],
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (clip_dir / "upload.json").write_text(json.dumps(upload_meta, indent=2), encoding="utf-8")
    return upload_meta
