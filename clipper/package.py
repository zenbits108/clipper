"""Package a finished clip.mp4 into a YouTube-Studio-ready folder.

Writes meta.json (title <=100 chars, description, hashtags, source
timestamp) and thumb.jpg -- a frame grabbed at the clip's "hook moment",
found by matching the LLM-written (paraphrased) hook line against the
clip's own transcript words rather than trusting a timestamp the LLM
never actually produced.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from clipper.captions import words_in_clip
from clipper.config import ChannelConfig

TITLE_MAX_CHARS = 100


class PackageError(Exception):
    """Raised when thumbnail extraction or metadata writing fails."""


def _truncate_title(title: str) -> str:
    title = title.strip()
    if len(title) <= TITLE_MAX_CHARS:
        return title
    truncated = title[:TITLE_MAX_CHARS].rsplit(" ", 1)[0]
    return truncated or title[:TITLE_MAX_CHARS]


def find_hook_moment(hook: str, words: list) -> float:
    """Best-effort clip-relative timestamp where the hook is actually
    spoken: the transcript span (same word-count as the hook) with the
    highest token overlap against the hook text. Falls back to the first
    word's timestamp, or 0.0 if there are no words at all.
    """
    if not words:
        return 0.0
    hook_tokens = set(re.findall(r"[a-z0-9']+", hook.lower()))
    if not hook_tokens:
        return words[0]["start"]

    window = max(3, len(hook_tokens))
    best_score = -1
    best_time = words[0]["start"]
    for i in range(len(words)):
        span = words[i : i + window]
        span_tokens = set(re.findall(r"[a-z0-9']+", " ".join(w["word"] for w in span).lower()))
        score = len(hook_tokens & span_tokens)
        if score > best_score:
            best_score = score
            best_time = span[0]["start"]
    return best_time


def build_meta(candidate: dict, cut_meta: dict, channel: ChannelConfig, source_video_path: str) -> dict:
    hashtags = list(channel.hashtags)
    description_parts = [candidate["hook"]]
    if hashtags:
        description_parts.append(" ".join(hashtags))

    return {
        "title": _truncate_title(candidate["title"]),
        "description": "\n\n".join(description_parts),
        "hashtags": hashtags,
        "source": {
            "video": Path(source_video_path).name,
            "video_path": source_video_path,
            "start": cut_meta["start"],
            "end": cut_meta["end"],
        },
        "channel": channel.name,
        "clip_id": candidate["id"],
        "score": candidate.get("score"),
    }


def _ffmpeg_thumbnail_command(clip_path: Path, timestamp: float, output_path: Path) -> list:
    return [
        "ffmpeg",
        "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", str(clip_path),
        "-frames:v", "1",
        "-update", "1",
        "-q:v", "2",
        str(output_path),
    ]


def package_clip(
    clip_dir: Path,
    transcript: dict,
    candidate: dict,
    cut_meta: dict,
    channel: ChannelConfig,
    source_video_path: str,
) -> dict:
    """Write <clip_dir>/meta.json and <clip_dir>/thumb.jpg for a finished
    clip.mp4. Returns the metadata dict that was written.
    """
    clip_dir = Path(clip_dir)
    clip_path = clip_dir / "clip.mp4"
    if not clip_path.exists():
        raise PackageError(f"clip.mp4 not found: {clip_path}")

    words = words_in_clip(transcript, cut_meta)
    hook_time = find_hook_moment(candidate["hook"], words)

    thumb_path = clip_dir / "thumb.jpg"
    cmd = _ffmpeg_thumbnail_command(clip_path, hook_time, thumb_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PackageError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-4000:]}")

    meta = build_meta(candidate, cut_meta, channel, source_video_path)
    meta["thumbnail_time"] = round(hook_time, 3)
    (clip_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta
