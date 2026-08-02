"""Transcript -> LLM -> ranked clip candidates JSON.

Always calls the configured provider (no caching): phase 1's job is tuning
selection-prompt quality, and caching would hide the effect of prompt edits.
Output is written to <output_dir>/<video-id>/candidates.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from clipper.config import ChannelConfig, Settings
from clipper.llm import get_provider
from clipper.transcribe import video_id_for

PROMPT_TEMPLATE = """{channel_prompt}

Tone guidance: {tone}

You are given a timestamped transcript of a long-form video. Identify the
strongest standalone clip candidates for short-form vertical video (YouTube
Shorts / TikTok / Reels).

Rules:
- Each clip must be between {min_s:.0f} and {max_s:.0f} seconds long.
- start and end timestamps must fall within the transcript below — do not
  invent timestamps outside the video's duration.
- Return at most {max_candidates} candidates.
- Each clip must work as a self-contained moment: a hook, a payoff, no
  dangling references to unseen context.

Respond with ONLY a JSON array (no prose, no markdown fences). Each element:
{{"start": <seconds, float>, "end": <seconds, float>, "hook": "<on-screen hook line>", "title": "<short YouTube Shorts title>", "score": <0-10 float>, "reason": "<one sentence on why this works>"}}

Transcript:
{transcript_block}
"""


class SelectError(Exception):
    """Raised when the LLM response can't be parsed into valid candidates."""


def _transcript_block(transcript: dict) -> str:
    return "\n".join(
        f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}" for seg in transcript["segments"]
    )


def build_prompt(transcript: dict, channel: ChannelConfig) -> str:
    return PROMPT_TEMPLATE.format(
        channel_prompt=channel.selection.prompt.strip(),
        tone=channel.selection.tone or "(none specified)",
        min_s=channel.selection.clip_length.min_seconds,
        max_s=channel.selection.clip_length.max_seconds,
        max_candidates=channel.selection.max_candidates,
        transcript_block=_transcript_block(transcript),
    )


def _extract_json_array(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise SelectError(f"Could not parse JSON array from LLM response: {exc}") from exc
    raise SelectError(f"LLM response did not contain a JSON array:\n{text[:500]}")


def _validate_candidates(raw: Any, transcript: dict, channel: ChannelConfig) -> list:
    if not isinstance(raw, list):
        raise SelectError(f"Expected a JSON array of candidates, got: {type(raw).__name__}")

    duration = transcript.get("duration", float("inf"))
    min_s = channel.selection.clip_length.min_seconds
    max_s = channel.selection.clip_length.max_seconds

    candidates = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
            hook = str(item["hook"])
            title = str(item["title"])
            score = float(item.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue

        if start < 0 or end <= start or end > duration + 1.0:
            continue
        length = end - start
        if length < min_s - 1.0 or length > max_s + 1.0:
            continue

        candidates.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "hook": hook.strip(),
                "title": title.strip(),
                "score": round(max(0.0, min(10.0, score)), 2),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[: channel.selection.max_candidates]
    for i, c in enumerate(candidates, start=1):
        c["id"] = f"clip-{i:02d}"
    return candidates


def select_clips(
    video_path: Path,
    transcript: dict,
    channel: ChannelConfig,
    settings: Settings,
) -> list:
    """Ask the configured LLM to rank clip candidates, validate them, and
    write the result to <output_dir>/<video-id>/candidates.json.
    """
    video_path = Path(video_path)
    provider = get_provider(channel.selection.provider, settings, channel.selection.model)
    prompt = build_prompt(transcript, channel)
    response = provider.complete(prompt)
    raw = _extract_json_array(response)
    candidates = _validate_candidates(raw, transcript, channel)

    output = {
        "video_path": str(video_path.resolve()),
        "video_id": video_id_for(video_path),
        "channel": channel.name,
        "provider": channel.selection.provider,
        "candidates": candidates,
    }

    out_path = settings.output_dir / video_id_for(video_path) / "candidates.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    return candidates
