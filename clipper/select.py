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
- The "hook" is one short line the speaker says. The clip's start/end are
  NOT just that line's timestamps — they must span the full continuous
  excerpt around it. Find the hook first, then expand outward (earlier for
  setup, later for the payoff or follow-through) until the excerpt is a
  complete, self-contained thought that reaches at least {min_s:.0f}
  seconds, stopping at a natural sentence/thought boundary no later than
  {max_s:.0f} seconds. If start/end tightly bound only the hook sentence
  itself, the clip is wrong — it is almost always far too short. Every
  clip must satisfy {min_s:.0f}-{max_s:.0f} seconds; this is a hard
  requirement, not a suggestion.
- start and end timestamps must fall within the transcript below — do not
  invent timestamps outside the video's duration.
- Return at most {max_candidates} candidates.
- Each clip must work as a self-contained moment: real setup, the hook,
  and a payoff or continuation, no dangling references to unseen context.
- "hook" in the JSON output is on-screen text, not a transcript quote:
  rewrite the speaker's point as a tight, grammatical line — cut filler
  words, false starts, and run-ons. Aim for under 12 words. It should read
  like a caption someone wrote on purpose, not a stretch of raw speech.

Output format is a hard requirement: respond with ONLY a JSON array. No
prose before or after it, no headers, no bullet points, no bold text, no
markdown fences, no explanation of what you're about to do. The very
first character of your response must be [ and the very last must be ].
Each element:
{{"start": <seconds, float — start of the FULL excerpt, not the hook line>, "end": <seconds, float — end of the FULL excerpt; end minus start must be {min_s:.0f}-{max_s:.0f}>, "hook": "<tight, cleaned-up on-screen hook line, NOT a verbatim transcript quote>", "title": "<short YouTube Shorts title>", "score": <0-10 float>, "reason": "<one sentence on why this works>"}}

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


MAX_FORMAT_ATTEMPTS = 2  # some models intermittently ignore "JSON only" and answer in prose


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

    raw = None
    parse_error = None
    for _ in range(MAX_FORMAT_ATTEMPTS):
        response = provider.complete(prompt)
        try:
            raw = _extract_json_array(response)
            parse_error = None
            break
        except SelectError as exc:
            parse_error = exc
    if parse_error is not None:
        raise parse_error

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
