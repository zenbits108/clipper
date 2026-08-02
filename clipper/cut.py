"""Frame-accurate clip extraction.

Cuts snap to sentence boundaries derived from word-level transcript data
(never clip a word mid-sentence), get ~0.3s of silence padding on each
side (clamped to neighboring words so padding never bleeds into another
sentence), and are always re-encoded via ffmpeg/NVENC — never
stream-copied — so cut points land on exact frames.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PAD_SECONDS = 0.3
AUDIO_FADE_SECONDS = 0.02


class CutError(Exception):
    """Raised when a cut window can't be computed or ffmpeg fails."""


def _flat_words(transcript: dict) -> list:
    words = [w for seg in transcript["segments"] for w in seg.get("words", [])]
    words.sort(key=lambda w: w["start"])
    return words


def _sentence_boundaries(transcript: dict) -> list:
    """Times that are safe to cut on: video start/end, plus every sentence
    start/end inferred from terminal punctuation (. ! ?) in the word-level
    transcript. Falls back to whisper segment boundaries if there's no
    word-level data or no punctuation to key off.
    """
    words = _flat_words(transcript)
    duration = transcript.get("duration")

    boundaries = {0.0}
    if duration is not None:
        boundaries.add(float(duration))

    if not words:
        for seg in transcript["segments"]:
            boundaries.add(seg["start"])
            boundaries.add(seg["end"])
        return sorted(boundaries)

    found_punctuation = False
    for i, w in enumerate(words):
        text = w["word"].strip()
        if text and text[-1] in ".!?":
            found_punctuation = True
            boundaries.add(w["end"])
            if i + 1 < len(words):
                boundaries.add(words[i + 1]["start"])

    if not found_punctuation:
        for seg in transcript["segments"]:
            boundaries.add(seg["start"])
            boundaries.add(seg["end"])

    boundaries.add(words[0]["start"])
    boundaries.add(words[-1]["end"])
    return sorted(boundaries)


def _snap_start(boundaries: list, req_start: float) -> float:
    candidates = [b for b in boundaries if b <= req_start]
    return candidates[-1] if candidates else boundaries[0]


def _snap_end(boundaries: list, req_end: float) -> float:
    candidates = [b for b in boundaries if b >= req_end]
    return candidates[0] if candidates else boundaries[-1]


def compute_cut_window(transcript: dict, req_start: float, req_end: float) -> tuple:
    """Snap (req_start, req_end) to sentence boundaries and add padding,
    without ever clipping a word or bleeding into a neighboring sentence.
    """
    if req_end <= req_start:
        raise CutError(f"end ({req_end}) must be after start ({req_start})")

    words = _flat_words(transcript)
    boundaries = _sentence_boundaries(transcript)
    duration = float(transcript.get("duration", boundaries[-1]))

    snapped_start = _snap_start(boundaries, req_start)
    snapped_end = _snap_end(boundaries, req_end)
    if snapped_end <= snapped_start:
        raise CutError(
            f"Snapped window is empty/inverted (start={snapped_start}, end={snapped_end}) "
            f"for requested ({req_start}, {req_end})"
        )

    prev_word_end = max((w["end"] for w in words if w["end"] <= snapped_start), default=0.0)
    next_word_start = min((w["start"] for w in words if w["start"] >= snapped_end), default=duration)

    padded_start = max(0.0, prev_word_end, snapped_start - PAD_SECONDS)
    padded_end = min(duration, next_word_start, snapped_end + PAD_SECONDS)

    return round(padded_start, 3), round(padded_end, 3)


def _ffmpeg_cut_command(video_path: Path, start: float, end: float, output_path: Path) -> list:
    duration = end - start
    fade_out_start = max(0.0, duration - AUDIO_FADE_SECONDS)
    return [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-af", f"afade=t=in:st=0:d={AUDIO_FADE_SECONDS},afade=t=out:st={fade_out_start:.3f}:d={AUDIO_FADE_SECONDS}",
        "-movflags", "+faststart",
        str(output_path),
    ]


def cut_clip(video_path: Path, transcript: dict, candidate: dict, clip_dir: Path) -> dict:
    """Extract a frame-accurate cut for `candidate` (as produced by
    select.py) into <clip_dir>/cut.mp4, writing a <clip_dir>/cut.json
    sidecar with the actual (snapped) window. Returns that metadata dict.
    """
    video_path = Path(video_path)
    clip_dir = Path(clip_dir)
    if not video_path.exists():
        raise CutError(f"Video file not found: {video_path}")

    start, end = compute_cut_window(transcript, candidate["start"], candidate["end"])

    clip_dir.mkdir(parents=True, exist_ok=True)
    output_path = clip_dir / "cut.mp4"

    cmd = _ffmpeg_cut_command(video_path, start, end, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise CutError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-4000:]}")

    meta = {
        "id": candidate.get("id"),
        "start": start,
        "end": end,
        "duration": round(end - start, 3),
        "requested_start": candidate["start"],
        "requested_end": candidate["end"],
        "output_path": str(output_path.resolve()),
    }
    (clip_dir / "cut.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
