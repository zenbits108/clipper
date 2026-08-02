"""Video -> transcript JSON with word-level timestamps.

Transcripts are cached under <output_dir>/<video-id>/transcript.json so
re-runs skip the (slow) whisper pass unless force=True.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clipper.config import Settings


class TranscribeError(Exception):
    """Raised when transcription fails or the source video is missing."""


def video_id_for(video_path: Path) -> str:
    return Path(video_path).stem


def transcript_cache_path(video_path: Path, settings: Settings) -> Path:
    return settings.output_dir / video_id_for(video_path) / "transcript.json"


def transcribe_video(video_path: Path, settings: Settings, force: bool = False) -> dict:
    """Transcribe `video_path`, returning the cached or freshly-created
    transcript dict (segment- and word-level timestamps).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise TranscribeError(f"Video file not found: {video_path}")

    cache_path = transcript_cache_path(video_path, settings)
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    transcript = _run_whisper(video_path, settings)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return transcript


def _run_whisper(video_path: Path, settings: Settings) -> dict:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscribeError(
            "faster-whisper is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    model = WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    segments_iter, info = model.transcribe(str(video_path), word_timestamps=True)

    segments = []
    for seg in segments_iter:
        words = [
            {
                "word": w.word.strip(),
                "start": w.start,
                "end": w.end,
                "probability": w.probability,
            }
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": words,
            }
        )

    return {
        "video_path": str(video_path.resolve()),
        "video_id": video_id_for(video_path),
        "language": info.language,
        "duration": info.duration,
        "model_size": settings.whisper_model_size,
        "segments": segments,
    }
