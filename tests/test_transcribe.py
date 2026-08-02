import json
from pathlib import Path

import pytest

from clipper.config import load_settings
from clipper.transcribe import TranscribeError, transcribe_video, transcript_cache_path


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.output_dir = tmp_path
    return s


def test_missing_video_raises(settings):
    with pytest.raises(TranscribeError):
        transcribe_video(Path("does_not_exist.mp4"), settings)


def test_cache_hit_skips_whisper(tmp_path, settings, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake video bytes")

    cache_path = transcript_cache_path(video_path, settings)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {
        "video_path": str(video_path.resolve()),
        "video_id": "clip",
        "language": "en",
        "duration": 12.3,
        "model_size": "large-v3",
        "segments": [],
    }
    cache_path.write_text(json.dumps(cached))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("whisper should not run on a cache hit")

    monkeypatch.setattr("clipper.transcribe._run_whisper", fail_if_called)

    result = transcribe_video(video_path, settings)
    assert result == cached
