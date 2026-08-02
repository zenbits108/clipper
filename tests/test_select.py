import json
from pathlib import Path

import pytest

from clipper.config import load_channel, load_settings
from clipper import select


class StubProvider:
    def __init__(self, response_text):
        self.response_text = response_text

    def complete(self, prompt):
        return self.response_text


FAKE_TRANSCRIPT = {
    "video_path": "/fake/video.mp4",
    "video_id": "video",
    "language": "en",
    "duration": 120.0,
    "segments": [
        {"start": 0.0, "end": 5.0, "text": "Hello and welcome to the show.", "words": []},
        {"start": 5.0, "end": 40.0, "text": "Here is a surprising fact you didn't know.", "words": []},
        {"start": 40.0, "end": 120.0, "text": "And that's how it all wrapped up.", "words": []},
    ],
}


@pytest.fixture
def settings(tmp_path):
    s = load_settings()
    s.output_dir = tmp_path
    return s


@pytest.fixture
def channel():
    return load_channel("example_channel")


def test_extract_json_array_handles_markdown_fences():
    text = '```json\n[{"start": 1, "end": 2}]\n```'
    assert select._extract_json_array(text) == [{"start": 1, "end": 2}]


def test_select_clips_filters_out_of_bounds_and_writes_json(monkeypatch, settings, channel, tmp_path):
    response = json.dumps(
        [
            {
                "start": 5.0,
                "end": 40.0,
                "hook": "Did you know...",
                "title": "A surprising fact",
                "score": 9,
                "reason": "strong hook",
            },
            {"start": 0.0, "end": 1.0, "hook": "too short", "title": "short", "score": 5},
            {"start": 40.0, "end": 500.0, "hook": "too long", "title": "long", "score": 5},
        ]
    )
    monkeypatch.setattr(select, "get_provider", lambda *a, **k: StubProvider(response))

    candidates = select.select_clips(Path("video.mp4"), FAKE_TRANSCRIPT, channel, settings)

    assert len(candidates) == 1
    assert candidates[0]["id"] == "clip-01"
    assert candidates[0]["title"] == "A surprising fact"

    out_path = tmp_path / "video" / "candidates.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text())
    assert saved["channel"] == "example_channel"
    assert len(saved["candidates"]) == 1


def test_select_clips_raises_on_non_json_response(monkeypatch, settings, channel):
    monkeypatch.setattr(select, "get_provider", lambda *a, **k: StubProvider("not json at all"))
    with pytest.raises(select.SelectError):
        select.select_clips(Path("video.mp4"), FAKE_TRANSCRIPT, channel, settings)
