import json
import subprocess

import pytest

from clipper import package
from clipper.config import load_channel


def _word(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_truncate_title_leaves_short_titles_alone():
    assert package._truncate_title("Zen Isn't About Escaping") == "Zen Isn't About Escaping"


def test_truncate_title_breaks_at_word_boundary_under_the_cap():
    title = "word " * 30  # far over 100 chars
    truncated = package._truncate_title(title)
    assert len(truncated) <= package.TITLE_MAX_CHARS
    assert not truncated.endswith(" ")
    assert truncated == truncated.strip()


def test_find_hook_moment_picks_best_matching_span():
    words = [
        _word("So", 0.0, 0.2),
        _word("anyway,", 0.2, 0.6),
        _word("Zen", 3.0, 3.3),
        _word("is", 3.3, 3.4),
        _word("not", 3.4, 3.6),
        _word("about", 3.6, 3.9),
        _word("escaping", 3.9, 4.4),
        _word("the", 4.4, 4.5),
        _word("world.", 4.5, 4.9),
    ]
    hook = "Zen is not about escaping"
    t = package.find_hook_moment(hook, words)
    assert t == pytest.approx(3.0)


def test_find_hook_moment_empty_words_returns_zero():
    assert package.find_hook_moment("anything", []) == 0.0


def test_find_hook_moment_no_token_overlap_falls_back_to_first_word():
    words = [_word("hello", 5.0, 5.4), _word("world", 5.4, 5.8)]
    assert package.find_hook_moment("completely different text", words) == 5.0


def test_build_meta_includes_required_fields():
    channel = load_channel("example_channel")
    candidate = {
        "id": "clip-01",
        "title": "A Sharp Reframe",
        "hook": "Zen is not about escaping.",
        "score": 9.2,
    }
    cut_meta = {"start": 4.8, "end": 27.5}
    meta = package.build_meta(candidate, cut_meta, channel, "/videos/talk.mp4")

    assert meta["title"] == "A Sharp Reframe"
    assert candidate["hook"] in meta["description"]
    assert meta["hashtags"] == channel.hashtags
    assert meta["source"] == {
        "video": "talk.mp4",
        "video_path": "/videos/talk.mp4",
        "start": 4.8,
        "end": 27.5,
    }
    assert meta["channel"] == "example_channel"
    assert meta["clip_id"] == "clip-01"
    assert meta["score"] == 9.2


def test_build_meta_title_over_limit_gets_truncated():
    channel = load_channel("example_channel")
    candidate = {"id": "clip-01", "title": "x" * 150, "hook": "hook", "score": 5}
    meta = package.build_meta(candidate, {"start": 0.0, "end": 20.0}, channel, "/v.mp4")
    assert len(meta["title"]) <= package.TITLE_MAX_CHARS


FAKE_TRANSCRIPT = {
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Zen is not about escaping the world.",
            "words": [
                _word("Zen", 0.0, 0.3),
                _word("is", 0.3, 0.4),
                _word("not", 0.4, 0.6),
                _word("about", 0.6, 0.9),
                _word("escaping", 0.9, 1.4),
                _word("the", 1.4, 1.5),
                _word("world.", 1.5, 1.9),
            ],
        }
    ]
}


def test_package_clip_invokes_ffmpeg_and_writes_sidecars(tmp_path, monkeypatch):
    clip_dir = tmp_path
    (clip_dir / "clip.mp4").write_bytes(b"fake")

    channel = load_channel("example_channel")
    candidate = {"id": "clip-01", "title": "Zen Isn't Escape", "hook": "Zen is not about escaping", "score": 9.0}
    cut_meta = {"start": 0.0, "end": 2.0}

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        (clip_dir / "thumb.jpg").write_bytes(b"fake jpg")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(package.subprocess, "run", fake_run)

    meta = package.package_clip(clip_dir, FAKE_TRANSCRIPT, candidate, cut_meta, channel, "/videos/talk.mp4")

    assert "ffmpeg" in captured["cmd"]
    assert (clip_dir / "thumb.jpg").exists()
    assert (clip_dir / "meta.json").exists()
    saved = json.loads((clip_dir / "meta.json").read_text())
    assert saved == meta
    assert meta["thumbnail_time"] == pytest.approx(0.0)  # "Zen is not about escaping" matches from t=0


def test_package_clip_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    clip_dir = tmp_path
    (clip_dir / "clip.mp4").write_bytes(b"fake")
    channel = load_channel("example_channel")
    candidate = {"id": "clip-01", "title": "t", "hook": "h", "score": 1}

    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(package.subprocess, "run", fake_run)

    with pytest.raises(package.PackageError):
        package.package_clip(clip_dir, FAKE_TRANSCRIPT, candidate, {"start": 0.0, "end": 2.0}, channel, "/v.mp4")


def test_package_clip_missing_video_raises(tmp_path):
    channel = load_channel("example_channel")
    candidate = {"id": "clip-01", "title": "t", "hook": "h", "score": 1}
    with pytest.raises(package.PackageError):
        package.package_clip(tmp_path, FAKE_TRANSCRIPT, candidate, {"start": 0.0, "end": 2.0}, channel, "/v.mp4")
