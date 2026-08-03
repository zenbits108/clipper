import json
import subprocess

import pytest

from clipper import captions
from clipper.config import load_channel


def _word(word, start, end):
    return {"word": word, "start": start, "end": end, "probability": 0.99}


FAKE_TRANSCRIPT = {
    "segments": [
        {
            "start": 0.0,
            "end": 6.3,
            "text": "Hello there. Welcome to the show today.",
            "words": [
                _word("Hello", 0.0, 0.4),
                _word("there.", 0.4, 0.8),
                _word("Welcome", 1.5, 1.9),
                _word("to", 1.9, 2.0),
                _word("the", 2.0, 2.1),
                _word("show", 2.1, 2.4),
                _word("today.", 2.4, 2.9),
            ],
        },
        {
            "start": 10.0,
            "end": 12.0,
            "text": "Way outside the clip.",
            "words": [_word("Way", 10.0, 10.3), _word("outside.", 11.5, 12.0)],
        },
    ],
}


def test_words_in_clip_caps_implausibly_long_word_duration():
    # Regression test: a real clip had faster-whisper report a single
    # short word ("or") as lasting 4.8s, which made its caption highlight
    # visibly stick on screen. Any word's display duration must be capped.
    transcript = {
        "segments": [
            {
                "start": 0.0,
                "end": 20.0,
                "text": "calm or sitting",
                "words": [
                    _word("calm", 0.0, 0.4),
                    _word("or", 0.68, 5.48),  # implausible 4.8s span
                    _word("sitting", 5.48, 5.86),
                ],
            }
        ]
    }
    words = captions.words_in_clip(transcript, {"start": 0.0, "end": 20.0})
    or_word = next(w for w in words if w["word"] == "or")
    assert (or_word["end"] - or_word["start"]) <= captions.MAX_WORD_DISPLAY_SECONDS


def test_words_in_clip_filters_and_shifts_to_clip_relative_time():
    cut_meta = {"start": 1.0, "end": 3.0}  # clip-relative window inside source video
    words = captions.words_in_clip(FAKE_TRANSCRIPT, cut_meta)

    # "Hello" (0.0-0.4) ends before clip_start=1.0 -> excluded
    # "there." (0.4-0.8) ends before clip_start=1.0 -> excluded
    # everything in the second segment starts after clip_end=3.0 -> excluded
    words_text = [w["word"] for w in words]
    assert words_text == ["Welcome", "to", "the", "show", "today."]
    assert words[0]["start"] == pytest.approx(0.5)  # 1.5 - 1.0
    assert words[-1]["end"] == pytest.approx(1.9)  # 2.9 - 1.0


def test_words_in_clip_clamps_to_clip_bounds():
    # A word straddling the clip boundary should be clamped, not excluded.
    cut_meta = {"start": 0.6, "end": 2.0}
    words = captions.words_in_clip(FAKE_TRANSCRIPT, cut_meta)
    there = next(w for w in words if w["word"] == "there.")
    assert there["start"] == 0.0  # clamped: 0.4 - 0.6 would be negative
    welcome = next(w for w in words if w["word"] == "Welcome")
    assert welcome["end"] == pytest.approx(1.3)  # 1.9 - 0.6, within bounds


def test_group_into_chunks_breaks_on_sentence_punctuation():
    words = captions.words_in_clip(FAKE_TRANSCRIPT, {"start": 0.0, "end": 6.3})
    chunks = captions.group_into_chunks(words)
    assert [w["word"] for w in chunks[0]] == ["Hello", "there."]
    assert [w["word"] for w in chunks[1]] == ["Welcome", "to", "the", "show", "today."]


def test_group_into_chunks_breaks_on_word_count_cap():
    # Tightly spaced (well under MAX_CHUNK_SECONDS even at 5 words), so the
    # word-count cap is the only thing that can force a break here.
    words = [_word(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(12)]
    chunks = captions.group_into_chunks(words)
    assert [len(c) for c in chunks] == [5, 5, 2]


def test_group_into_chunks_breaks_on_duration_cap():
    # 4 words spaced 1s apart -> exceeds MAX_CHUNK_SECONDS (2.5) before
    # hitting the word-count cap.
    words = [_word(f"w{i}", float(i), float(i) + 0.3) for i in range(4)]
    chunks = captions.group_into_chunks(words)
    assert len(chunks) > 1
    for c in chunks:
        assert (c[-1]["end"] - c[0]["start"]) <= captions.MAX_CHUNK_SECONDS


def test_format_ass_time():
    assert captions._format_ass_time(0.0) == "0:00:00.00"
    assert captions._format_ass_time(65.5) == "0:01:05.50"
    assert captions._format_ass_time(3661.25) == "1:01:01.25"


def test_override_color_strips_alpha_byte():
    assert captions._override_color("&H00FFFFFF") == "&HFFFFFF&"
    assert captions._override_color("&H0000D7FF") == "&H00D7FF&"


def test_build_ass_contains_style_and_per_word_dialogue():
    channel = load_channel("example_channel")
    words = captions.words_in_clip(FAKE_TRANSCRIPT, {"start": 0.0, "end": 6.3})
    chunks = captions.group_into_chunks(words)
    ass = captions.build_ass(chunks, channel.caption)

    assert "[V4+ Styles]" in ass
    assert channel.caption.font in ass
    # one Dialogue line per word across both chunks (2 + 5 = 7)
    assert ass.count("Dialogue:") == 7
    # the highlight color override appears (word-by-word highlighting)
    assert captions._override_color(channel.caption.highlight_color) in ass


def test_build_ass_bottom_safe_keeps_margin_out_of_bottom_quarter():
    channel = load_channel("example_channel")
    channel.caption.position = "bottom_safe"
    words = captions.words_in_clip(FAKE_TRANSCRIPT, {"start": 0.0, "end": 6.3})
    chunks = captions.group_into_chunks(words)
    ass = captions.build_ass(chunks, channel.caption)

    alignment, margin_v = captions._ALIGNMENT_AND_MARGIN["bottom_safe"]
    assert margin_v == int(captions.TARGET_H * 0.25)
    assert f",{alignment},40,40,{margin_v},1" in ass


def test_caption_clip_invokes_ffmpeg_and_writes_sidecars(tmp_path, monkeypatch):
    reframed_path = tmp_path / "reframed.mp4"
    reframed_path.write_bytes(b"fake")
    clip_dir = tmp_path

    channel = load_channel("example_channel")
    cut_meta = {"start": 0.0, "end": 6.3}

    captured = {}

    def fake_run(cmd, capture_output, text, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        (clip_dir / "clip.mp4").write_bytes(b"fake output")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(captions.subprocess, "run", fake_run)

    meta = captions.caption_clip(reframed_path, clip_dir, FAKE_TRANSCRIPT, cut_meta, channel)

    assert "h264_nvenc" in captured["cmd"]
    assert captured["cwd"] == clip_dir
    # the ass reference in the filter must be a bare filename, not an
    # absolute path (Windows-safe: sidesteps colon-escaping entirely)
    assert "ass=captions.ass" in captured["cmd"]
    assert meta["word_count"] == 7
    assert (clip_dir / "captions.ass").exists()
    assert (clip_dir / "captions.json").exists()
    saved = json.loads((clip_dir / "captions.json").read_text())
    assert saved == meta


def test_caption_clip_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    reframed_path = tmp_path / "reframed.mp4"
    reframed_path.write_bytes(b"fake")
    channel = load_channel("example_channel")

    def fake_run(cmd, capture_output, text, cwd):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(captions.subprocess, "run", fake_run)

    with pytest.raises(captions.CaptionError):
        captions.caption_clip(reframed_path, tmp_path, FAKE_TRANSCRIPT, {"start": 0.0, "end": 6.3}, channel)


def test_caption_clip_missing_video_raises(tmp_path):
    channel = load_channel("example_channel")
    with pytest.raises(captions.CaptionError):
        captions.caption_clip(
            tmp_path / "nope.mp4", tmp_path, FAKE_TRANSCRIPT, {"start": 0.0, "end": 1.0}, channel
        )
