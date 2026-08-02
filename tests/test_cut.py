import json
import subprocess

import pytest

from clipper import cut


def _word(word, start, end):
    return {"word": word, "start": start, "end": end, "probability": 0.99}


FAKE_TRANSCRIPT = {
    "duration": 30.0,
    "segments": [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hello there. Welcome to the show.",
            "words": [
                _word("Hello", 0.0, 0.4),
                _word("there.", 0.4, 0.8),
                _word("Welcome", 1.5, 1.9),
                _word("to", 1.9, 2.0),
                _word("the", 2.0, 2.1),
                _word("show.", 2.1, 2.5),
            ],
        },
        {
            "start": 5.0,
            "end": 12.0,
            "text": "Here is a fact you did not know. It changed everything.",
            "words": [
                _word("Here", 5.0, 5.2),
                _word("is", 5.2, 5.3),
                _word("a", 5.3, 5.35),
                _word("fact", 5.35, 5.6),
                _word("you", 5.6, 5.7),
                _word("did", 5.7, 5.8),
                _word("not", 5.8, 5.9),
                _word("know.", 5.9, 6.3),
                _word("It", 8.0, 8.1),
                _word("changed", 8.1, 8.5),
                _word("everything.", 8.5, 9.0),
            ],
        },
        {
            "start": 12.0,
            "end": 20.0,
            "text": "And that's how it wrapped up.",
            "words": [
                _word("And", 12.0, 12.1),
                _word("that's", 12.1, 12.3),
                _word("how", 12.3, 12.4),
                _word("it", 12.4, 12.45),
                _word("wrapped", 12.45, 12.7),
                _word("up.", 12.7, 13.0),
            ],
        },
    ],
}


def test_snaps_to_sentence_boundary_not_mid_word():
    # Requested start (0.6) falls inside "there." (0.4-0.8) — must snap back
    # to the sentence start (0.0), never mid-word.
    start, end = cut.compute_cut_window(FAKE_TRANSCRIPT, 0.6, 5.5)
    assert start == 0.0  # first sentence start, no earlier boundary to pad into
    assert end >= 6.3  # snapped forward past "know." not mid-sentence


def test_pads_without_bleeding_into_previous_sentence():
    # Second sentence ("Here is a fact...") starts at 5.0, right after the
    # first sentence ends at 2.5 with room to spare — padding should apply.
    start, end = cut.compute_cut_window(FAKE_TRANSCRIPT, 5.0, 9.0)
    assert start == pytest.approx(4.7, abs=1e-6)  # 5.0 - 0.3 pad
    assert end == pytest.approx(9.3, abs=1e-6)  # 9.0 + 0.3 pad, gap allows it


def test_end_pads_fully_when_gap_to_next_word_is_wide():
    start, end = cut.compute_cut_window(FAKE_TRANSCRIPT, 5.0, 13.0)
    assert start == pytest.approx(4.7, abs=1e-6)
    # no word after 13.0, so padding is bounded only by video duration (30.0)
    assert end == pytest.approx(13.3, abs=1e-6)


def test_padding_clamped_by_tight_neighboring_word():
    # Sentence A ends at 2.0, the next word starts at 2.15 -- a 0.15s gap,
    # narrower than the 0.3s pad. End-padding must clamp to that word's
    # start rather than bleed 0.05s into the next sentence.
    transcript = {
        "duration": 10.0,
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "Short one.",
                "words": [_word("Short", 0.0, 1.5), _word("one.", 1.5, 2.0)],
            },
            {
                "start": 2.15,
                "end": 4.0,
                "text": "Right after it.",
                "words": [_word("Right", 2.15, 2.4), _word("after", 2.4, 2.6), _word("it.", 2.6, 3.0)],
            },
        ],
    }
    start, end = cut.compute_cut_window(transcript, 0.0, 2.0)
    assert start == 0.0
    assert end == pytest.approx(2.15, abs=1e-6)  # clamped, not 2.0 + 0.3 = 2.3


def test_rejects_inverted_window():
    with pytest.raises(cut.CutError):
        cut.compute_cut_window(FAKE_TRANSCRIPT, 5.0, 5.0)


def test_cut_clip_invokes_ffmpeg_and_writes_sidecar(tmp_path, monkeypatch):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"fake")
    clip_dir = tmp_path / "clip-01"

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        (clip_dir / "cut.mp4").write_bytes(b"fake output")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cut.subprocess, "run", fake_run)

    candidate = {"id": "clip-01", "start": 5.0, "end": 9.0}
    meta = cut.cut_clip(video_path, FAKE_TRANSCRIPT, candidate, clip_dir)

    assert "-y" in captured["cmd"]
    assert "h264_nvenc" in captured["cmd"]
    assert "copy" not in captured["cmd"]  # never stream-copy
    assert str(video_path) in captured["cmd"]

    assert meta["id"] == "clip-01"
    assert meta["requested_start"] == 5.0
    assert meta["requested_end"] == 9.0
    assert (clip_dir / "cut.json").exists()
    saved = json.loads((clip_dir / "cut.json").read_text())
    assert saved == meta


def test_cut_clip_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"fake")
    clip_dir = tmp_path / "clip-01"

    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(cut.subprocess, "run", fake_run)

    candidate = {"id": "clip-01", "start": 5.0, "end": 9.0}
    with pytest.raises(cut.CutError):
        cut.cut_clip(video_path, FAKE_TRANSCRIPT, candidate, clip_dir)


def test_cut_clip_missing_video_raises(tmp_path):
    candidate = {"id": "clip-01", "start": 5.0, "end": 9.0}
    with pytest.raises(cut.CutError):
        cut.cut_clip(tmp_path / "nope.mp4", FAKE_TRANSCRIPT, candidate, tmp_path / "clip-01")
