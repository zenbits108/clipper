import json
import subprocess

import pytest

from clipper import reframe
from clipper.config import load_channel


def test_crop_dimensions_landscape_pans_horizontally():
    crop_w, crop_h, axis = reframe.crop_dimensions(1920, 1080)
    assert axis == "x"
    assert crop_h == 1080
    assert crop_w == round(1080 * reframe.TARGET_ASPECT)
    assert crop_w % 2 == 0


def test_crop_dimensions_narrow_source_pans_vertically():
    crop_w, crop_h, axis = reframe.crop_dimensions(1000, 2000)
    assert axis == "y"
    assert crop_w == 1000
    assert crop_h % 2 == 0


def test_ema_smooth_dampens_a_single_outlier():
    samples = [(0.0, 100.0), (1.0, 100.0), (2.0, 400.0), (3.0, 100.0), (4.0, 100.0)]
    smoothed = reframe._ema_smooth(samples, alpha=0.15)
    outlier_smoothed = smoothed[2][1]
    assert 100.0 < outlier_smoothed < 400.0
    assert outlier_smoothed == pytest.approx(145.0)  # 0.15*400 + 0.85*100, heavily damped


def test_clamp_positions_bounds_to_range():
    samples = [(0.0, -50.0), (1.0, 500.0), (2.0, 200.0)]
    clamped = reframe._clamp_positions(samples, max_pos=300.0)
    assert clamped == [(0.0, 0.0), (1.0, 300.0), (2.0, 200.0)]


def test_compress_keyframes_collapses_still_stretches():
    # Long flat run, then a real move, then flat again.
    samples = [(float(i), 100.0) for i in range(10)] + [(10.0, 250.0)] + [
        (float(i), 250.0) for i in range(11, 20)
    ]
    keyframes = reframe._compress_keyframes(samples, threshold_px=8)
    # First point, the move, and the final point -- not all 20 samples.
    assert len(keyframes) <= 4
    assert keyframes[0] == (0.0, 100.0)
    assert keyframes[-1][1] == 250.0


def test_build_pan_keyframes_starts_at_zero_even_if_first_sample_is_later():
    samples = [(2.0, 100.0), (3.0, 100.0), (4.0, 100.0)]
    keyframes = reframe.build_pan_keyframes(samples, max_pos=500.0)
    assert keyframes[0][0] == 0.0


def test_build_pan_expr_constant_for_single_keyframe():
    expr = reframe.build_pan_expr([(0.0, 42.0)])
    assert expr == "42.00"


def test_build_pan_expr_interpolates_between_keyframes():
    expr = reframe.build_pan_expr([(0.0, 0.0), (2.0, 100.0)])
    assert "if(lt(t,2.000)" in expr
    assert "100.00" in expr


def test_plan_reframe_center_mode_skips_face_sampling():
    channel = load_channel("example_channel")
    channel.reframe.mode = "center"
    probe = {"width": 1920, "height": 1080}

    def fail_if_called(*a, **k):
        raise AssertionError("center mode must not sample faces")

    plan = reframe.plan_reframe(probe, channel, sample_fn=fail_if_called, video_path="x.mp4")
    assert plan["mode_used"] == "center"
    assert plan["detection_rate"] is None
    assert "crop=" in plan["vf"]


def test_plan_reframe_blur_pillarbox_skips_face_sampling():
    channel = load_channel("example_channel")
    channel.reframe.mode = "blur-pillarbox"
    probe = {"width": 1920, "height": 1080}

    def fail_if_called(*a, **k):
        raise AssertionError("blur-pillarbox mode must not sample faces")

    plan = reframe.plan_reframe(probe, channel, sample_fn=fail_if_called, video_path="x.mp4")
    assert plan["mode_used"] == "blur-pillarbox"
    assert "gblur" in plan["vf"]


def test_plan_reframe_face_track_uses_tracking_when_detection_good():
    channel = load_channel("example_channel")
    channel.reframe.mode = "face-track"
    probe = {"width": 1920, "height": 1080}

    samples = [(float(i), 960.0) for i in range(10)]

    def fake_sample_fn(video_path, axis, width, height):
        return samples, 0.9

    plan = reframe.plan_reframe(probe, channel, sample_fn=fake_sample_fn, video_path="x.mp4")
    assert plan["mode_used"] == "face-track"
    assert plan["detection_rate"] == 0.9


def test_plan_reframe_face_track_centers_the_face_not_its_edge():
    # Regression test: sample_fn reports the face's CENTER pixel; the crop
    # filter's x parameter is the crop window's LEFT EDGE. Feeding the
    # center straight in as x jams the face against the crop's left border
    # instead of centering it -- this caught that exact bug on a real video.
    channel = load_channel("example_channel")
    channel.reframe.mode = "face-track"
    probe = {"width": 1920, "height": 1080}  # crop_w=608, crop_h=1080 (axis="x")

    face_center_x = 960.0  # dead center of the source frame
    samples = [(0.0, face_center_x), (1.0, face_center_x)]

    def fake_sample_fn(video_path, axis, width, height):
        return samples, 1.0

    plan = reframe.plan_reframe(probe, channel, sample_fn=fake_sample_fn, video_path="x.mp4")

    expected_crop_x = face_center_x - 608 / 2  # 656.00: crop left edge that centers the face
    assert "crop=608:1080:" in plan["vf"]
    assert f"{expected_crop_x:.2f}" in plan["vf"]
    assert "960.00" not in plan["vf"]  # the un-shifted center must not leak through


def test_plan_reframe_face_track_falls_back_when_detection_poor():
    channel = load_channel("example_channel")
    channel.reframe.mode = "face-track"
    probe = {"width": 1920, "height": 1080}

    def fake_sample_fn(video_path, axis, width, height):
        return [(0.0, 960.0)], 0.1  # below MIN_DETECTION_RATE

    plan = reframe.plan_reframe(probe, channel, sample_fn=fake_sample_fn, video_path="x.mp4")
    assert plan["mode_used"] == "center"
    assert plan["detection_rate"] == 0.1


def test_reframe_clip_invokes_ffmpeg_and_writes_sidecar(tmp_path, monkeypatch):
    cut_path = tmp_path / "cut.mp4"
    cut_path.write_bytes(b"fake")
    clip_dir = tmp_path

    channel = load_channel("example_channel")
    channel.reframe.mode = "center"

    monkeypatch.setattr(
        reframe, "probe_video", lambda p: {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
    )

    captured = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        (clip_dir / "reframed.mp4").write_bytes(b"fake output")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(reframe.subprocess, "run", fake_run)

    meta = reframe.reframe_clip(cut_path, clip_dir, channel)

    assert "h264_nvenc" in captured["cmd"]
    assert meta["mode_used"] == "center"
    assert (clip_dir / "reframe.json").exists()
    saved = json.loads((clip_dir / "reframe.json").read_text())
    assert saved == meta


def test_reframe_clip_raises_on_ffmpeg_failure(tmp_path, monkeypatch):
    cut_path = tmp_path / "cut.mp4"
    cut_path.write_bytes(b"fake")
    channel = load_channel("example_channel")
    channel.reframe.mode = "center"

    monkeypatch.setattr(
        reframe, "probe_video", lambda p: {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
    )

    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(reframe.subprocess, "run", fake_run)

    with pytest.raises(reframe.ReframeError):
        reframe.reframe_clip(cut_path, tmp_path, channel)


def test_reframe_clip_missing_video_raises(tmp_path):
    channel = load_channel("example_channel")
    with pytest.raises(reframe.ReframeError):
        reframe.reframe_clip(tmp_path / "nope.mp4", tmp_path, channel)
