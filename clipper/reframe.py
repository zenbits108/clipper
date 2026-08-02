"""9:16 reframe.

`face-track` mode samples a face position every N frames with mediapipe,
smooths it with an exponential moving average, collapses long still
stretches into a handful of keyframes, and drives ffmpeg's `crop` filter
with a piecewise-linear pan expression built from those keyframes so the
speaker stays centered without visible jitter. `center` and
`blur-pillarbox` are static, config-selectable fallbacks; `face-track`
also *auto*-falls back to a centered crop for a given clip if too few
sampled frames have a detectable face.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clipper.config import ChannelConfig

TARGET_W = 1080
TARGET_H = 1920
TARGET_ASPECT = TARGET_W / TARGET_H  # 0.5625, width/height

SAMPLE_EVERY_N_FRAMES = 15  # ~0.5s at 30fps, ~0.25s at 60fps
EMA_ALPHA = 0.15  # lower = smoother/slower to react, higher = snappier/more jitter
MOVE_THRESHOLD_PX = 8  # collapse samples within this many px into one keyframe
MIN_DETECTION_RATE = 0.5  # below this, face-track auto-falls back to center
FACE_MIN_CONFIDENCE = 0.5

# mediapipe's Tasks API (>=0.10.18, and the only API left as of 1.0) ships no
# bundled model — it's downloaded once and cached here. Official model from
# Google's mediapipe model zoo (see the Face Detector task guide).
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)
FACE_MODEL_CACHE = Path.home() / ".cache" / "clipper" / "models" / "blaze_face_short_range.tflite"


class ReframeError(Exception):
    """Raised when a video can't be probed/read or ffmpeg fails."""


def probe_video(video_path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ReframeError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise ReframeError(f"Could not read frame dimensions for {video_path}")
    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count}


def _even(n: float) -> int:
    n = int(round(n))
    return n - 1 if n % 2 else n


def crop_dimensions(width: int, height: int) -> tuple:
    """Return (crop_w, crop_h, axis) for a 9:16 crop of a source frame.
    Landscape sources (the norm for talking-head recordings) get a
    full-height crop panned horizontally (axis="x"); sources already
    narrower than 9:16 get a full-width crop with a fixed vertical center
    (axis="y") — a talking head rarely needs vertical panning, so no
    tracking is done on that axis.
    """
    if width / height >= TARGET_ASPECT:
        crop_h = height
        crop_w = min(width, _even(height * TARGET_ASPECT))
        return crop_w, crop_h, "x"
    crop_w = width
    crop_h = min(height, _even(width / TARGET_ASPECT))
    return crop_w, crop_h, "y"


def _face_model_path() -> Path:
    if not FACE_MODEL_CACHE.exists():
        import requests

        FACE_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(FACE_MODEL_URL, timeout=60)
        resp.raise_for_status()
        FACE_MODEL_CACHE.write_bytes(resp.content)
    return FACE_MODEL_CACHE


def _sample_face_centers(video_path: Path, axis: str, width: int, height: int) -> tuple:
    """Sample every SAMPLE_EVERY_N_FRAMES-th frame, return
    (list[(t_seconds, pixel_coord_along_axis)], detection_rate) using only
    frames where a face was actually found.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base_options = mp_python.BaseOptions(model_asset_path=str(_face_model_path()))
    options = vision.FaceDetectorOptions(
        base_options=base_options, min_detection_confidence=FACE_MIN_CONFIDENCE
    )
    detector = vision.FaceDetector.create_from_options(options)

    samples = []
    sampled = 0
    hits = 0

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        frame_idx = 0
        while True:
            if not cap.grab():
                break
            if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
                ok, frame = cap.retrieve()
                if ok:
                    sampled += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect(mp_image)
                    if result.detections:
                        best = max(result.detections, key=lambda d: d.categories[0].score)
                        bbox = best.bounding_box  # absolute pixel coords
                        cx = bbox.origin_x + bbox.width / 2
                        cy = bbox.origin_y + bbox.height / 2
                        coord = cx if axis == "x" else cy
                        samples.append((frame_idx / fps, coord))
                        hits += 1
            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    detection_rate = hits / sampled if sampled else 0.0
    return samples, detection_rate


def _ema_smooth(samples: list, alpha: float) -> list:
    if not samples:
        return []
    smoothed = [samples[0]]
    prev = samples[0][1]
    for t, x in samples[1:]:
        prev = alpha * x + (1 - alpha) * prev
        smoothed.append((t, prev))
    return smoothed


def _clamp_positions(samples: list, max_pos: float) -> list:
    return [(t, max(0.0, min(max_pos, x))) for t, x in samples]


def _compress_keyframes(samples: list, threshold_px: float) -> list:
    if not samples:
        return []
    keyframes = [samples[0]]
    last_kept = samples[0][1]
    for t, x in samples[1:]:
        if abs(x - last_kept) >= threshold_px:
            keyframes.append((t, x))
            last_kept = x
    if keyframes[-1][0] != samples[-1][0]:
        keyframes.append(samples[-1])
    return keyframes


def build_pan_keyframes(samples: list, max_pos: float) -> list:
    """Turn raw (t, coord) face samples into smoothed, clamped,
    run-length-compressed keyframes covering the clip from t=0, suitable
    for driving the crop filter's pan expression. Returns [] if there are
    no samples to work with.
    """
    if not samples:
        return []
    smoothed = _ema_smooth(samples, EMA_ALPHA)
    clamped = _clamp_positions(smoothed, max_pos)
    keyframes = _compress_keyframes(clamped, MOVE_THRESHOLD_PX)
    if keyframes[0][0] > 0.0:
        keyframes.insert(0, (0.0, keyframes[0][1]))
    return keyframes


def centered_keyframes(max_pos: float) -> list:
    return [(0.0, max_pos / 2)]


def build_pan_expr(keyframes: list) -> str:
    """Build a piecewise-linear ffmpeg expression in `t` from (time, coord)
    keyframes: linear interpolation between consecutive keyframes, holding
    the last value beyond the final keyframe.
    """
    if len(keyframes) == 1:
        return f"{keyframes[0][1]:.2f}"

    expr = f"{keyframes[-1][1]:.2f}"
    for i in range(len(keyframes) - 2, -1, -1):
        t0, x0 = keyframes[i]
        t1, x1 = keyframes[i + 1]
        if x0 == x1:
            segment = f"{x0:.2f}"
        else:
            segment = f"({x0:.2f}+({x1:.2f}-{x0:.2f})*(t-{t0:.3f})/{(t1 - t0):.3f})"
        expr = f"if(lt(t,{t1:.3f}),{segment},{expr})"
    return expr


def _crop_and_scale_filter(crop_w: int, crop_h: int, keyframes: list, axis: str) -> str:
    # commas from if(...) are argument separators within the expression, but
    # ffmpeg's filtergraph parser treats top-level commas as chaining to the
    # next filter — escape them so they stay inside crop's x/y expression.
    pan_expr = build_pan_expr(keyframes).replace(",", r"\,")
    if axis == "x":
        crop = f"crop={crop_w}:{crop_h}:{pan_expr}:0"
    else:
        crop = f"crop={crop_w}:{crop_h}:0:{pan_expr}"
    return f"{crop},scale={TARGET_W}:{TARGET_H}:flags=lanczos"


def _blur_pillarbox_filter() -> str:
    return (
        "split=2[bg][fg];"
        f"[bg]scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=increase,"
        f"crop={TARGET_W}:{TARGET_H},gblur=sigma=25[bg2];"
        f"[fg]scale={TARGET_W}:-2[fg2];"
        "[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
    )


def _ffmpeg_reframe_command(input_path: Path, vf: str, output_path: Path) -> list:
    return [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", "19",
        "-pix_fmt", "yuv420p",
        # Video is re-encoded (crop/scale requires it) but audio is untouched
        # by this stage — no trimming or filtering — so stream-copy is safe
        # and avoids a needless quality-losing re-encode.
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]


def plan_reframe(
    probe: dict,
    channel: ChannelConfig,
    sample_fn=_sample_face_centers,
    video_path: Path = None,
) -> dict:
    """Decide the crop/pan plan for a clip without touching ffmpeg: which
    mode actually gets used (honoring face-track's auto-fallback), the
    filter string, and detection stats for the metadata sidecar.
    """
    mode = channel.reframe.mode

    if mode == "blur-pillarbox":
        return {
            "mode_requested": mode,
            "mode_used": "blur-pillarbox",
            "detection_rate": None,
            "vf": _blur_pillarbox_filter(),
        }

    crop_w, crop_h, axis = crop_dimensions(probe["width"], probe["height"])
    max_pos = (probe["width"] - crop_w) if axis == "x" else (probe["height"] - crop_h)

    detection_rate = None
    mode_used = "center"
    keyframes = centered_keyframes(max_pos)

    if mode == "face-track" and axis == "x":
        samples, detection_rate = sample_fn(video_path, axis, probe["width"], probe["height"])
        if detection_rate >= MIN_DETECTION_RATE and samples:
            # samples are face-center coordinates; the crop filter's x/y
            # wants the crop window's edge, so shift by half the crop
            # extent to actually center the face in the window.
            half_extent = (crop_w / 2) if axis == "x" else (crop_h / 2)
            crop_samples = [(t, coord - half_extent) for t, coord in samples]
            keyframes = build_pan_keyframes(crop_samples, max_pos)
            mode_used = "face-track"

    return {
        "mode_requested": mode,
        "mode_used": mode_used,
        "detection_rate": detection_rate,
        "vf": _crop_and_scale_filter(crop_w, crop_h, keyframes, axis),
    }


def reframe_clip(
    cut_video_path: Path,
    clip_dir: Path,
    channel: ChannelConfig,
    sample_fn=_sample_face_centers,
) -> dict:
    """Reframe <clip_dir>/cut.mp4 to 9:16 into <clip_dir>/reframed.mp4,
    writing a <clip_dir>/reframe.json sidecar with the plan actually used.
    """
    cut_video_path = Path(cut_video_path)
    clip_dir = Path(clip_dir)
    if not cut_video_path.exists():
        raise ReframeError(f"Cut video not found: {cut_video_path}")

    probe = probe_video(cut_video_path)
    plan = plan_reframe(probe, channel, sample_fn=sample_fn, video_path=cut_video_path)

    clip_dir.mkdir(parents=True, exist_ok=True)
    output_path = clip_dir / "reframed.mp4"

    cmd = _ffmpeg_reframe_command(cut_video_path, plan["vf"], output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReframeError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-4000:]}")

    meta = {
        "mode_requested": plan["mode_requested"],
        "mode_used": plan["mode_used"],
        "detection_rate": plan["detection_rate"],
        "source_width": probe["width"],
        "source_height": probe["height"],
        "output_path": str(output_path.resolve()),
    }
    (clip_dir / "reframe.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
