"""Word-level .ass caption generation + burn-in.

Groups whisper words into short on-screen chunks (a handful of words,
capped by duration and broken on sentence punctuation). For every word,
emits one ASS Dialogue line spanning that word's timestamps, showing the
chunk's text with the spoken word in the channel's highlight color and
everything else in its primary color -- the classic word-by-word
highlight look. Position is channel-configured; the default keeps
captions out of the bottom ~25% of the frame (Shorts UI safe zone).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from clipper.config import CaptionConfig, ChannelConfig
from clipper.reframe import TARGET_H, TARGET_W

MAX_WORDS_PER_CHUNK = 5
MAX_CHUNK_SECONDS = 2.5
MAX_WORD_DISPLAY_SECONDS = 1.5  # faster-whisper occasionally mis-times a
# short word's end far past when it was actually spoken (seen on a real
# clip: "or" timestamped as lasting 4.8s); without this cap that word's
# highlight sticks on screen for however long whisper claims it did.

# Alignment follows ASS numpad convention (2=bottom-center, 5=middle-center,
# 8=top-center); margin is measured inward from the edge that alignment
# anchors to.
_ALIGNMENT_AND_MARGIN = {
    "bottom_safe": (2, int(TARGET_H * 0.25)),
    "middle": (5, 0),
    "top_safe": (8, int(TARGET_H * 0.10)),
}


class CaptionError(Exception):
    """Raised when caption generation or burn-in fails."""


def words_in_clip(transcript: dict, cut_meta: dict) -> list:
    """Transcript words overlapping the clip's actual (padded) cut window,
    with timestamps shifted to be relative to the clip's own start.
    """
    clip_start = cut_meta["start"]
    clip_end = cut_meta["end"]
    words = []
    for seg in transcript["segments"]:
        for w in seg.get("words", []):
            if w["end"] <= clip_start or w["start"] >= clip_end:
                continue
            text = w["word"].strip()
            if not text:
                continue
            start = max(0.0, w["start"] - clip_start)
            end = min(clip_end - clip_start, w["end"] - clip_start)
            end = min(end, start + MAX_WORD_DISPLAY_SECONDS)
            words.append({"word": text, "start": start, "end": end})
    return words


def group_into_chunks(words: list) -> list:
    """Group words into short on-screen chunks, breaking after sentence
    punctuation or when a chunk would exceed the word/duration cap.
    """
    chunks = []
    current = []
    for w in words:
        if current:
            prev_ends_sentence = current[-1]["word"][-1] in ".!?"
            too_many_words = len(current) >= MAX_WORDS_PER_CHUNK
            too_long = (w["end"] - current[0]["start"]) > MAX_CHUNK_SECONDS
            if prev_ends_sentence or too_many_words or too_long:
                chunks.append(current)
                current = []
        current.append(w)
    if current:
        chunks.append(current)
    return chunks


def _format_ass_time(seconds: float) -> str:
    centis = round(max(0.0, seconds) * 100)
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _override_color(style_color: str) -> str:
    """Convert a Style-line &HAABBGGRR color to the &HBBGGRR& form used in
    inline {\\c...} override tags (no alpha byte).
    """
    hex_part = style_color.upper().removeprefix("&H")
    bgr = hex_part[-6:]
    return f"&H{bgr}&"


def build_ass(chunks: list, caption: CaptionConfig) -> str:
    alignment, margin_v = _ALIGNMENT_AND_MARGIN.get(
        caption.position, _ALIGNMENT_AND_MARGIN["bottom_safe"]
    )
    primary = _override_color(caption.primary_color)
    highlight = _override_color(caption.highlight_color)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {TARGET_W}\n"
        f"PlayResY: {TARGET_H}\n"
        "ScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{caption.font},{caption.font_size},{caption.primary_color},"
        f"{caption.primary_color},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,"
        f"{alignment},40,40,{margin_v},1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = []
    for chunk in chunks:
        for i, w in enumerate(chunk):
            parts = []
            for j, cw in enumerate(chunk):
                text = _escape_ass_text(cw["word"])
                parts.append(f"{{\\c{highlight}}}{text}{{\\c{primary}}}" if j == i else text)
            text_line = " ".join(parts)
            start = _format_ass_time(w["start"])
            end = _format_ass_time(w["end"])
            lines.append(f"Dialogue: 0,{start},{end},Caption,,0,0,0,,{text_line}")

    return header + "\n".join(lines) + "\n"


def _ffmpeg_caption_command(input_path: Path, ass_filename: str, output_path: Path) -> list:
    return [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vf", f"ass={ass_filename}",
        "-c:v", "h264_nvenc",
        "-preset", "p5",
        "-rc", "vbr",
        "-cq", "19",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]


def caption_clip(
    reframed_video_path: Path,
    clip_dir: Path,
    transcript: dict,
    cut_meta: dict,
    channel: ChannelConfig,
) -> dict:
    """Burn word-level highlighted captions into <clip_dir>/reframed.mp4,
    writing <clip_dir>/clip.mp4 -- the pipeline's final rendered clip --
    plus a <clip_dir>/captions.ass sidecar. Returns caption metadata.
    """
    reframed_video_path = Path(reframed_video_path)
    clip_dir = Path(clip_dir)
    if not reframed_video_path.exists():
        raise CaptionError(f"Reframed video not found: {reframed_video_path}")

    words = words_in_clip(transcript, cut_meta)
    chunks = group_into_chunks(words)
    ass_content = build_ass(chunks, channel.caption)

    clip_dir.mkdir(parents=True, exist_ok=True)
    ass_path = clip_dir / "captions.ass"
    ass_path.write_text(ass_content, encoding="utf-8")

    output_path = clip_dir / "clip.mp4"
    # cwd=clip_dir + a bare filename in the filter string sidesteps the
    # ass/subtitles filter's colon-as-option-separator parsing entirely --
    # a Windows path like C:\Users\...\captions.ass would otherwise need
    # fragile double-escaping.
    cmd = _ffmpeg_caption_command(reframed_video_path.resolve(), ass_path.name, output_path.resolve())
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=clip_dir)
    if result.returncode != 0:
        raise CaptionError(f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-4000:]}")

    meta = {
        "word_count": len(words),
        "chunk_count": len(chunks),
        "ass_path": str(ass_path.resolve()),
        "output_path": str(output_path.resolve()),
    }
    (clip_dir / "captions.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
