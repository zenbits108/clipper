"""Orchestrator CLI.

Phase 1 wires only the `select` subcommand:
    clipper select video.mp4 --channel <name>

Later phases add cut/reframe/captions/package plus the approval gate that
sits between selection and rendering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clipper.config import ConfigError, load_channel, load_settings
from clipper.llm import ProviderError
from clipper.select import SelectError, select_clips
from clipper.transcribe import TranscribeError, transcribe_video


def _cmd_select(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    try:
        settings = load_settings()
        channel = load_channel(args.channel)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    try:
        transcript = transcribe_video(video_path, settings, force=args.force_transcribe)
    except TranscribeError as exc:
        print(f"Transcription error: {exc}", file=sys.stderr)
        return 1

    try:
        candidates = select_clips(video_path, transcript, channel, settings)
    except (SelectError, ProviderError) as exc:
        print(f"Selection error: {exc}", file=sys.stderr)
        return 1

    if not candidates:
        print("No candidates passed validation. Check the selection prompt/model output.")
        return 0

    for c in candidates:
        length = c["end"] - c["start"]
        print(f"[{c['id']}] score={c['score']:.1f}  {c['start']:.1f}s-{c['end']:.1f}s ({length:.1f}s)")
        print(f"    title: {c['title']}")
        print(f"    hook:  {c['hook']}")
        if c.get("reason"):
            print(f"    why:   {c['reason']}")

    out_path = settings.output_dir / transcript["video_id"] / "candidates.json"
    print(f"\n{len(candidates)} candidates written to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser(
        "select", help="Transcribe (cached) and select clip candidates"
    )
    select_parser.add_argument("video", help="Path to the source video file")
    select_parser.add_argument(
        "--channel", required=True, help="Channel config name (config/channels/<name>.yaml)"
    )
    select_parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="Re-run whisper even if a cached transcript exists",
    )
    select_parser.set_defaults(func=_cmd_select)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
