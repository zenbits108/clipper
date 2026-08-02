"""Orchestrator CLI.

    clipper select video.mp4 --channel <name>   # phase 1
    clipper cut video.mp4                        # phase 2

`cut` is the approval gate: it reads the candidates `select` already wrote,
shows them again, and only renders the ids the user accepts (or --all /
--clips to skip the prompt). Later phases add reframe/captions/package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clipper.config import ConfigError, load_channel, load_settings
from clipper.cut import CutError, cut_clip
from clipper.llm import ProviderError
from clipper.select import SelectError, select_clips
from clipper.transcribe import TranscribeError, transcribe_video, video_id_for


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


def _print_candidates(candidates: list) -> None:
    for c in candidates:
        length = c["end"] - c["start"]
        print(f"[{c['id']}] score={c['score']:.1f}  {c['start']:.1f}s-{c['end']:.1f}s ({length:.1f}s)  {c['title']}")


def _cmd_cut(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    video_dir = settings.output_dir / video_id_for(video_path)
    transcript_path = video_dir / "transcript.json"
    candidates_path = video_dir / "candidates.json"
    if not transcript_path.exists() or not candidates_path.exists():
        print(
            f"No cached transcript/candidates for this video in {video_dir}. "
            "Run `clipper select` first.",
            file=sys.stderr,
        )
        return 1

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))["candidates"]
    if not candidates:
        print("No candidates to cut.")
        return 0

    by_id = {c["id"]: c for c in candidates}

    if args.all:
        selected_ids = list(by_id)
    elif args.clips:
        selected_ids = [c.strip() for c in args.clips.split(",") if c.strip()]
    else:
        _print_candidates(candidates)
        raw = input("\nRender which clips? Comma-separated ids, 'all', or blank to cancel: ").strip()
        if not raw:
            print("Cancelled.")
            return 0
        selected_ids = list(by_id) if raw.lower() == "all" else [c.strip() for c in raw.split(",") if c.strip()]

    unknown = [i for i in selected_ids if i not in by_id]
    if unknown:
        print(f"Unknown clip id(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    failures = []
    for clip_id in selected_ids:
        candidate = by_id[clip_id]
        clip_dir = video_dir / clip_id
        try:
            meta = cut_clip(video_path, transcript, candidate, clip_dir)
        except CutError as exc:
            print(f"[{clip_id}] FAILED: {exc}", file=sys.stderr)
            failures.append(clip_id)
            continue
        print(f"[{clip_id}] cut {meta['start']:.2f}s-{meta['end']:.2f}s -> {meta['output_path']}")

    if failures:
        print(f"\n{len(failures)} clip(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{len(selected_ids)} clip(s) rendered.")
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

    cut_parser = subparsers.add_parser(
        "cut", help="Render frame-accurate cuts for approved clip candidates"
    )
    cut_parser.add_argument("video", help="Path to the source video file")
    cut_parser.add_argument(
        "--all", action="store_true", help="Render every candidate without prompting"
    )
    cut_parser.add_argument(
        "--clips",
        help="Comma-separated candidate ids to render without prompting, e.g. clip-01,clip-03",
    )
    cut_parser.set_defaults(func=_cmd_cut)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
