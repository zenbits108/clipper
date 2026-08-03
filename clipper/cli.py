"""Orchestrator CLI.

    clipper select video.mp4 --channel <name>   # phase 1
    clipper cut video.mp4                        # phase 2
    clipper reframe video.mp4 --channel <name>   # phase 3
    clipper caption video.mp4 --channel <name>   # phase 4
    clipper package video.mp4 --channel <name>   # phase 5
    clipper upload video.mp4 --channel <name>    # phase 6

`cut` and `upload` are both approval gates: they re-display candidates/clips
and only act on the ids the user accepts (or --all / --clips to skip the
prompt) -- `upload` additionally never re-uploads a clip that already has an
upload.json unless --force, since re-uploading creates a brand new video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clipper.captions import CaptionError, caption_clip
from clipper.config import ConfigError, load_channel, load_settings
from clipper.cut import CutError, cut_clip
from clipper.llm import ProviderError
from clipper.package import PackageError, package_clip
from clipper.reframe import ReframeError, reframe_clip
from clipper.select import SelectError, select_clips
from clipper.transcribe import TranscribeError, transcribe_video, video_id_for
from clipper.upload import UploadError, upload_clip


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


def _cmd_reframe(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    try:
        settings = load_settings()
        channel = load_channel(args.channel)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    video_dir = settings.output_dir / video_id_for(video_path)
    clip_dirs = sorted(
        d for d in video_dir.glob("clip-*") if d.is_dir() and (d / "cut.mp4").exists()
    )
    if not clip_dirs:
        print(
            f"No cut clips found under {video_dir}. Run `clipper cut` first.",
            file=sys.stderr,
        )
        return 1

    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        unknown = wanted - {d.name for d in clip_dirs}
        if unknown:
            print(f"Unknown clip id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        clip_dirs = [d for d in clip_dirs if d.name in wanted]

    failures = []
    processed = 0
    for clip_dir in clip_dirs:
        if (clip_dir / "reframed.mp4").exists() and not args.force:
            print(f"[{clip_dir.name}] already reframed, skipping (use --force to redo)")
            continue
        try:
            meta = reframe_clip(clip_dir / "cut.mp4", clip_dir, channel)
        except ReframeError as exc:
            print(f"[{clip_dir.name}] FAILED: {exc}", file=sys.stderr)
            failures.append(clip_dir.name)
            continue
        rate = f"{meta['detection_rate']:.0%}" if meta["detection_rate"] is not None else "n/a"
        print(f"[{clip_dir.name}] mode={meta['mode_used']} detection={rate} -> {meta['output_path']}")
        processed += 1

    if failures:
        print(f"\n{len(failures)} clip(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{processed} clip(s) reframed.")
    return 0


def _cmd_caption(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    try:
        settings = load_settings()
        channel = load_channel(args.channel)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    video_dir = settings.output_dir / video_id_for(video_path)
    transcript_path = video_dir / "transcript.json"
    if not transcript_path.exists():
        print(
            f"No cached transcript for this video in {video_dir}. Run `clipper select` first.",
            file=sys.stderr,
        )
        return 1
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))

    clip_dirs = sorted(
        d for d in video_dir.glob("clip-*") if d.is_dir() and (d / "reframed.mp4").exists()
    )
    if not clip_dirs:
        print(
            f"No reframed clips found under {video_dir}. Run `clipper reframe` first.",
            file=sys.stderr,
        )
        return 1

    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        unknown = wanted - {d.name for d in clip_dirs}
        if unknown:
            print(f"Unknown clip id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        clip_dirs = [d for d in clip_dirs if d.name in wanted]

    failures = []
    processed = 0
    for clip_dir in clip_dirs:
        if (clip_dir / "clip.mp4").exists() and not args.force:
            print(f"[{clip_dir.name}] already captioned, skipping (use --force to redo)")
            continue
        cut_meta_path = clip_dir / "cut.json"
        if not cut_meta_path.exists():
            print(f"[{clip_dir.name}] FAILED: missing cut.json", file=sys.stderr)
            failures.append(clip_dir.name)
            continue
        cut_meta = json.loads(cut_meta_path.read_text(encoding="utf-8"))
        try:
            meta = caption_clip(clip_dir / "reframed.mp4", clip_dir, transcript, cut_meta, channel)
        except CaptionError as exc:
            print(f"[{clip_dir.name}] FAILED: {exc}", file=sys.stderr)
            failures.append(clip_dir.name)
            continue
        print(
            f"[{clip_dir.name}] {meta['word_count']} words, {meta['chunk_count']} chunks "
            f"-> {meta['output_path']}"
        )
        processed += 1

    if failures:
        print(f"\n{len(failures)} clip(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{processed} clip(s) captioned.")
    return 0


def _cmd_package(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    try:
        settings = load_settings()
        channel = load_channel(args.channel)
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
    candidates_data = json.loads(candidates_path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in candidates_data["candidates"]}
    source_video_path = candidates_data["video_path"]

    clip_dirs = sorted(
        d for d in video_dir.glob("clip-*") if d.is_dir() and (d / "clip.mp4").exists()
    )
    if not clip_dirs:
        print(
            f"No finished clips found under {video_dir}. Run `clipper caption` first.",
            file=sys.stderr,
        )
        return 1

    if args.clips:
        wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
        unknown = wanted - {d.name for d in clip_dirs}
        if unknown:
            print(f"Unknown clip id(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        clip_dirs = [d for d in clip_dirs if d.name in wanted]

    failures = []
    processed = 0
    for clip_dir in clip_dirs:
        if (clip_dir / "meta.json").exists() and (clip_dir / "thumb.jpg").exists() and not args.force:
            print(f"[{clip_dir.name}] already packaged, skipping (use --force to redo)")
            continue
        candidate = by_id.get(clip_dir.name)
        cut_meta_path = clip_dir / "cut.json"
        if candidate is None or not cut_meta_path.exists():
            print(f"[{clip_dir.name}] FAILED: missing candidate entry or cut.json", file=sys.stderr)
            failures.append(clip_dir.name)
            continue
        cut_meta = json.loads(cut_meta_path.read_text(encoding="utf-8"))
        try:
            meta = package_clip(clip_dir, transcript, candidate, cut_meta, channel, source_video_path)
        except PackageError as exc:
            print(f"[{clip_dir.name}] FAILED: {exc}", file=sys.stderr)
            failures.append(clip_dir.name)
            continue
        print(f"[{clip_dir.name}] \"{meta['title']}\" -> {clip_dir}")
        processed += 1

    if failures:
        print(f"\n{len(failures)} clip(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{processed} clip(s) packaged.")
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    try:
        settings = load_settings()
        channel = load_channel(args.channel)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    video_dir = settings.output_dir / video_id_for(video_path)
    clip_dirs = sorted(
        d for d in video_dir.glob("clip-*")
        if d.is_dir() and (d / "clip.mp4").exists() and (d / "meta.json").exists()
    )
    if not clip_dirs:
        print(
            f"No packaged clips found under {video_dir}. Run `clipper package` first.",
            file=sys.stderr,
        )
        return 1
    by_id = {d.name: d for d in clip_dirs}

    if args.all:
        selected_ids = list(by_id)
    elif args.clips:
        selected_ids = [c.strip() for c in args.clips.split(",") if c.strip()]
    else:
        for clip_dir in clip_dirs:
            meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
            tag = " [already uploaded]" if (clip_dir / "upload.json").exists() else ""
            print(f"[{clip_dir.name}] \"{meta['title']}\"{tag}")
        privacy = args.privacy or channel.upload.privacy_status
        raw = input(
            f"\nUpload which clips to YouTube as '{privacy}'? "
            "Comma-separated ids, 'all', or blank to cancel: "
        ).strip()
        if not raw:
            print("Cancelled.")
            return 0
        selected_ids = list(by_id) if raw.lower() == "all" else [c.strip() for c in raw.split(",") if c.strip()]

    unknown = [i for i in selected_ids if i not in by_id]
    if unknown:
        print(f"Unknown clip id(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    failures = []
    processed = 0
    for clip_id in selected_ids:
        clip_dir = by_id[clip_id]
        if (clip_dir / "upload.json").exists() and not args.force:
            print(f"[{clip_id}] already uploaded, skipping (use --force to re-upload)")
            continue
        try:
            meta = upload_clip(clip_dir, channel, settings, privacy_status=args.privacy)
        except UploadError as exc:
            print(f"[{clip_id}] FAILED: {exc}", file=sys.stderr)
            failures.append(clip_id)
            continue
        print(f"[{clip_id}] uploaded ({meta['privacy_status']}) -> {meta['url']}")
        processed += 1

    if failures:
        print(f"\n{len(failures)} clip(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\n{processed} clip(s) uploaded.")
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

    reframe_parser = subparsers.add_parser(
        "reframe", help="Crop cut clips to 9:16 (face-tracked or config-selected fallback)"
    )
    reframe_parser.add_argument("video", help="Path to the source video file")
    reframe_parser.add_argument(
        "--channel", required=True, help="Channel config name (config/channels/<name>.yaml)"
    )
    reframe_parser.add_argument(
        "--clips", help="Comma-separated clip ids to reframe, e.g. clip-01,clip-03"
    )
    reframe_parser.add_argument(
        "--force", action="store_true", help="Re-reframe clips that already have reframed.mp4"
    )
    reframe_parser.set_defaults(func=_cmd_reframe)

    caption_parser = subparsers.add_parser(
        "caption", help="Burn word-level highlighted captions into reframed clips"
    )
    caption_parser.add_argument("video", help="Path to the source video file")
    caption_parser.add_argument(
        "--channel", required=True, help="Channel config name (config/channels/<name>.yaml)"
    )
    caption_parser.add_argument(
        "--clips", help="Comma-separated clip ids to caption, e.g. clip-01,clip-03"
    )
    caption_parser.add_argument(
        "--force", action="store_true", help="Re-caption clips that already have clip.mp4"
    )
    caption_parser.set_defaults(func=_cmd_caption)

    package_parser = subparsers.add_parser(
        "package", help="Write meta.json + thumb.jpg for finished clips"
    )
    package_parser.add_argument("video", help="Path to the source video file")
    package_parser.add_argument(
        "--channel", required=True, help="Channel config name (config/channels/<name>.yaml)"
    )
    package_parser.add_argument(
        "--clips", help="Comma-separated clip ids to package, e.g. clip-01,clip-03"
    )
    package_parser.add_argument(
        "--force", action="store_true", help="Re-package clips that already have meta.json/thumb.jpg"
    )
    package_parser.set_defaults(func=_cmd_package)

    upload_parser = subparsers.add_parser(
        "upload", help="Upload packaged clips to YouTube (defaults to private)"
    )
    upload_parser.add_argument("video", help="Path to the source video file")
    upload_parser.add_argument(
        "--channel", required=True, help="Channel config name (config/channels/<name>.yaml)"
    )
    upload_parser.add_argument(
        "--all", action="store_true", help="Upload every eligible clip without prompting"
    )
    upload_parser.add_argument(
        "--clips", help="Comma-separated clip ids to upload without prompting, e.g. clip-01,clip-03"
    )
    upload_parser.add_argument(
        "--privacy",
        choices=["private", "unlisted", "public"],
        help="Override the channel's configured privacy_status for this run",
    )
    upload_parser.add_argument(
        "--force", action="store_true", help="Re-upload clips that already have upload.json (creates a NEW video)"
    )
    upload_parser.set_defaults(func=_cmd_upload)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
