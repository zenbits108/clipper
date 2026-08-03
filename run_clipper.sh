#!/usr/bin/env bash
# Run the clipper pipeline (select -> cut -> reframe -> caption -> package)
# on a single video, for a given channel config.
#
# Usage:
#   ./run_clipper.sh <video.mp4> <channel_name> [options]
#
# Options:
#   --all                Skip the cut approval prompt, render every candidate
#   --clips <ids>         Skip the cut approval prompt, render only these ids
#                          (comma-separated, e.g. clip-01,clip-03)
#   --force-transcribe    Re-run whisper even if a cached transcript exists
#   --upload               Also run `clipper upload` at the end (still
#                          respects the channel's configured privacy_status,
#                          default private -- nothing goes public automatically)
#
# Without --all/--clips, `cut` prompts interactively for which candidates to
# render -- run this in a real terminal (not piped or backgrounded) so you
# can answer it.

set -euo pipefail

usage() {
  echo "Usage: $0 <video.mp4> <channel_name> [--all] [--clips clip-01,clip-03] [--force-transcribe] [--upload]" >&2
  exit 1
}

if [ $# -lt 2 ]; then
  usage
fi

VIDEO="$1"
CHANNEL="$2"
shift 2

CUT_ARGS=()
FORCE_TRANSCRIBE_ARGS=()
DO_UPLOAD=0

while [ $# -gt 0 ]; do
  case "$1" in
    --all)
      CUT_ARGS=(--all)
      shift
      ;;
    --clips)
      [ $# -ge 2 ] || { echo "--clips needs a value" >&2; usage; }
      CUT_ARGS=(--clips "$2")
      shift 2
      ;;
    --force-transcribe)
      FORCE_TRANSCRIBE_ARGS=(--force-transcribe)
      shift
      ;;
    --upload)
      DO_UPLOAD=1
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      ;;
  esac
done

if [ ! -f "$VIDEO" ]; then
  echo "Video file not found: $VIDEO" >&2
  exit 1
fi

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "Note: OPENROUTER_API_KEY is not set. Fine if '$CHANNEL' uses provider: ollama; required if it uses openrouter." >&2
fi

CLIPPER=(python3 -m clipper)
if command -v clipper >/dev/null 2>&1; then
  CLIPPER=(clipper)
fi

echo "== select =="
"${CLIPPER[@]}" select "$VIDEO" --channel "$CHANNEL" "${FORCE_TRANSCRIBE_ARGS[@]}"

echo
echo "== cut =="
"${CLIPPER[@]}" cut "$VIDEO" "${CUT_ARGS[@]}"

echo
echo "== reframe =="
"${CLIPPER[@]}" reframe "$VIDEO" --channel "$CHANNEL"

echo
echo "== caption =="
"${CLIPPER[@]}" caption "$VIDEO" --channel "$CHANNEL"

echo
echo "== package =="
"${CLIPPER[@]}" package "$VIDEO" --channel "$CHANNEL"

if [ "$DO_UPLOAD" -eq 1 ]; then
  echo
  echo "== upload =="
  "${CLIPPER[@]}" upload "$VIDEO" --channel "$CHANNEL"
fi

VIDEO_ID="$(basename "$VIDEO")"
VIDEO_ID="${VIDEO_ID%.*}"
echo
echo "Done. See output/${VIDEO_ID}/ for finished clips."
