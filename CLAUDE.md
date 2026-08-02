# Clipper — long-form video → 9:16 Shorts pipeline

## What this is
A local Python pipeline that takes long-form video, uses AI to identify strong clip candidates, cuts them frame-accurately, reframes to 9:16 (1080×1920), burns in styled captions, and packages each clip with metadata ready for YouTube Shorts upload.

## Environment
- Windows desktop: RTX 4070 Ti Super, Ryzen 7 7800X3D, 64GB RAM
- Python 3.11, ffmpeg with NVENC available on PATH
- GPU encode: use `h264_nvenc` for all re-encodes
- LLM calls: Ollama locally, OpenRouter for higher-quality selection (config-driven, never hardcode a provider)

## Stack
- `faster-whisper` (large-v3) — transcription with word-level timestamps
- `ffmpeg` via subprocess — cutting, cropping, caption burn
- `mediapipe` — face detection for 9:16 crop tracking
- LLM — clip selection + metadata generation, prompts stored per-channel in config

## Repo layout
```
clipper/
├── CLAUDE.md
├── config/                # per-channel YAML profiles
├── clipper/
│   ├── transcribe.py      # video → cached transcript JSON (word timestamps)
│   ├── select.py          # transcript → LLM → ranked clip candidates JSON
│   ├── cut.py             # frame-accurate extraction, sentence-boundary snapping
│   ├── reframe.py         # face-tracked 9:16 crop, smoothed path
│   ├── captions.py        # word-level .ass generation + burn
│   ├── package.py         # output folders: clip.mp4, meta.json, thumb.jpg
│   └── cli.py             # orchestrator with approval gate
└── output/<video-id>/<clip-NN>/
```

## Per-channel config (YAML)
Each channel profile defines: selection prompt + tone guidance, clip length bounds (default 20–90s), caption style (font, colors, position), reframe mode (`face-track` | `center` | `blur-pillarbox`), hashtag pool.

## Build phases (work in this order)
1. **transcribe.py + select.py** — Done when: `clipper select video.mp4 --channel <name>` returns ranked candidates with start/end timestamps, hook line, title, score. Transcripts cached to JSON so re-runs skip whisper.
2. **cut.py** — Done when: cuts never clip a word (snap to sentence boundaries from whisper data, ~0.3s padding) and there are no audio pops. Re-encode, never stream-copy — cuts must be frame-accurate.
3. **reframe.py** — Done when: speaker stays centered through natural movement with no crop jitter. Sample face position per N frames, smooth with exponential moving average, drive ffmpeg crop filter. Fallback mode when no face detected, per channel config.
4. **captions.py** — Done when: word-by-word highlight captions render inside Shorts safe zones (avoid bottom ~25% of frame), styled per channel config.
5. **package.py** — Done when: each clip folder can be dragged straight into YouTube Studio. meta.json: title ≤100 chars, description, hashtags, source timestamp. Thumbnail grabbed at hook moment.
6. **(later) YouTube Data API upload** — do not build until phases 1–5 are trusted.

## Design rules
- Approval gate in cli.py: after selection, present candidates and wait for accept/reject before rendering. This is the prompt-tuning feedback loop — do not make the pipeline fully automatic in v1.
- Every stage reads/writes JSON artifacts so stages can be re-run independently.
- No hardcoded paths, prompts, or channel names — everything channel-specific lives in config/.
- Windows-safe paths (pathlib everywhere, no shell=True with unquoted paths).
- Keep it simple: this is a transcript-and-ffmpeg problem. No frameworks, no web UI in v1.

## Testing
Keep a short reference video in `tests/fixtures/` (30–60s, one speaker). Each stage should have a smoke test runnable without the LLM (mock the selection JSON).
