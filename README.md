# Clipper

Turns a long-form video into ready-to-upload YouTube Shorts: AI-assisted clip
selection, frame-accurate cuts, face-tracked 9:16 reframing, word-by-word
highlighted captions, and per-clip metadata + thumbnail.

See `CLAUDE.md` for the architecture/design rules. This doc is the practical
"how do I run it" reference.

## Requirements

- Python 3.11+
- `ffmpeg` on PATH, built with `h264_nvenc` (NVIDIA GPU encode)
- [Ollama](https://ollama.com) running locally, and/or an [OpenRouter](https://openrouter.ai) API key
- An NVIDIA GPU is strongly recommended — both whisper transcription and every
  render stage (cut/reframe/caption) use it

## Install

```bash
pip install -r requirements.txt
```

or, for the `clipper` command instead of `python -m clipper`:

```bash
pip install -e .
```

## Quick start

```bash
clipper select "path/to/video.mp4" --channel example_channel
clipper cut "path/to/video.mp4"
clipper reframe "path/to/video.mp4" --channel example_channel
clipper caption "path/to/video.mp4" --channel example_channel
clipper package "path/to/video.mp4" --channel example_channel
clipper upload "path/to/video.mp4" --channel example_channel   # optional, see YouTube upload setup below
```

Each finished clip lands in `output/<video-id>/<clip-NN>/` with `clip.mp4`,
`meta.json`, and `thumb.jpg` — either drag `clip.mp4` straight into YouTube
Studio yourself, or run `clipper upload` to push it there via the API
(uploads land **private** by default; you still review and publish manually).

Or run all five as one command with `./run_clipper.sh`:

```bash
./run_clipper.sh "path/to/video.mp4" example_channel
```

`cut`'s approval prompt still runs interactively in the middle of it — this
just chains the commands, it doesn't skip the gate. Pass `--all` or
`--clips clip-01,clip-03` to skip that prompt, `--force-transcribe` to
re-run whisper, or `--upload` to also run `clipper upload` at the end.

## Set up a channel first

Every run needs `--channel <name>`, which points at
`config/channels/<name>.yaml`. Copy the template and edit it for your content:

```bash
cp config/channels/example_channel.yaml config/channels/my_channel.yaml
```

Fields that matter most:

- `selection.prompt` — the main lever for clip quality. Be specific about your
  audience and what makes a clip worth cutting for this particular channel.
- `selection.clip_length.{min,max}_seconds` — hard bounds; candidates outside
  this range are rejected automatically. YouTube Shorts' actual platform
  ceiling is 180s, but the engagement sweet spot is 30-60s — a reasonable
  default once a channel has some history is `min: 25, max: 120` with the
  prompt telling the model to prefer 30-60s and only go longer when the
  argument genuinely needs the runway. **For a channel's first real uploads**
  (no watch-history yet), start tighter — `20-30s` — to build a retention
  baseline before trusting the algorithm with longer content; loosen back up
  once YouTube Analytics > Research shows what length actually holds this
  niche's audience. Whatever the bounds, the "no padding" rule matters more
  than hitting a target length: a clip should be as long as its argument
  needs and no longer — stretching a 40s idea to fill 60s hurts retention.
  The hook also has to work in the first ~3 seconds regardless of length,
  since 50-60% of viewers decide whether to swipe away almost instantly.
- `selection.provider` / `selection.model` — `ollama` (local) or `openrouter`
  (hosted, needs `OPENROUTER_API_KEY` set). Connection details live in
  `config/settings.yaml`, not here.
- `reframe.mode` — `face-track` (crops follow a detected speaker),
  `center` (fixed center crop — use this for B-roll/narration with no
  talking head), or `blur-pillarbox` (full frame letterboxed over a blurred
  background). `face-track` automatically falls back to `center` per-clip if
  too few sampled frames have a detectable face.
- `caption.position` — `bottom_safe` (default, stays out of the bottom ~25%
  of the frame — the Shorts UI safe zone), `middle`, or `top_safe`.
- `upload.privacy_status` — `private` (default), `unlisted`, or `public`.
  Videos are never uploaded public by default — you publish manually after
  reviewing in Studio. `upload.category_id` is a YouTube category id
  (default `"22"`, People & Blogs).
- `hashtags` — pool used in every clip's `meta.json` description.

## The six stages

### 1. `select` — transcribe + pick candidates

```bash
clipper select video.mp4 --channel my_channel [--force-transcribe]
```

Transcribes with faster-whisper (word-level timestamps, cached to
`output/<video-id>/transcript.json` — re-runs skip this unless
`--force-transcribe`), then asks the configured LLM to rank clip candidates.
Selection is **never** cached — it always re-calls the LLM, since tuning
`selection.prompt` and re-running to see the effect is the whole point of
this stage. Writes `output/<video-id>/candidates.json` and prints each
candidate's id, score, span, title, and hook.

### 2. `cut` — the approval gate

```bash
clipper cut video.mp4                          # interactive
clipper cut video.mp4 --all                     # render every candidate
clipper cut video.mp4 --clips clip-01,clip-03   # render specific ones
```

Re-displays the candidates from `select` and only renders the ids you
accept. Cuts snap to sentence boundaries from the word-level transcript
(never clips a word mid-sentence) with ~0.3s padding, re-encoded via
`h264_nvenc` for frame accuracy — never stream-copied. Writes
`<clip-id>/cut.mp4` + `cut.json`.

### 3. `reframe` — crop to 9:16

```bash
clipper reframe video.mp4 --channel my_channel [--clips ...] [--force]
```

Processes every clip that has a `cut.mp4`. Writes `<clip-id>/reframed.mp4` +
`reframe.json` (records which mode actually got used, plus face-detection
rate for `face-track`). Skips clips that already have `reframed.mp4` unless
`--force`.

### 4. `caption` — burn in word-level captions

```bash
clipper caption video.mp4 --channel my_channel [--clips ...] [--force]
```

Processes every clip that has a `reframed.mp4`. Writes `<clip-id>/clip.mp4`
(the pipeline's final rendered video) + `captions.ass` + `captions.json`.

### 5. `package` — metadata + thumbnail

```bash
clipper package video.mp4 --channel my_channel [--clips ...] [--force]
```

Processes every clip that has a `clip.mp4`. Writes `<clip-id>/meta.json`
(title/description/hashtags/source timestamp) and `<clip-id>/thumb.jpg`
(grabbed at the moment the hook is actually spoken, found by matching the
LLM's hook line against the clip's own transcript).

### 6. `upload` — push to YouTube

```bash
clipper upload video.mp4 --channel my_channel                        # interactive
clipper upload video.mp4 --channel my_channel --all                  # upload every eligible clip
clipper upload video.mp4 --channel my_channel --clips clip-01,clip-03
clipper upload video.mp4 --channel my_channel --privacy unlisted     # override for this run
```

Requires one-time setup — see **YouTube upload setup** below. Re-displays
the packaged clips and only uploads the ids you accept, same as `cut`. A
clip that already has an `upload.json` is skipped by default (re-uploading
creates a **new**, separate YouTube video, not an update to the old one) —
pass `--force` if that's genuinely what you want. Writes `<clip-id>/upload.json`
with the resulting video id and URL. Sets the video's thumbnail from
`thumb.jpg` after upload; a thumbnail failure doesn't fail the whole upload.

## Output layout

```
output/<video-id>/
├── transcript.json          # from select (cached, word-level timestamps)
├── candidates.json          # from select (regenerated every run)
└── clip-01/
    ├── cut.mp4  cut.json
    ├── reframed.mp4  reframe.json
    ├── clip.mp4  captions.ass  captions.json    # final video
    ├── meta.json  thumb.jpg                      # ready to upload
    └── upload.json                                # after `clipper upload`
```

Every stage reads/writes its own JSON artifact, so any stage can be re-run
independently without redoing earlier ones (e.g. tweak `caption` config and
re-run just `clipper caption --force` without re-cutting or re-reframing).

## Config files

- `config/settings.yaml` — LLM provider connection details (Ollama base URL,
  OpenRouter API key env var name, default models), whisper defaults
  (`large-v3`, device/compute type `auto`), and the shared YouTube OAuth
  client path/scopes. Not channel-specific.
- `config/channels/*.yaml` — everything channel-specific: selection prompt,
  clip length, caption style, reframe mode, upload privacy/category,
  hashtags.

## YouTube upload setup

`clipper upload` needs an OAuth client registered in Google Cloud Console —
this is a one-time, per-Google-account setup step you do yourself (it can't
be automated, since it requires your own account and browser):

1. Create a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable the **YouTube Data API v3** for that project.
3. Configure the OAuth consent screen (External is fine for personal use;
   add your own Google account as a test user if the app stays in "Testing"
   mode — no need to publish it).
4. Create an **OAuth client ID** of type **Desktop app**.
5. Download the client JSON and save it at the path in
   `config/settings.yaml`'s `youtube.client_secrets_path` (default
   `config/youtube_client_secret.json`) — this file is gitignored, never
   commit it.

The first `clipper upload` run for a given channel opens a browser for you
to sign in and grant access; the resulting token is cached at
`.youtube_tokens/<channel>.json` (also gitignored) so later runs don't
prompt again, and refreshes automatically until it's revoked.

**Quota**: the default YouTube Data API quota is 10,000 units/day, and a
single video upload costs ~1,600 units — roughly 6 uploads/day per Google
Cloud project on the default quota. Request a quota increase in the Cloud
Console if you need more.

## Testing

```bash
python -m pytest
```

Fast, no GPU/LLM/network required — LLM calls and ffmpeg/mediapipe are
mocked at the subprocess/provider boundary.

## Tips from real use

- **GPU contention with local models**: if you're running whisper and a
  large local Ollama model back-to-back on a GPU with limited VRAM, the LLM
  can stay resident (Ollama's keep-alive) and starve whisper's next load
  with a CUDA out-of-memory error. `ollama stop <model>` before a
  transcription-heavy run avoids it. This is a non-issue on a GPU with
  enough VRAM for both models at once.
- **Very long transcripts**: local models get meaningfully more likely to
  ignore the "JSON only" instruction and answer in prose as transcript length
  grows — confirmed in practice on this project: a 14B local model failed to
  produce valid JSON on every video over ~30 minutes of source audio (retried
  automatically, still failed), while short videos worked fine. `select_clips`
  retries once automatically, but for long-form source video, route that
  channel to `provider: openrouter` with a strong hosted model — in testing,
  switching fixed every video that had failed locally, first try, no other
  changes needed.
- **`face-track` needs an actual face**: it works well on talking-head
  footage, but lightweight face detectors can occasionally false-positive on
  simple cartoon/logo faces. For narration-over-B-roll content with no
  speaker on screen, use `center` or `blur-pillarbox` instead.
