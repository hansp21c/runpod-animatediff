# Runpod AnimateDiff (Text-to-Video + Motion LoRAs)

Serverless GPU worker that turns a **text prompt** into a short video clip
using **AnimateDiff** — a research-grade motion module that bolts onto any
Stable Diffusion 1.5 or SDXL checkpoint and gives it temporal awareness.
Served via the `diffusers` library on Runpod serverless. Pick a base model,
pick a motion adapter, optionally stack a camera-motion LoRA, get back an
mp4, an animated GIF, or the raw frame list.

> Fills the **text-to-video** slot next to the cousin worker `runpod-svd`
> (which is image-to-video). SVD needs a still image to animate;
> AnimateDiff generates from prose. Use both when you need either
> direction.

---

## Why this exists

The runpod-svd worker is excellent at animating a still image but you have
to *have* that still image first. AnimateDiff goes the other way: feed it
a prompt and it samples a 16-24 frame clip end-to-end from text — no input
image required. That makes it the right tool when:

- You don't have a still and don't want to generate one separately.
- You want camera-motion control via thin trajectory LoRAs (zoom, pan,
  tilt, roll) — SVD has no equivalent.
- You want fast LCM-style generation (4-8 steps with AnimateLCM).
- You want to layer art-style LoRAs (a customizable SD1.5 backbone gives
  you the entire community LoRA ecosystem).

---

## Features

- **Two pipeline families** — `AnimateDiffPipeline` (SD1.5) and
  `AnimateDiffSDXLPipeline` (SDXL beta).
- **Three motion adapters** on SD1.5 (`v1-5-3` default, `v1-5-2`, plus
  `AnimateLCM` for 4-8 step generation) and one on SDXL (`sdxl-beta`).
- **Three base backbones** on SD1.5 (`epiCRealism`, `Realistic Vision v5.1`,
  `SD v1-5`) and SDXL (`stable-diffusion-xl-base-1.0`).
- **Eight camera-motion LoRAs** stackable on SD1.5: zoom-in/out, pan-left/
  right, tilt-up/down, rolling clockwise/anticlockwise.
- **Three schedulers**: `DDIM` (default), `EulerDiscrete`, `LCM` (for
  AnimateLCM).
- **Three output formats**: **mp4** (default, via `diffusers.utils.export_to_video`),
  animated **gif** (via Pillow, no ffmpeg), or **frames** (list of base64 PNGs).
- **Auto base/adapter pairing** + family-mismatch rejection upfront with
  clear errors (no silent silently-broken runs).
- **Pipeline cache** keyed by `(base, adapter, sd_version, dtype, device)`
  so cold-start is paid once per combination.
- **Motion-LoRA state cache** — re-requesting the same LoRA set is a no-op;
  changing the set unloads + reloads (`set_adapters`, NOT `fuse_lora` —
  fuse is irreversible and breaks stacking).
- **Multi-prompt batch** with per-prompt error capture so one bad prompt
  never kills the whole request.

---

## Base + adapter pairing

The `MotionAdapter` is **family-bound** to its base. SD1.5 adapters only
work with SD1.5 bases; the SDXL beta adapter only works with the SDXL base.
The handler rejects mismatches upfront so you don't burn GPU time on a
silently-broken combo.

| Base model                                       | Family | Canonical adapter                                  |
|--------------------------------------------------|--------|----------------------------------------------------|
| `emilianJR/epiCRealism` (default)                | SD1.5  | `guoyww/animatediff-motion-adapter-v1-5-3`         |
| `SG161222/Realistic_Vision_V5.1_noVAE`           | SD1.5  | `guoyww/animatediff-motion-adapter-v1-5-3`         |
| `runwayml/stable-diffusion-v1-5`                 | SD1.5  | `guoyww/animatediff-motion-adapter-v1-5-3`         |
| `stabilityai/stable-diffusion-xl-base-1.0`       | SDXL   | `guoyww/animatediff-motion-adapter-sdxl-beta`      |

If you omit `motion_adapter` in the request, the canonical adapter for the
chosen base is picked automatically. To opt in to off-list HF ids, set the
env var `ANIMATEDIFF_ALLOW_ANY_HF_MODEL=1`.

### Available motion adapters

| Adapter id                                         | Family | Notes |
|----------------------------------------------------|--------|-------|
| `guoyww/animatediff-motion-adapter-v1-5-3` (default)| SD1.5  | Most polished release of the original AnimateDiff series. |
| `guoyww/animatediff-motion-adapter-v1-5-2`         | SD1.5  | Previous revision — slightly more motion, less stability. |
| `wangfuyun/AnimateLCM`                             | SD1.5  | LCM-distilled. Use 4-8 inference steps + CFG 1.0-2.0 + `LCM` scheduler. |
| `guoyww/animatediff-motion-adapter-sdxl-beta`      | SDXL   | Higher resolution but more VRAM. Beta quality. |

---

## Motion LoRA catalog

These eight thin adapters encode **camera trajectories** (not subject
motion) and stack on top of any SD1.5 base + motion adapter. They are
**not** compatible with the SDXL adapter — the handler rejects that
combination explicitly.

| LoRA repo                                                          | Camera move                |
|--------------------------------------------------------------------|----------------------------|
| `guoyww/animatediff-motion-lora-zoom-in`                           | Dolly in toward subject    |
| `guoyww/animatediff-motion-lora-zoom-out`                          | Pull back from subject     |
| `guoyww/animatediff-motion-lora-pan-left`                          | Horizontal pan to the left |
| `guoyww/animatediff-motion-lora-pan-right`                         | Horizontal pan to the right|
| `guoyww/animatediff-motion-lora-tilt-up`                           | Tilt camera upward         |
| `guoyww/animatediff-motion-lora-tilt-down`                         | Tilt camera downward       |
| `guoyww/animatediff-motion-lora-rolling-clockwise`                 | Roll camera clockwise      |
| `guoyww/animatediff-motion-lora-rolling-anticlockwise`             | Roll camera anticlockwise  |

You can stack multiple — e.g. zoom-in + pan-right at half weight each for
a diagonal push-in. Weights are clamped only by what the underlying LoRA
adapter can handle; values in `[0.3, 1.0]` are sane.

---

## Input schema

```jsonc
{
  "input": {
    // Required
    "prompt": "a single red rose covered in dew",
    // OR for batch:
    "prompts": ["prompt A", "prompt B"],

    // Optional negatives (single string applies to all, or aligned list)
    "negative_prompt":  "blurry, low quality, distorted, watermark",
    "negative_prompts": ["neg A", "neg B"],

    // Task selection
    "task": "text_to_video",       // only supported task

    // Base + motion adapter
    "base_model":     "emilianJR/epiCRealism",
    "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
    "fp16":           true,

    // Scheduler ("DDIM" default | "EulerDiscrete" | "LCM")
    "scheduler": "DDIM",

    // Motion LoRAs (SD1.5 only)
    "motion_loras": [
      {"repo": "guoyww/animatediff-motion-lora-zoom-in", "weight": 0.8}
    ],

    // Generation params
    "num_frames":          16,    // 16 default; 24 ok with extra VRAM
    "num_inference_steps": 25,    // 6-8 for LCM
    "guidance_scale":      7.5,   // 1.0-2.0 for LCM
    "width":               512,   // 1024 for SDXL
    "height":              512,   // 1024 for SDXL
    "seed":                null,  // int | null (random when null)

    // Output
    "output_format": "mp4",       // "mp4" | "gif" | "frames"
    "fps":           8,           // applied to mp4 if mp4_fps unset
    "mp4_fps":       8,
    "gif_fps":       8
  }
}
```

---

## Output shape

```jsonc
{
  "results": [
    {
      "prompt":         "a single red rose covered in dew",
      "negative_prompt": null,
      "format":         "mp4",
      "num_frames":     16,
      "fps":            8,
      "width":          512,
      "height":         512,
      "seed_used":      1739471293,
      "scheduler":      "DDIMScheduler",
      "base_model":     "emilianJR/epiCRealism",
      "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
      "motion_loras":   [{"repo": "guoyww/animatediff-motion-lora-zoom-in", "weight": 0.8}],
      "video_b64":      "AAAAGGZ0eX...",        // mp4 or gif (when format != frames)
      "frames_b64":     null                     // list of PNG b64 (when format=frames)
    }
  ],
  "task":           "text_to_video",
  "base_model":     "emilianJR/epiCRealism",
  "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
  "motion_loras":   [{"repo": "...zoom-in", "weight": 0.8}],
  "sd_version":     "sd15",
  "scheduler":      "DDIM",
  "count":          1,
  "fp16":           true,
  "defaults": { /* echoed cfg used for this request */ }
}
```

Each item in `results` carries its own optional `"error"` string when that
particular prompt fails. The batch keeps going.

---

## Example payloads

### 1) Basic 16-frame mp4 (SD1.5 realistic)

```json
{
  "input": {
    "prompt": "a golden retriever puppy running across a sunlit meadow, cinematic, 4k",
    "negative_prompt": "blurry, low quality, distorted, watermark",
    "output_format": "mp4",
    "num_frames": 16,
    "fps": 8,
    "seed": 42
  }
}
```

### 2) Animated GIF at 10 fps

```json
{
  "input": {
    "prompt": "ocean waves crashing on a rocky coastline at sunset",
    "output_format": "gif",
    "num_frames": 16,
    "gif_fps": 10
  }
}
```

### 3) Frames-only — no muxing, you compose the video

```json
{
  "input": {
    "prompt": "a campfire flickering in the woods at night, embers floating up",
    "output_format": "frames",
    "num_frames": 16,
    "num_inference_steps": 20
  }
}
```

### 4) Fast AnimateLCM (6 steps, low CFG, LCM scheduler)

```json
{
  "input": {
    "prompt": "a neon-lit cyberpunk street at night, rainy reflections",
    "motion_adapter": "wangfuyun/AnimateLCM",
    "scheduler": "LCM",
    "num_inference_steps": 6,
    "guidance_scale": 1.5,
    "num_frames": 16,
    "output_format": "mp4"
  }
}
```

### 5) Zoom-in motion LoRA on a tight subject

```json
{
  "input": {
    "prompt": "a single red rose covered in dew, macro shot",
    "motion_loras": [
      {"repo": "guoyww/animatediff-motion-lora-zoom-in", "weight": 0.8}
    ],
    "output_format": "mp4",
    "num_frames": 16
  }
}
```

### 6) Stacked motion LoRAs — diagonal push-in

```json
{
  "input": {
    "prompt": "a vintage train winding through snowy mountains",
    "motion_loras": [
      {"repo": "guoyww/animatediff-motion-lora-zoom-in", "weight": 0.6},
      {"repo": "guoyww/animatediff-motion-lora-pan-right", "weight": 0.5}
    ],
    "output_format": "mp4",
    "num_frames": 16
  }
}
```

### 7) SDXL beta adapter at 1024x1024

```json
{
  "input": {
    "prompt": "a majestic eagle soaring over alpine peaks, ultra detailed",
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "motion_adapter": "guoyww/animatediff-motion-adapter-sdxl-beta",
    "output_format": "mp4",
    "num_frames": 16,
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 25,
    "guidance_scale": 7.5
  }
}
```

### 8) Multi-prompt batch (two gifs at once)

```json
{
  "input": {
    "prompts": [
      "a flock of birds taking off from a lake at sunrise",
      "a sailing ship cresting a large ocean wave"
    ],
    "output_format": "gif",
    "num_frames": 16,
    "gif_fps": 8
  }
}
```

### 9) Long 24-frame clip with custom aspect ratio

```json
{
  "input": {
    "prompt": "a hot air balloon drifting over rolling hills, slow camera pan",
    "output_format": "mp4",
    "num_frames": 24,
    "width": 768,
    "height": 432,
    "fps": 8
  }
}
```

---

## VRAM notes

- **Minimum:** ~12 GB VRAM for SD1.5 AnimateDiff at 512x512, 16 frames.
- **Recommended:** **16-24 GB** for SD1.5 at 24 frames, or for the SDXL
  adapter at 1024x1024. RTX 4090, L4, A10G all work well.
- AnimateLCM (`wangfuyun/AnimateLCM`) is the cheapest path — 6 inference
  steps instead of 25 = ~4x speedup on the same VRAM budget.
- VAE slicing + tiling are enabled by default on load.
- The SDXL beta adapter is more VRAM-hungry than SD1.5; a 24 GB GPU is the
  comfortable floor.

---

## Cousin worker comparison

|                      | **runpod-animatediff** (this worker) | **runpod-svd** (image-to-video) |
|----------------------|--------------------------------------|---------------------------------|
| Driven by            | text prompt                          | a single still image            |
| Input                | `prompt` / `prompts`                 | `image_url` / `image_b64`       |
| Native length        | 16-24 frames                         | 14 or 25 (model-dependent)      |
| Camera-motion control| Eight motion LoRAs (zoom/pan/tilt/roll) | `motion_bucket_id` scalar    |
| Style control        | Swappable SD1.5/SDXL base + LoRAs    | Fixed (Stability SVD checkpoints) |
| Fast preset          | `AnimateLCM` + LCM scheduler (6 steps) | None — fixed inference steps  |
| Output formats       | mp4 / gif / frames                   | mp4 / gif / frames              |
| License              | guoyww research / per-base-model     | Stability AI Non-Commercial     |

Pick AnimateDiff when you want **text-to-video** with camera control or
style swaps. Pick SVD when you have **an existing still** you want to
breathe motion into.

---

## License — IMPORTANT

The handler code in this repository is permissively licensed. The model
weights this worker loads have **different terms**:

- **AnimateDiff motion modules** (`guoyww/animatediff-motion-adapter-*`,
  `guoyww/animatediff-motion-lora-*`) ship under a **research-only**
  license. They are **not** licensed for direct commercial use without
  arrangement with the authors. Check the model card on HuggingFace before
  deploying commercially.
- **AnimateLCM** (`wangfuyun/AnimateLCM`) ships under the LCM adaptation
  license; review the card.
- **Base diffusion models** follow their own licenses:
  - `runwayml/stable-diffusion-v1-5` — CreativeML Open RAIL-M
  - `stabilityai/stable-diffusion-xl-base-1.0` — CreativeML Open RAIL++-M
  - `emilianJR/epiCRealism`, `SG161222/Realistic_Vision_V5.1_noVAE` —
    community SD1.5 fine-tunes; check each card for downstream restrictions.

**You are responsible** for ensuring your use of the weights complies with
their terms — read the LICENSE on every model you load.

---

## Deployment

```bash
# Build the image (CUDA 12.1 base; bundles ffmpeg + torch 2.3.1+cu121)
docker build -t runpod-animatediff:latest .

# Push to your registry of choice, then point a Runpod serverless template
# at the image. Set ANIMATEDIFF_BASE + ANIMATEDIFF_MOTION env vars to
# preload the desired base/adapter combo on cold start.
```

### Environment variables

| Variable                          | Default                                            | Purpose |
|-----------------------------------|----------------------------------------------------|---------|
| `ANIMATEDIFF_BASE`                | `emilianJR/epiCRealism`                            | Default base model preloaded on cold start |
| `ANIMATEDIFF_MOTION`              | `guoyww/animatediff-motion-adapter-v1-5-3`         | Default motion adapter |
| `ANIMATEDIFF_ALLOW_ANY_HF_MODEL`  | (unset)                                            | Set `1` to permit off-list HF model/adapter ids per request |
| `HF_HOME`                         | `/root/.cache/huggingface`                         | HuggingFace cache root — mount a volume to persist weights across container restarts |
| `PYTHONUNBUFFERED`                | `1`                                                | Real-time log output |

---

## Local testing

The CPU-only test suite mocks `torch`, `diffusers`, `diffusers.utils`,
`transformers`, `peft`, and `runpod` via `sys.modules` injection — no GPU
and no network required.

```bash
python3 test_handler.py
# -> ALL TESTS PASSED
```

The suite covers:

- Base + motion-adapter family pairing (SD1.5 base rejects SDXL adapter
  and vice versa; SDXL+SDXL adapter accepted)
- Auto-pairing when only base is provided
- Scheduler selection (DDIM, EulerDiscrete, LCM) and invalid-scheduler
  rejection
- Motion LoRA list validation (unknown repo, missing repo, wrong family),
  load+`set_adapters` flow with multiple LoRAs
- Single prompt vs `prompts` list dispatch
- End-to-end mp4 output (non-empty `video_b64`)
- gif output round-trips through real Pillow (real GIF89a magic bytes)
- frames-only mode returns real PNGs (PNG magic bytes per frame)
- Per-prompt error capture in a batch (one failing prompt does not kill
  the request)
- Pipeline cache reuse vs cache distinction for different adapters
- All numeric pipeline kwargs (`num_frames`, `num_inference_steps`,
  `guidance_scale`, `width`, `height`, `negative_prompt`) reach the pipe
- Top-level metadata echoed (`task`, `count`, `base_model`,
  `motion_adapter`, `sd_version`, `defaults`)
- `image_to_video` task stub redirects to runpod-svd
