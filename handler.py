"""Runpod serverless handler for AnimateDiff (text-to-video).

Wraps `diffusers.AnimateDiffPipeline` (SD1.5 backbone) and
`diffusers.AnimateDiffSDXLPipeline` (SDXL backbone). A `MotionAdapter` is
loaded separately and bolted onto a community SD1.5/SDXL checkpoint —
that's how AnimateDiff turns a still-image model into a video generator.
Optional camera-motion LoRAs (zoom-in/out, pan-left/right, tilt-up/down,
rolling cw/ccw) stack on top of the base pipeline using `set_adapters`
(NOT `fuse_lora` — fuse is irreversible and breaks LoRA stacking).

Output formats:
  - mp4 (default): base64-encoded mp4 bytes, encoded via
    `diffusers.utils.export_to_video` (imageio+ffmpeg under the hood) with
    a manual `ffmpeg` subprocess fallback when the helper is missing.
  - gif: base64-encoded animated GIF via Pillow (no ffmpeg required).
  - frames: list of base64-encoded PNG frames (clients compose their own).

Pipeline cache key includes (base_model, motion_adapter, sd_version, dtype,
device) so cold-start is paid once per combination. Motion-LoRA state is
tracked per-pipeline-id; re-requesting the same LoRA set is a no-op,
changing the set triggers an unload+reload.

Multi-prompt batching: each prompt captures its own error so a bad prompt
does not kill the request. Mismatched base+adapter combinations (e.g.
SDXL base with an SD1.5 adapter) are rejected upfront with a clear error.
"""
from __future__ import annotations

import base64
import io
import os
import random
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import runpod
except Exception:
    runpod = None

try:
    import torch
except Exception as e:
    raise RuntimeError(
        "torch is required. Install a CUDA-enabled torch wheel."
    ) from e

try:
    import diffusers
    from diffusers import (
        AnimateDiffPipeline,
        AnimateDiffSDXLPipeline,
        MotionAdapter,
        LCMScheduler,
        DDIMScheduler,
        EulerDiscreteScheduler,
    )
except Exception as e:
    raise RuntimeError(
        "diffusers is required. `pip install diffusers>=0.27.0`"
    ) from e

try:
    from diffusers.utils import export_to_video as _diffusers_export_to_video
except Exception:
    _diffusers_export_to_video = None

try:
    from diffusers.utils import export_to_gif as _diffusers_export_to_gif
except Exception:
    _diffusers_export_to_gif = None

try:
    import transformers
except Exception as e:
    raise RuntimeError("transformers is required.") from e

try:
    import numpy as np
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Pillow and numpy are required for image I/O."
    ) from e


ALLOWED_BASE_MODELS_SD15 = [
    "emilianJR/epiCRealism",
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "runwayml/stable-diffusion-v1-5",
]


ALLOWED_BASE_MODELS_SDXL = [
    "stabilityai/stable-diffusion-xl-base-1.0",
]


ALLOWED_BASE_MODELS = ALLOWED_BASE_MODELS_SD15 + ALLOWED_BASE_MODELS_SDXL


ALLOWED_MOTION_ADAPTERS_SD15 = [
    "guoyww/animatediff-motion-adapter-v1-5-3",
    "guoyww/animatediff-motion-adapter-v1-5-2",
    "wangfuyun/AnimateLCM",
]


ALLOWED_MOTION_ADAPTERS_SDXL = [
    "guoyww/animatediff-motion-adapter-sdxl-beta",
]


ALLOWED_MOTION_ADAPTERS = ALLOWED_MOTION_ADAPTERS_SD15 + ALLOWED_MOTION_ADAPTERS_SDXL


_AUTO_ADAPTER_FOR_BASE = {
    "emilianJR/epiCRealism": "guoyww/animatediff-motion-adapter-v1-5-3",
    "SG161222/Realistic_Vision_V5.1_noVAE": "guoyww/animatediff-motion-adapter-v1-5-3",
    "runwayml/stable-diffusion-v1-5": "guoyww/animatediff-motion-adapter-v1-5-3",
    "stabilityai/stable-diffusion-xl-base-1.0": "guoyww/animatediff-motion-adapter-sdxl-beta",
}


ALLOWED_MOTION_LORAS = [
    "guoyww/animatediff-motion-lora-zoom-in",
    "guoyww/animatediff-motion-lora-zoom-out",
    "guoyww/animatediff-motion-lora-pan-left",
    "guoyww/animatediff-motion-lora-pan-right",
    "guoyww/animatediff-motion-lora-tilt-up",
    "guoyww/animatediff-motion-lora-tilt-down",
    "guoyww/animatediff-motion-lora-rolling-clockwise",
    "guoyww/animatediff-motion-lora-rolling-anticlockwise",
]


ALLOWED_SCHEDULERS = ["DDIM", "EulerDiscrete", "LCM"]


_SCHEDULER_CLASSES = {
    "DDIM": DDIMScheduler,
    "EulerDiscrete": EulerDiscreteScheduler,
    "LCM": LCMScheduler,
}


ALLOWED_TASKS = ["text_to_video", "image_to_video"]


ALLOWED_FORMATS = {"mp4", "gif", "frames"}


DEFAULT_BASE = os.getenv("ANIMATEDIFF_BASE", "emilianJR/epiCRealism")
DEFAULT_ADAPTER = os.getenv(
    "ANIMATEDIFF_MOTION", "guoyww/animatediff-motion-adapter-v1-5-3"
)


def _truthy(v: Optional[str]) -> bool:
    return bool(v) and v.lower() in ("1", "true", "yes", "y", "on")


ALLOW_ANY_HF_MODEL = _truthy(os.getenv("ANIMATEDIFF_ALLOW_ANY_HF_MODEL", ""))


def _sd_version(model_or_adapter: str) -> str:
    """Return 'sd15' or 'sdxl' for a known base/adapter id, or 'unknown'."""
    if model_or_adapter in ALLOWED_BASE_MODELS_SD15:
        return "sd15"
    if model_or_adapter in ALLOWED_BASE_MODELS_SDXL:
        return "sdxl"
    if model_or_adapter in ALLOWED_MOTION_ADAPTERS_SD15:
        return "sd15"
    if model_or_adapter in ALLOWED_MOTION_ADAPTERS_SDXL:
        return "sdxl"
    return "unknown"


def _resolve_base_and_adapter(
    base_model: Optional[str],
    motion_adapter: Optional[str],
) -> Tuple[str, str, str]:
    """Validate + auto-pair (base, adapter) and return (base, adapter, sd_version).

    Rules:
      - Both default to env-driven defaults when missing.
      - If only base is provided, pick its canonical adapter from the table.
      - Both must belong to the same SD family (sd15 vs sdxl). Mismatch is
        rejected unless ANIMATEDIFF_ALLOW_ANY_HF_MODEL is set (escape hatch).
    """
    base = base_model or DEFAULT_BASE
    adapter = motion_adapter

    if adapter is None:
        adapter = _AUTO_ADAPTER_FOR_BASE.get(base, DEFAULT_ADAPTER)

    if not ALLOW_ANY_HF_MODEL:
        if base not in ALLOWED_BASE_MODELS:
            raise ValueError(
                f"unknown base_model '{base}'. allowed: {ALLOWED_BASE_MODELS} "
                f"(set ANIMATEDIFF_ALLOW_ANY_HF_MODEL=1 to permit any HF id)"
            )
        if adapter not in ALLOWED_MOTION_ADAPTERS:
            raise ValueError(
                f"unknown motion_adapter '{adapter}'. allowed: {ALLOWED_MOTION_ADAPTERS}"
            )
        base_v = _sd_version(base)
        adapter_v = _sd_version(adapter)
        if base_v != adapter_v:
            raise ValueError(
                f"base_model/motion_adapter family mismatch: "
                f"base='{base}' ({base_v}), adapter='{adapter}' ({adapter_v}). "
                f"SD1.5 bases need SD1.5 adapters; SDXL bases need the sdxl-beta adapter."
            )
        return base, adapter, base_v

    v = _sd_version(base)
    if v == "unknown":
        v = _sd_version(adapter)
    if v == "unknown":
        v = "sd15"
    return base, adapter, v


_PIPE_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}
_LORA_LOADED_FOR: Dict[int, Optional[Tuple[Tuple[str, float], ...]]] = {}


def _device_and_dtype(fp16: bool) -> Tuple[str, "torch.dtype"]:
    if torch.cuda.is_available():
        return "cuda", torch.float16 if fp16 else torch.float32
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16 if fp16 else torch.float32
    return "cpu", torch.float32


def _pipeline_class_for_family(sd_version: str):
    if sd_version == "sd15":
        return AnimateDiffPipeline
    if sd_version == "sdxl":
        return AnimateDiffSDXLPipeline
    raise ValueError(f"unknown sd_version '{sd_version}'")


def get_pipeline(
    base_model: str,
    motion_adapter: str,
    sd_version: str,
    fp16: bool,
) -> Dict[str, Any]:
    """Load (or return cached) AnimateDiff pipeline + adapter bundle."""
    device, dtype = _device_and_dtype(fp16)
    key = (base_model, motion_adapter, sd_version, str(dtype), device)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    adapter = MotionAdapter.from_pretrained(motion_adapter, torch_dtype=dtype)
    pipe_cls = _pipeline_class_for_family(sd_version)

    load_kwargs: Dict[str, Any] = {
        "motion_adapter": adapter,
        "torch_dtype": dtype,
    }
    pipe = pipe_cls.from_pretrained(base_model, **load_kwargs)
    pipe = pipe.to(device)

    for hook in ("enable_vae_slicing", "enable_vae_tiling"):
        try:
            getattr(pipe, hook)()
        except Exception:
            pass

    bundle = {
        "pipe": pipe,
        "adapter": adapter,
        "device": device,
        "dtype": dtype,
        "base_model": base_model,
        "motion_adapter": motion_adapter,
        "sd_version": sd_version,
    }
    _PIPE_CACHE[key] = bundle
    return bundle


def apply_scheduler(pipe: Any, scheduler: Optional[str]) -> str:
    """Replace the pipeline scheduler in-place. Returns the active name."""
    if not scheduler:
        return type(pipe.scheduler).__name__
    if scheduler not in _SCHEDULER_CLASSES:
        raise ValueError(
            f"unknown scheduler '{scheduler}'. allowed: {ALLOWED_SCHEDULERS}"
        )
    cls = _SCHEDULER_CLASSES[scheduler]
    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config)
    except Exception as e:
        raise RuntimeError(
            f"failed to set scheduler {scheduler!r}: {e}"
        ) from e
    return scheduler


def _normalize_motion_loras(raw: Any) -> List[Dict[str, Any]]:
    """Validate and normalize a list of motion-lora dicts.

    Accepted shapes:
      - [{"repo": "guoyww/...", "weight": 0.8}, ...]
      - ["guoyww/..."]  (weight defaults to 1.0)
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("motion_loras must be a list")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            repo, weight = item, 1.0
        elif isinstance(item, dict):
            repo = item.get("repo")
            if not isinstance(repo, str) or not repo:
                raise ValueError(f"motion_loras[{i}].repo must be a non-empty string")
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                raise ValueError(f"motion_loras[{i}].weight must be a number")
        else:
            raise ValueError(f"motion_loras[{i}] must be a string or object")
        out.append({"repo": repo, "weight": weight})
    return out


def maybe_load_motion_loras(
    pipe: Any,
    sd_version: str,
    loras: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Load (or reuse cached) motion LoRAs onto a pipeline.

    Skips reload when the same set is already active on this pipeline
    instance. Always uses `load_lora_weights(..., adapter_name=...)` +
    `set_adapters([...], adapter_weights=[...])` so multiple LoRAs stack.
    Never `fuse_lora` — that's irreversible and breaks stacking.
    """
    if not loras:
        pid = id(pipe)
        if _LORA_LOADED_FOR.get(pid):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
            _LORA_LOADED_FOR[pid] = None
        return []

    if sd_version != "sd15":
        raise ValueError(
            "motion_loras are only supported on SD1.5 bases (guoyww series). "
            "Drop motion_loras when using the SDXL adapter."
        )

    if not ALLOW_ANY_HF_MODEL:
        for lora in loras:
            if lora["repo"] not in ALLOWED_MOTION_LORAS:
                raise ValueError(
                    f"unknown motion_lora '{lora['repo']}'. "
                    f"allowed: {ALLOWED_MOTION_LORAS}"
                )

    pid = id(pipe)
    new_state = tuple(sorted((lora["repo"], float(lora["weight"])) for lora in loras))
    if _LORA_LOADED_FOR.get(pid) == new_state:
        return list(loras)

    if _LORA_LOADED_FOR.get(pid):
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

    adapter_names: List[str] = []
    weights: List[float] = []
    for i, lora in enumerate(loras):
        name = f"motion_{i}"
        try:
            pipe.load_lora_weights(lora["repo"], adapter_name=name)
        except Exception as e:
            raise RuntimeError(
                f"failed to load motion_lora '{lora['repo']}': {e}"
            ) from e
        adapter_names.append(name)
        weights.append(float(lora["weight"]))

    try:
        pipe.set_adapters(adapter_names, adapter_weights=weights)
    except Exception as e:
        raise RuntimeError(f"failed to activate motion_loras: {e}") from e

    _LORA_LOADED_FOR[pid] = new_state
    return list(loras)


def _encode_pil_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_pil_jpg_b64(img: Image.Image, quality: int = 90) -> str:
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def frames_to_gif_b64(frames: List[Image.Image], fps: int = 8) -> str:
    """Encode a frame list as an animated GIF (Pillow-based, ffmpeg-free)."""
    if not frames:
        raise ValueError("no frames to encode")
    duration_ms = max(1, int(round(1000.0 / max(1, int(fps)))))
    rgb_frames = [f.convert("RGB") if f.mode != "RGB" else f for f in frames]
    buf = io.BytesIO()
    rgb_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=rgb_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _frames_to_mp4_with_ffmpeg(
    frames: List[Image.Image], out_path: str, fps: int
) -> str:
    """Fallback mp4 encoder: pipe PNG frames into `ffmpeg`.

    Used when `diffusers.utils.export_to_video` is missing or imageio/ffmpeg
    isn't usable on the system. Requires `ffmpeg` on PATH.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not on PATH and diffusers.utils.export_to_video unavailable"
        )
    tmpdir = tempfile.mkdtemp(prefix="animatediff_frames_")
    try:
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmpdir, f"f_{i:05d}.png"))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(int(fps)),
            "-i", os.path.join(tmpdir, "f_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            out_path,
        ]
        subprocess.run(cmd, check=True)
        return out_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def frames_to_mp4_b64(frames: List[Image.Image], fps: int = 8) -> str:
    """Encode frames to mp4 and return as base64."""
    if not frames:
        raise ValueError("no frames to encode")
    fd, out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        wrote = False
        if _diffusers_export_to_video is not None:
            try:
                _diffusers_export_to_video(frames, out_path, fps=int(fps))
                wrote = os.path.getsize(out_path) > 0
            except Exception:
                wrote = False
        if not wrote:
            _frames_to_mp4_with_ffmpeg(frames, out_path, int(fps))
        with open(out_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("ascii")
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def collect_prompts(inp: Dict[str, Any]) -> List[str]:
    prompts: List[str] = []
    if isinstance(inp.get("prompts"), list):
        for p in inp["prompts"]:
            if isinstance(p, str) and p.strip():
                prompts.append(p)
    elif isinstance(inp.get("prompt"), str) and inp["prompt"].strip():
        prompts.append(inp["prompt"])
    return prompts


def collect_negative_prompts(inp: Dict[str, Any], n: int) -> List[Optional[str]]:
    """Return one negative prompt slot per positive prompt (string or list)."""
    if isinstance(inp.get("negative_prompts"), list):
        out: List[Optional[str]] = []
        for i in range(n):
            v = (
                inp["negative_prompts"][i]
                if i < len(inp["negative_prompts"])
                else None
            )
            out.append(v if (isinstance(v, str) and v.strip()) else None)
        return out
    np_str = inp.get("negative_prompt")
    if isinstance(np_str, str) and np_str.strip():
        return [np_str] * n
    return [None] * n


def _make_generator(device: str, seed: Optional[Any]) -> Tuple[Any, int]:
    if seed is None or seed in ("", "random"):
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    if device.startswith("cuda"):
        g = torch.Generator(device=device).manual_seed(seed)
    else:
        g = torch.Generator().manual_seed(seed)
    return g, seed


def _resolve_pipeline_output_frames(out: Any) -> List[Image.Image]:
    """AnimateDiff pipelines return either `frames=[[frames...]]` (list of
    videos) or `frames=[frames...]` depending on diffusers version. Always
    return a flat list of PIL images for the first/only video."""
    raw_frames = getattr(out, "frames", None)
    if raw_frames is None:
        raise RuntimeError("pipeline output missing `.frames`")
    if (
        isinstance(raw_frames, list)
        and raw_frames
        and isinstance(raw_frames[0], list)
    ):
        return list(raw_frames[0])
    return list(raw_frames)


def generate_for_prompt(
    pipe: Any,
    prompt: str,
    negative_prompt: Optional[str],
    cfg: Dict[str, Any],
    device: str,
) -> Tuple[List[Image.Image], int]:
    """Run the pipeline for one prompt; return (frames, seed_used)."""
    generator, seed_used = _make_generator(device, cfg.get("seed"))

    call_kwargs: Dict[str, Any] = {
        "prompt": prompt,
        "num_frames": int(cfg["num_frames"]),
        "num_inference_steps": int(cfg["num_inference_steps"]),
        "guidance_scale": float(cfg["guidance_scale"]),
        "width": int(cfg["width"]),
        "height": int(cfg["height"]),
        "generator": generator,
    }
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    try:
        ctx = torch.no_grad()
    except Exception:
        ctx = torch.no_grad

    with ctx:
        out = pipe(**call_kwargs)

    frames = _resolve_pipeline_output_frames(out)
    return frames, seed_used


def process_prompt(
    bundle: Dict[str, Any],
    prompt: str,
    negative_prompt: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run end-to-end for one prompt; capture errors per prompt."""
    pipe = bundle["pipe"]
    device = bundle["device"]

    out: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
    }

    try:
        frames, seed_used = generate_for_prompt(
            pipe, prompt, negative_prompt, cfg, device
        )
    except Exception as e:
        out["error"] = f"pipeline call failed: {e}"
        return out

    out["num_frames"] = len(frames)
    out["fps"] = int(cfg["mp4_fps" if cfg["output_format"] == "mp4" else "gif_fps"])
    out["seed_used"] = seed_used
    out["base_model"] = bundle["base_model"]
    out["motion_adapter"] = bundle["motion_adapter"]
    out["motion_loras"] = cfg.get("motion_loras") or []
    out["format"] = cfg["output_format"]
    out["width"] = int(cfg["width"])
    out["height"] = int(cfg["height"])
    out["scheduler"] = type(pipe.scheduler).__name__

    fmt = cfg["output_format"]
    try:
        if fmt == "frames":
            out["frames_b64"] = [_encode_pil_png_b64(f) for f in frames]
            out["video_b64"] = None
        elif fmt == "gif":
            out["video_b64"] = frames_to_gif_b64(frames, fps=int(cfg["gif_fps"]))
            out["frames_b64"] = None
        elif fmt == "mp4":
            out["video_b64"] = frames_to_mp4_b64(frames, fps=int(cfg["mp4_fps"]))
            out["frames_b64"] = None
        else:
            out["error"] = f"unknown output_format '{fmt}'"
    except Exception as e:
        out["error"] = f"encoding failed: {e}"

    return out


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    inp = event.get("input") or {}
    task = inp.get("task", "text_to_video")
    if task not in ALLOWED_TASKS:
        return {"error": f"unknown task '{task}'. allowed: {ALLOWED_TASKS}"}

    if task == "image_to_video":
        return {
            "error": (
                "image_to_video is not supported in AnimateDiff (it is a "
                "text-driven motion module). Use the runpod-svd worker for "
                "image-to-video generation."
            )
        }

    prompts = collect_prompts(inp)
    if not prompts:
        return {"error": "Missing 'prompt' or 'prompts' in input."}

    try:
        base_model, motion_adapter, sd_version = _resolve_base_and_adapter(
            inp.get("base_model"),
            inp.get("motion_adapter"),
        )
    except Exception as e:
        return {"error": str(e)}

    output_format = str(inp.get("output_format", "mp4")).lower()
    if output_format not in ALLOWED_FORMATS:
        return {
            "error": (
                f"unknown output_format '{output_format}'. "
                f"allowed: {sorted(ALLOWED_FORMATS)}"
            )
        }

    scheduler = inp.get("scheduler")
    if scheduler is not None and scheduler not in ALLOWED_SCHEDULERS:
        return {
            "error": f"unknown scheduler '{scheduler}'. allowed: {ALLOWED_SCHEDULERS}"
        }

    try:
        motion_loras = _normalize_motion_loras(inp.get("motion_loras"))
    except Exception as e:
        return {"error": str(e)}

    fp16 = bool(inp.get("fp16", True))

    try:
        bundle = get_pipeline(base_model, motion_adapter, sd_version, fp16)
    except Exception as e:
        return {"error": f"pipeline load failed: {e}"}

    try:
        active_scheduler = apply_scheduler(bundle["pipe"], scheduler)
    except Exception as e:
        return {"error": str(e)}

    try:
        active_loras = maybe_load_motion_loras(
            bundle["pipe"], sd_version, motion_loras
        )
    except Exception as e:
        return {"error": str(e)}

    cfg: Dict[str, Any] = {
        "num_frames": int(inp.get("num_frames", 16)),
        "num_inference_steps": int(inp.get("num_inference_steps", 25)),
        "guidance_scale": float(inp.get("guidance_scale", 7.5)),
        "width": int(inp.get("width", 512)),
        "height": int(inp.get("height", 512)),
        "seed": inp.get("seed"),
        "output_format": output_format,
        "mp4_fps": int(inp.get("mp4_fps", inp.get("fps", 8))),
        "gif_fps": int(inp.get("gif_fps", inp.get("fps", 8))),
        "jpeg_quality": int(inp.get("jpeg_quality", 90)),
        "motion_loras": active_loras,
    }

    negatives = collect_negative_prompts(inp, len(prompts))

    results: List[Dict[str, Any]] = []
    for prompt, neg in zip(prompts, negatives):
        try:
            r = process_prompt(bundle, prompt, neg, cfg)
        except Exception as e:
            r = {
                "prompt": prompt,
                "negative_prompt": neg,
                "error": f"unexpected: {e}",
                "base_model": base_model,
                "motion_adapter": motion_adapter,
                "motion_loras": active_loras,
            }
        results.append(r)

    return {
        "results": results,
        "task": task,
        "base_model": base_model,
        "motion_adapter": motion_adapter,
        "motion_loras": active_loras,
        "sd_version": sd_version,
        "scheduler": active_scheduler,
        "count": len(results),
        "fp16": fp16,
        "defaults": {
            "num_frames": cfg["num_frames"],
            "num_inference_steps": cfg["num_inference_steps"],
            "guidance_scale": cfg["guidance_scale"],
            "width": cfg["width"],
            "height": cfg["height"],
            "output_format": cfg["output_format"],
            "mp4_fps": cfg["mp4_fps"],
            "gif_fps": cfg["gif_fps"],
        },
    }


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError(
            "runpod is not installed; cannot start serverless worker."
        )
    runpod.serverless.start({"handler": handler})
