"""CPU-only tests for the AnimateDiff text-to-video handler.

Mocks `torch`, `runpod`, `transformers`, `peft`, `diffusers`, and
`diffusers.utils` via `sys.modules` injection BEFORE importing handler.py.
The fake `AnimateDiffPipeline` / `AnimateDiffSDXLPipeline` record `__call__`
kwargs onto the instance so we can assert the handler forwarded user
parameters (e.g. num_frames, num_inference_steps, guidance_scale, motion
LoRAs). `PIL` is kept REAL so gif encoding is exercised against the actual
Pillow code path.

Run with `python3 test_handler.py`. Prints "ALL TESTS PASSED" on success.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile
import types
from typing import Any, Dict, List




_runpod = types.ModuleType("runpod")
_serverless = types.ModuleType("runpod.serverless")


def _serverless_start(_arg):
    return None


_serverless.start = _serverless_start
_runpod.serverless = _serverless
sys.modules["runpod"] = _runpod
sys.modules["runpod.serverless"] = _serverless


class _FakeTorchGenerator:
    def __init__(self, *args, **kwargs):
        self._seed = 0
        self.device = kwargs.get("device", "cpu")

    def manual_seed(self, seed: int) -> "_FakeTorchGenerator":
        self._seed = int(seed)
        return self


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def device_count() -> int:
        return 0


class _FakeBackendsMPS:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeBackends:
    mps = _FakeBackendsMPS()


_torch = types.ModuleType("torch")
_torch.Generator = _FakeTorchGenerator
_torch.float16 = "float16"
_torch.float32 = "float32"
_torch.bfloat16 = "bfloat16"
_torch.cuda = _FakeCuda()
_torch.backends = _FakeBackends()
_torch.no_grad = lambda: _FakeNoGrad()
_torch.device = lambda x: x
sys.modules["torch"] = _torch


_transformers = types.ModuleType("transformers")
sys.modules["transformers"] = _transformers


_peft = types.ModuleType("peft")
sys.modules["peft"] = _peft




class _FakeSchedulerConfig:
    """Minimal config object the handler passes through `from_config`."""

    def __init__(self, name: str = "FakeSchedulerConfig"):
        self.name = name


def _make_scheduler_class(name: str):
    """Factory: build a fake scheduler class with `from_config` classmethod.

    The handler does `cls.from_config(pipe.scheduler.config)` — we just need
    each class to return a fresh instance with a `.config` attribute so the
    next scheduler swap still works.
    """

    class _FakeScheduler:
        def __init__(self):
            self.config = _FakeSchedulerConfig(name=f"{name}Config")

        @classmethod
        def from_config(cls, _config):
            return cls()

    _FakeScheduler.__name__ = name
    return _FakeScheduler


_FakeDDIMScheduler = _make_scheduler_class("DDIMScheduler")
_FakeEulerDiscreteScheduler = _make_scheduler_class("EulerDiscreteScheduler")
_FakeLCMScheduler = _make_scheduler_class("LCMScheduler")


class _FakeMotionAdapter:
    """Stand-in for diffusers.MotionAdapter — only needs `from_pretrained`."""

    def __init__(self, repo: str = "fake-adapter"):
        self.repo = repo

    @classmethod
    def from_pretrained(cls, repo: str, **kwargs) -> "_FakeMotionAdapter":
        inst = cls(repo)
        inst.from_pretrained_kwargs = kwargs
        return inst


class _FakeAnimateDiffPipelineBase:
    """Shared fake pipeline for both SD1.5 and SDXL AnimateDiff variants.

    Records:
      - `last_call_kwargs` — captures the dict passed to `__call__`
      - `loaded_loras` — list of (repo, adapter_name) tuples
      - `active_adapters` — last `(names, weights)` tuple from `set_adapters`
      - `lora_unloaded_count` — number of `unload_lora_weights()` calls
    """

    family = "sd15"

    def __init__(self, model_name: str = "fake-base", motion_adapter: Any = None):
        self.model_name = model_name
        self.motion_adapter = motion_adapter
        self.last_call_kwargs: Dict[str, Any] = {}
        self.scheduler = _FakeDDIMScheduler()
        self.unet = types.SimpleNamespace()
        self.vae = types.SimpleNamespace()
        self.loaded_loras: List[Any] = []
        self.active_adapters: Any = None
        self.lora_unloaded_count: int = 0

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "_FakeAnimateDiffPipelineBase":
        inst = cls(model_name, motion_adapter=kwargs.get("motion_adapter"))
        inst.from_pretrained_kwargs = kwargs
        return inst

    def to(self, device: Any) -> "_FakeAnimateDiffPipelineBase":
        self.device = device
        return self

    def enable_model_cpu_offload(self, *args, **kwargs):
        return self

    def enable_sequential_cpu_offload(self, *args, **kwargs):
        return self

    def enable_vae_slicing(self, *args, **kwargs):
        return self

    def enable_vae_tiling(self, *args, **kwargs):
        return self

    def enable_xformers_memory_efficient_attention(self, *args, **kwargs):
        return self

    def load_lora_weights(self, repo: str, adapter_name: str = "default", **kwargs):
        self.loaded_loras.append((repo, adapter_name))
        return self

    def unload_lora_weights(self):
        self.lora_unloaded_count += 1
        self.loaded_loras = []
        self.active_adapters = None
        return self

    def set_adapters(self, names: List[str], adapter_weights: List[float] = None, **kwargs):
        self.active_adapters = (list(names), list(adapter_weights or []))
        return self

    def __call__(self, **kwargs) -> Any:
        from PIL import Image as _PILImage

        self.last_call_kwargs = dict(kwargs)
        num_frames = int(kwargs.get("num_frames") or 16)
        width = int(kwargs.get("width") or 512)
        height = int(kwargs.get("height") or 512)
        frames = []
        for i in range(num_frames):
            img = _PILImage.new(
                "RGB",
                (width, height),
                color=(i * 7 % 256, 128, (255 - i * 5) % 256),
            )
            frames.append(img)
        return types.SimpleNamespace(frames=[frames])


class _FakeAnimateDiffPipeline(_FakeAnimateDiffPipelineBase):
    family = "sd15"


class _FakeAnimateDiffSDXLPipeline(_FakeAnimateDiffPipelineBase):
    family = "sdxl"


def _fake_export_to_video(frames: List[Any], path: str, fps: int = 8) -> str:
    """Write a recognizable placeholder mp4-ish file the handler can re-read."""
    with open(path, "wb") as f:
        f.write(b"FAKE_MP4_BYTES_" + str(len(frames)).encode("ascii"))
    return path


def _fake_export_to_gif(frames: List[Any], path: str = "") -> str:
    """Write a recognizable placeholder gif file. Handler uses Pillow for the
    gif path anyway, so this only exists for the import not to fail."""
    out = path or tempfile.mkstemp(suffix=".gif")[1]
    with open(out, "wb") as f:
        f.write(b"GIF89a_FAKE_" + str(len(frames)).encode("ascii"))
    return out


_diffusers = types.ModuleType("diffusers")
_diffusers.AnimateDiffPipeline = _FakeAnimateDiffPipeline
_diffusers.AnimateDiffSDXLPipeline = _FakeAnimateDiffSDXLPipeline
_diffusers.MotionAdapter = _FakeMotionAdapter
_diffusers.DDIMScheduler = _FakeDDIMScheduler
_diffusers.EulerDiscreteScheduler = _FakeEulerDiscreteScheduler
_diffusers.LCMScheduler = _FakeLCMScheduler
_diffusers_utils = types.ModuleType("diffusers.utils")
_diffusers_utils.export_to_video = _fake_export_to_video
_diffusers_utils.export_to_gif = _fake_export_to_gif
_diffusers.utils = _diffusers_utils
sys.modules["diffusers"] = _diffusers
sys.modules["diffusers.utils"] = _diffusers_utils


import requests as _requests


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_requests_get(url: str, timeout: int = 60):
    return _FakeResponse(b"")


_requests.get = _fake_requests_get




sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handler




def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _clear_state() -> None:
    handler._PIPE_CACHE.clear()
    handler._LORA_LOADED_FOR.clear()




def test_basic_text_to_video_mp4():
    """Default request returns a populated mp4 video_b64."""
    _clear_state()
    out = handler.handler({"input": {"prompt": "a sailboat on calm water"}})
    _assert("results" in out, f"expected results key, got {out}")
    item = out["results"][0]
    _assert("error" not in item, f"unexpected error: {item.get('error')}")
    _assert(item["format"] == "mp4", f"expected mp4, got {item['format']}")
    _assert(item.get("video_b64"), "video_b64 should be non-empty for mp4")
    raw = base64.b64decode(item["video_b64"])
    _assert(raw.startswith(b"FAKE_MP4_BYTES_"), f"expected fake mp4 marker, got {raw[:20]!r}")


def test_base_adapter_pairing_sd15_reject_sdxl_adapter():
    """SD1.5 base with SDXL adapter must be rejected upfront."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "anything",
            "base_model": "emilianJR/epiCRealism",
            "motion_adapter": "guoyww/animatediff-motion-adapter-sdxl-beta",
        }
    })
    _assert("error" in out, f"family mismatch should error, got {out}")
    _assert(
        "mismatch" in out["error"].lower() or "family" in out["error"].lower(),
        f"error should mention family mismatch, got {out['error']}"
    )


def test_base_adapter_pairing_sdxl_reject_sd15_adapter():
    """SDXL base with SD1.5 adapter must be rejected upfront."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "anything",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
        }
    })
    _assert("error" in out, f"family mismatch should error, got {out}")


def test_base_adapter_pairing_sdxl_ok():
    """SDXL base + SDXL adapter is a valid pairing."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "alpine peaks at sunset",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "motion_adapter": "guoyww/animatediff-motion-adapter-sdxl-beta",
            "output_format": "frames",
            "num_frames": 4,
            "width": 64,
            "height": 64,
        }
    })
    _assert("error" not in out, f"valid SDXL pairing should not error: {out}")
    item = out["results"][0]
    _assert(item["base_model"].endswith("xl-base-1.0"), f"base echoed: {item['base_model']}")
    _assert("sdxl" in item["motion_adapter"], f"adapter is sdxl: {item['motion_adapter']}")
    pipe = next(iter(handler._PIPE_CACHE.values()))["pipe"]
    _assert(isinstance(pipe, _FakeAnimateDiffSDXLPipeline), "SDXL pipeline class used")


def test_auto_pair_when_only_base_provided():
    """When motion_adapter is omitted, the canonical adapter is picked."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "base_model": "SG161222/Realistic_Vision_V5.1_noVAE",
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert("error" not in out, f"auto-pair should succeed: {out}")
    _assert(out["motion_adapter"] == "guoyww/animatediff-motion-adapter-v1-5-3",
            f"auto-paired adapter expected, got {out['motion_adapter']}")


def test_scheduler_selection_lcm():
    """LCM scheduler should be installed onto the pipeline."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "lcm fast generation",
            "motion_adapter": "wangfuyun/AnimateLCM",
            "scheduler": "LCM",
            "num_inference_steps": 6,
            "guidance_scale": 1.5,
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert("error" not in out, f"LCM swap failed: {out}")
    item = out["results"][0]
    _assert(item["scheduler"] == "LCMScheduler", f"scheduler should be LCM, got {item['scheduler']}")
    _assert(out["scheduler"] == "LCM", f"top-level echo, got {out['scheduler']}")


def test_scheduler_selection_euler():
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "scheduler": "EulerDiscrete",
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert("error" not in out, f"Euler swap failed: {out}")
    item = out["results"][0]
    _assert(item["scheduler"] == "EulerDiscreteScheduler", f"got {item['scheduler']}")


def test_scheduler_invalid_rejected():
    _clear_state()
    out = handler.handler({
        "input": {"prompt": "x", "scheduler": "NotARealScheduler"}
    })
    _assert("error" in out, f"invalid scheduler should error, got {out}")


def test_motion_lora_list_validation_unknown_repo():
    """Unknown LoRA repos should be rejected."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "motion_loras": [{"repo": "not/a-real-lora", "weight": 1.0}],
        }
    })
    _assert("error" in out, f"unknown LoRA repo should error, got {out}")


def test_motion_lora_rejected_on_sdxl():
    """Motion LoRAs are SD1.5-only; SDXL base should reject."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "motion_adapter": "guoyww/animatediff-motion-adapter-sdxl-beta",
            "motion_loras": [{"repo": "guoyww/animatediff-motion-lora-zoom-in", "weight": 0.8}],
        }
    })
    _assert("error" in out, f"motion LoRA on SDXL should error, got {out}")


def test_motion_lora_load_and_set_adapters():
    """Valid motion LoRAs should be loaded via load_lora_weights + set_adapters."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "a single red rose",
            "motion_loras": [
                {"repo": "guoyww/animatediff-motion-lora-zoom-in", "weight": 0.8},
                {"repo": "guoyww/animatediff-motion-lora-pan-right", "weight": 0.5},
            ],
            "output_format": "frames",
            "num_frames": 4,
            "width": 64,
            "height": 64,
        }
    })
    _assert("error" not in out, f"valid LoRA stack should succeed: {out}")
    pipe = next(iter(handler._PIPE_CACHE.values()))["pipe"]
    _assert(len(pipe.loaded_loras) == 2, f"two LoRAs loaded, got {pipe.loaded_loras}")
    repos_loaded = [r for r, _name in pipe.loaded_loras]
    _assert("guoyww/animatediff-motion-lora-zoom-in" in repos_loaded, "zoom-in present")
    _assert("guoyww/animatediff-motion-lora-pan-right" in repos_loaded, "pan-right present")
    _assert(pipe.active_adapters is not None, "set_adapters was called")
    names, weights = pipe.active_adapters
    _assert(len(names) == 2 and len(weights) == 2, f"two adapter weights, got {pipe.active_adapters}")
    _assert(set(weights) == {0.8, 0.5}, f"weights propagated, got {weights}")
    item = out["results"][0]
    _assert(len(item["motion_loras"]) == 2, "motion_loras echoed in result")


def test_motion_lora_shape_validation():
    """motion_loras must be a list of strings or dicts with repo."""
    _clear_state()
    out_bad = handler.handler({"input": {"prompt": "x", "motion_loras": "not-a-list"}})
    _assert("error" in out_bad, f"non-list motion_loras should error, got {out_bad}")
    out_bad2 = handler.handler({"input": {"prompt": "x", "motion_loras": [{"weight": 1.0}]}})
    _assert("error" in out_bad2, f"missing repo should error, got {out_bad2}")


def test_input_dispatch_single_prompt():
    """Single 'prompt' string -> one result."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "single",
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert(len(out["results"]) == 1, f"single prompt -> 1 result, got {len(out['results'])}")
    _assert(out["count"] == 1, "top-level count == 1")


def test_input_dispatch_prompts_list():
    """'prompts' list -> one result each."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompts": ["one", "two", "three"],
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert(len(out["results"]) == 3, f"three prompts -> 3 results, got {len(out['results'])}")
    _assert(out["count"] == 3, "top-level count == 3")


def test_end_to_end_mp4_non_empty():
    """End-to-end mp4 result has non-empty video_b64."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "cinematic ocean",
            "output_format": "mp4",
            "num_frames": 8,
            "fps": 8,
            "width": 64,
            "height": 64,
        }
    })
    item = out["results"][0]
    _assert("error" not in item, f"no error: {item.get('error')}")
    _assert(item.get("video_b64"), "mp4 video_b64 non-empty")
    raw = base64.b64decode(item["video_b64"])
    _assert(raw.startswith(b"FAKE_MP4_BYTES_"), f"fake mp4 marker, got {raw[:20]!r}")
    _assert(item["num_frames"] == 8, f"num_frames echoed, got {item['num_frames']}")
    _assert(item["fps"] == 8, f"fps echoed, got {item['fps']}")


def test_gif_output_via_pillow():
    """gif mode uses real Pillow on fake PIL frames -> real GIF89a bytes."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "ocean waves",
            "output_format": "gif",
            "num_frames": 8,
            "gif_fps": 10,
            "width": 96,
            "height": 64,
        }
    })
    item = out["results"][0]
    _assert(item["format"] == "gif", f"format should be gif, got {item['format']}")
    _assert(item.get("video_b64"), "video_b64 carries the gif blob")
    raw = base64.b64decode(item["video_b64"])
    _assert(raw[:6] in (b"GIF87a", b"GIF89a"), f"should be real GIF, got {raw[:6]!r}")


def test_frames_only_mode():
    """frames-only output produces list of base64 PNGs."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "a campfire",
            "output_format": "frames",
            "num_frames": 5,
            "width": 64,
            "height": 64,
        }
    })
    item = out["results"][0]
    _assert(item["format"] == "frames", f"format should be frames, got {item['format']}")
    _assert(item.get("frames_b64") is not None, "frames_b64 should exist")
    _assert(len(item["frames_b64"]) == 5, f"expected 5 frames, got {len(item['frames_b64'])}")
    for f in item["frames_b64"]:
        raw = base64.b64decode(f)
        _assert(raw[:4] == b"\x89PNG", f"each frame should be a real PNG, got {raw[:4]!r}")
    _assert(item.get("video_b64") is None, "frames-only should not produce video_b64")


def test_per_prompt_error_capture():
    """A failing prompt should not kill the whole batch."""
    _clear_state()

    orig_call = _FakeAnimateDiffPipeline.__call__
    call_count = {"n": 0}

    def flaky_call(self, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated pipeline failure")
        return orig_call(self, **kwargs)

    _FakeAnimateDiffPipeline.__call__ = flaky_call
    try:
        out = handler.handler({
            "input": {
                "prompts": ["ok one", "this will fail", "ok three"],
                "output_format": "frames",
                "num_frames": 2,
                "width": 32,
                "height": 32,
            }
        })
        _assert(len(out["results"]) == 3, f"3 entries, got {len(out['results'])}")
        _assert("error" not in out["results"][0], f"first ok, got {out['results'][0].get('error')}")
        _assert("error" in out["results"][1], f"second should error per-prompt, got {out['results'][1]}")
        _assert("error" not in out["results"][2], f"third ok, got {out['results'][2].get('error')}")
    finally:
        _FakeAnimateDiffPipeline.__call__ = orig_call


def test_seed_is_echoed():
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "seed": 1234,
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    item = out["results"][0]
    _assert(item["seed_used"] == 1234, f"seed echoed, got {item['seed_used']}")


def test_pipeline_cache_reuse():
    """Same (base, adapter) combo should reuse a single cached pipeline."""
    _clear_state()
    handler.handler({
        "input": {"prompt": "a", "output_format": "frames", "num_frames": 2, "width": 32, "height": 32}
    })
    size_after_first = len(handler._PIPE_CACHE)
    handler.handler({
        "input": {"prompt": "b", "output_format": "frames", "num_frames": 2, "width": 32, "height": 32}
    })
    size_after_second = len(handler._PIPE_CACHE)
    _assert(size_after_first == 1 and size_after_second == 1,
            f"cache should reuse, sizes: {size_after_first}/{size_after_second}")


def test_pipeline_cache_distinct_for_different_adapter():
    """Switching motion adapter should create a new cache entry."""
    _clear_state()
    handler.handler({
        "input": {
            "prompt": "x",
            "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-3",
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    handler.handler({
        "input": {
            "prompt": "x",
            "motion_adapter": "wangfuyun/AnimateLCM",
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert(len(handler._PIPE_CACHE) == 2,
            f"two distinct adapters -> two cache entries, got {len(handler._PIPE_CACHE)}")


def test_missing_prompt_rejected():
    _clear_state()
    out = handler.handler({"input": {}})
    _assert("error" in out, f"missing prompt should error, got {out}")


def test_invalid_output_format_rejected():
    _clear_state()
    out = handler.handler({"input": {"prompt": "x", "output_format": "mkv"}})
    _assert("error" in out, f"unknown output_format should error, got {out}")


def test_kwargs_forwarded_to_pipeline():
    """num_frames, num_inference_steps, guidance_scale, width, height reach the pipe."""
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "num_frames": 12,
            "num_inference_steps": 18,
            "guidance_scale": 6.5,
            "width": 256,
            "height": 384,
            "negative_prompt": "blurry",
            "output_format": "frames",
        }
    })
    _assert("error" not in out, f"unexpected error: {out}")
    pipe = next(iter(handler._PIPE_CACHE.values()))["pipe"]
    kw = pipe.last_call_kwargs
    _assert(kw.get("num_frames") == 12, f"num_frames, got {kw.get('num_frames')}")
    _assert(kw.get("num_inference_steps") == 18, f"num_inference_steps, got {kw.get('num_inference_steps')}")
    _assert(kw.get("guidance_scale") == 6.5, f"guidance_scale, got {kw.get('guidance_scale')}")
    _assert(kw.get("width") == 256, f"width, got {kw.get('width')}")
    _assert(kw.get("height") == 384, f"height, got {kw.get('height')}")
    _assert(kw.get("negative_prompt") == "blurry", f"negative_prompt, got {kw.get('negative_prompt')}")


def test_top_level_metadata_present():
    _clear_state()
    out = handler.handler({
        "input": {
            "prompt": "x",
            "output_format": "frames",
            "num_frames": 2,
            "width": 32,
            "height": 32,
        }
    })
    _assert(out.get("task") == "text_to_video", f"task echoed, got {out.get('task')}")
    _assert(out.get("count") == 1, f"count echoed, got {out.get('count')}")
    _assert(out.get("base_model"), "base_model echoed")
    _assert(out.get("motion_adapter"), "motion_adapter echoed")
    _assert(out.get("sd_version") in ("sd15", "sdxl"), f"sd_version echoed, got {out.get('sd_version')}")
    _assert("defaults" in out, "defaults dict echoed")


def test_image_to_video_task_stub_rejected():
    """The image_to_video stub should redirect callers to runpod-svd."""
    _clear_state()
    out = handler.handler({"input": {"prompt": "x", "task": "image_to_video"}})
    _assert("error" in out, f"image_to_video should error, got {out}")
    _assert("svd" in out["error"].lower() or "image" in out["error"].lower(),
            f"error should mention SVD or image, got {out['error']}")




TESTS = [
    test_basic_text_to_video_mp4,
    test_base_adapter_pairing_sd15_reject_sdxl_adapter,
    test_base_adapter_pairing_sdxl_reject_sd15_adapter,
    test_base_adapter_pairing_sdxl_ok,
    test_auto_pair_when_only_base_provided,
    test_scheduler_selection_lcm,
    test_scheduler_selection_euler,
    test_scheduler_invalid_rejected,
    test_motion_lora_list_validation_unknown_repo,
    test_motion_lora_rejected_on_sdxl,
    test_motion_lora_load_and_set_adapters,
    test_motion_lora_shape_validation,
    test_input_dispatch_single_prompt,
    test_input_dispatch_prompts_list,
    test_end_to_end_mp4_non_empty,
    test_gif_output_via_pillow,
    test_frames_only_mode,
    test_per_prompt_error_capture,
    test_seed_is_echoed,
    test_pipeline_cache_reuse,
    test_pipeline_cache_distinct_for_different_adapter,
    test_missing_prompt_rejected,
    test_invalid_output_format_rejected,
    test_kwargs_forwarded_to_pipeline,
    test_top_level_metadata_present,
    test_image_to_video_task_stub_rejected,
]


def main() -> int:
    failures: List[str] = []
    for t in TESTS:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:
            failures.append(f"{t.__name__}: {e}")
            print(f"  FAIL {t.__name__}: {e}")
    if failures:
        print(f"\n{len(failures)} test(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
