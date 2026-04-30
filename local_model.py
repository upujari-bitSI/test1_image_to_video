"""Local image-to-video generation optimized for RTX 3060 (8-12GB VRAM).

Uses Stable Video Diffusion XT (SVD-XT) by Stability AI - the best supported
open-source I2V model that fits comfortably in 12GB with optimizations enabled,
and runs on 8GB with sequential CPU offload.
"""

from __future__ import annotations

import gc
import os
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

# Mirrors tried in order when the primary HuggingFace endpoint is unreachable.
# hf-mirror.com is a well-known community mirror that is often accessible when
# huggingface.co itself is blocked (common in China and some other regions).
_HF_MIRRORS = [
    os.environ.get("HF_ENDPOINT", ""),   # user override first
    "https://huggingface.co",            # official
    "https://hf-mirror.com",             # community mirror
]

_pipe: StableVideoDiffusionPipeline | None = None


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _try_snapshot(model_id: str, ignore: list[str], endpoint: str) -> str | None:
    """Attempt snapshot_download against one HF endpoint. Returns path or None."""
    from huggingface_hub import snapshot_download

    env_backup = os.environ.get("HF_ENDPOINT")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    elif "HF_ENDPOINT" in os.environ:
        del os.environ["HF_ENDPOINT"]

    try:
        return snapshot_download(
            model_id,
            ignore_patterns=ignore,
            resume_download=True,
            max_workers=2,
        )
    except Exception as exc:
        print(f"[i2v] endpoint {endpoint or 'default'} failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        if env_backup is not None:
            os.environ["HF_ENDPOINT"] = env_backup
        elif "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def _try_local_cache(model_id: str, variant: str | None) -> str | None:
    """Return a cached local snapshot path if one already exists, else None."""
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_id,
            ignore_patterns=[],
            local_files_only=True,
        )
    except Exception:
        return None


def _prefetch_weights(variant: str | None) -> str:
    """Download model weights, trying mirrors on failure.

    Strategy (in order):
    1. User-set HF_ENDPOINT (if any)
    2. Official huggingface.co  (up to 3 retries)
    3. hf-mirror.com community mirror
    4. Already-cached local snapshot (useful when network is fully down)
    """
    ignore = ["*.bin", "*.fp32.safetensors"] if variant == "fp16" else ["*.fp16.safetensors"]

    endpoints = [ep for ep in _HF_MIRRORS if ep]  # drop empty strings

    for endpoint in endpoints:
        for attempt in range(1, 4):
            result = _try_snapshot(MODEL_ID, ignore, endpoint)
            if result:
                return result
            if attempt < 3:
                wait = 2 ** attempt
                print(f"[i2v] retrying in {wait}s (attempt {attempt}/3) ...")
                time.sleep(wait)

    # Last resort: return whatever is in the local cache, even if incomplete.
    cached = _try_local_cache(MODEL_ID, variant)
    if cached:
        print("[i2v] Network unreachable; loading from local cache.")
        return cached

    raise RuntimeError(
        f"Cannot download {MODEL_ID}: all endpoints unreachable and no local cache found.\n\n"
        "Options:\n"
        "  1. Check your internet connection and try again.\n"
        "  2. Use the mirror explicitly:\n"
        "       Windows PowerShell: $env:HF_ENDPOINT='https://hf-mirror.com'\n"
        "       Windows CMD:        set HF_ENDPOINT=https://hf-mirror.com\n"
        "  3. Download the model manually from https://hf-mirror.com/stabilityai/stable-video-diffusion-img2vid-xt\n"
        "     and set HF_HUB_CACHE to the parent folder."
    )


def _load_pipeline() -> StableVideoDiffusionPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Local generation requires an NVIDIA GPU."
        )

    dtype = torch.float16

    # Try fp16 variant first; fall back to default precision if unavailable.
    try:
        local_dir = _prefetch_weights(variant="fp16")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            local_dir,
            torch_dtype=dtype,
            variant="fp16",
            local_files_only=True,
        )
    except (ValueError, OSError) as e:
        print(f"[i2v] fp16 variant unavailable ({e}); using default weights")
        local_dir = _prefetch_weights(variant=None)
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            local_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )

    if _vram_gb() < 10:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    pipe.unet.enable_forward_chunking()
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except AttributeError:
        pass

    _pipe = pipe
    return pipe


def unload() -> None:
    global _pipe
    _pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    target_ratio = width / height
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        image = image.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        image = image.crop((0, top, src_w, top + new_h))
    return image.resize((width, height), Image.LANCZOS)


def generate_video(
    image: Image.Image,
    seed: int = 42,
    num_frames: int = 25,
    fps: int = 7,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    decode_chunk_size: int = 4,
    width: int = 1024,
    height: int = 576,
    progress_callback=None,
) -> str:
    if image is None:
        raise ValueError("An input image is required.")

    pipe = _load_pipeline()
    img = _prepare_image(image, width, height)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    callback_kwargs = {}
    if progress_callback is not None:
        def _cb(pipe_self, step, timestep, kwargs):
            progress_callback(step, num_frames)
            return kwargs
        callback_kwargs["callback_on_step_end"] = _cb

    result = pipe(
        img,
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
        **callback_kwargs,
    )
    frames = result.frames[0]

    out_dir = Path(tempfile.gettempdir()) / "i2v_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"svd_{seed}_{os.urandom(3).hex()}.mp4"
    export_to_video(frames, str(out_path), fps=fps)
    return str(out_path)
