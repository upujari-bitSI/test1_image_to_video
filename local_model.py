"""Local image-to-video generation optimized for RTX 3060 (8-12GB VRAM).

Uses Stable Video Diffusion XT (SVD-XT) by Stability AI - the best supported
open-source I2V model that fits comfortably in 12GB with optimizations enabled,
and runs on 8GB with sequential CPU offload.
"""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"

_pipe: StableVideoDiffusionPipeline | None = None


def _vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / (1024 ** 3)


def _load_pipeline() -> StableVideoDiffusionPipeline:
    global _pipe
    if _pipe is not None:
        return _pipe

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. Local generation requires an NVIDIA GPU."
        )

    dtype = torch.float16
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16",
    )

    vram = _vram_gb()
    if vram < 10:
        # 8GB cards: trade speed for memory
        pipe.enable_sequential_cpu_offload()
    else:
        # RTX 3060 12GB: model offload is faster and still fits
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
    """Free GPU memory by dropping the cached pipeline."""
    global _pipe
    _pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    # Center-crop to target aspect, then resize
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
    """Generate a video clip from a single image. Returns path to mp4."""
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
