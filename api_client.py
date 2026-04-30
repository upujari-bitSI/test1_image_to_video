"""API-based image-to-video generation via Replicate.

Replicate hosts the best closed and open I2V models behind one API:
- Wan 2.5 I2V (Alibaba, the model in the screenshot)
- Wan 2.2 I2V (open weights, also runnable locally if you have the VRAM)
- Kling v2.1, Hailuo-02, Seedance, etc.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

# Friendly name -> Replicate model slug.
# Slugs without a version suffix resolve to the latest published version.
API_MODELS: dict[str, str] = {
    "Wan 2.5 I2V (Alibaba)": "wan-video/wan-2.5-i2v",
    "Wan 2.2 I2V": "wan-video/wan-2.2-i2v-a14b",
    "Kling v2.1": "kwaivgi/kling-v2.1",
    "Hailuo-02": "minimax/hailuo-02",
    "Seedance 1 Pro": "bytedance/seedance-1-pro",
}


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def generate_video_api(
    image: Image.Image,
    prompt: str,
    model_name: str,
    duration: int = 5,
    negative_prompt: str = "",
    api_token: str | None = None,
) -> str:
    """Run an I2V job on Replicate and download the resulting mp4."""
    if image is None:
        raise ValueError("An input image is required.")
    if not prompt:
        raise ValueError("A text prompt is required for API models.")

    token = api_token or os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Set REPLICATE_API_TOKEN in your environment or paste it in the UI. "
            "Get one at https://replicate.com/account/api-tokens"
        )

    if model_name not in API_MODELS:
        raise ValueError(f"Unknown API model: {model_name}")

    # Import lazily so the local-only path doesn't require the package at import.
    import replicate

    client = replicate.Client(api_token=token)

    inputs: dict = {
        "image": _image_to_data_uri(image),
        "prompt": prompt,
        "duration": int(duration),
    }
    if negative_prompt:
        inputs["negative_prompt"] = negative_prompt

    output = client.run(API_MODELS[model_name], input=inputs)

    # Replicate may return a FileOutput, a URL string, or a list of either.
    if isinstance(output, list):
        output = output[0]

    out_dir = Path(tempfile.gettempdir()) / "i2v_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"api_{int(time.time())}_{os.urandom(3).hex()}.mp4"

    if hasattr(output, "read"):
        with open(out_path, "wb") as f:
            f.write(output.read())
    else:
        url = str(output)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)

    return str(out_path)
