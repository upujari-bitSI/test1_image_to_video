# Image to Video Generator

Interactive Gradio app that turns a still image into a short video clip using
either a **local open-source model** (free, runs on your RTX 3060) or a
**hosted API** (Wan 2.5 I2V and other top models via Replicate).

## Features

- **Local backend**: Stable Video Diffusion XT (Stability AI). The most
  reliable open I2V model that fits in 12 GB VRAM. 25-frame clips at 1024x576.
- **API backend**: One-click access to Wan 2.5 I2V, Wan 2.2, Kling v2.1,
  Hailuo-02 and Seedance via Replicate.
- Single Gradio UI with separate tabs for each backend.

## Why these models for an RTX 3060?

| Model | VRAM (fp16) | Notes |
| --- | --- | --- |
| **Stable Video Diffusion XT** | ~10 GB w/ model offload | Picked - runs well on 3060 12 GB, mature `diffusers` support |
| CogVideoX-5B I2V | ~16 GB (8 GB w/ int8) | Slow on 3060, needs quantization |
| Wan 2.1/2.2 I2V (open weights) | 24 GB+ for 14B | Too large for 3060; use the API tab |
| AnimateDiff | ~8 GB | Lower quality, kept SVD instead |

SVD is image-conditioned only (no text prompt). For text-guided generation use
the API tab.

## Install

Requires Python 3.10+, an NVIDIA GPU with recent drivers, and CUDA 12.x.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install PyTorch with CUDA 12.1 wheels first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open http://127.0.0.1:7860.

The first local generation downloads the SVD-XT weights (~9 GB) into the
HuggingFace cache. Subsequent runs start fast.

## API setup (optional)

1. Create a token at https://replicate.com/account/api-tokens.
2. Either paste it into the API tab, or:

```bash
cp .env.example .env
# edit .env and set REPLICATE_API_TOKEN
```

Pricing is pay-per-run (Wan 2.5 480p/5s is roughly $0.25).

## Troubleshooting downloads (Windows / flaky networks)

The first run pulls ~9 GB from HuggingFace. If you see
`WinError 10054 ... connection forcibly closed` or
`variant=fp16 ... no such modeling files`:

1. Just re-run - downloads now resume from where they stopped.
2. Speed up + stabilize:
   ```bash
   pip install hf_transfer
   set HF_HUB_ENABLE_HF_TRANSFER=1     # PowerShell: $env:HF_HUB_ENABLE_HF_TRANSFER="1"
   ```
3. If your ISP throttles HuggingFace, use a mirror:
   ```bash
   set HF_ENDPOINT=https://hf-mirror.com
   ```
4. As a last resort, delete the partial cache and retry:
   `%USERPROFILE%\.cache\huggingface\hub\models--stabilityai--stable-video-diffusion-img2vid-xt`

## Tips

- If you hit CUDA OOM on the local tab, switch resolution to 768x432 or lower
  the frame count.
- "Motion strength" (motion_bucket_id) controls how much the scene moves -
  raise it for action shots, lower it for portraits.
- Use the "Unload model" button to free VRAM between runs.
