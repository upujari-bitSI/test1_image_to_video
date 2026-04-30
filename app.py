"""Interactive image-to-video generator.

Two backends:
  - Local: Stable Video Diffusion XT, runs on RTX 3060 (8-12GB VRAM).
  - API:   Replicate-hosted models (Wan 2.5 I2V, Kling, Hailuo, Seedance, ...).

Run:
    python app.py
"""

from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

import local_model
from api_client import API_MODELS, generate_video_api

load_dotenv()


def _gpu_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return f"CUDA OK - {name} ({vram:.1f} GB VRAM)"
        return "No CUDA GPU detected. Local tab will not work."
    except Exception as e:
        return f"GPU check failed: {e}"


def run_local(image, seed, num_frames, fps, motion_bucket_id, noise_aug, resolution, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image.")
    width, height = (1024, 576) if resolution == "1024x576" else (768, 432)

    def _cb(step, total):
        progress((step + 1) / max(total, 1), desc=f"Denoising step {step + 1}")

    progress(0.0, desc="Loading model (first run downloads ~9GB)...")
    path = local_model.generate_video(
        image=image,
        seed=int(seed),
        num_frames=int(num_frames),
        fps=int(fps),
        motion_bucket_id=int(motion_bucket_id),
        noise_aug_strength=float(noise_aug),
        width=width,
        height=height,
        progress_callback=_cb,
    )
    return path


def run_api(image, prompt, negative_prompt, model_name, duration, api_token):
    if image is None:
        raise gr.Error("Please upload an image.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    try:
        return generate_video_api(
            image=image,
            prompt=prompt,
            model_name=model_name,
            duration=int(duration),
            negative_prompt=negative_prompt,
            api_token=api_token.strip() or None,
        )
    except Exception as e:
        raise gr.Error(str(e))


def unload_local():
    local_model.unload()
    return "Local model unloaded."


with gr.Blocks(title="Image to Video Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Image to Video Generator\n"
        "Pick **Local** for free offline generation on your RTX 3060, "
        "or **API** to call Wan 2.5 / Kling / Hailuo via Replicate."
    )
    gr.Markdown(f"**System:** {_gpu_info()}")

    with gr.Tabs():
        # ----- Local tab -----
        with gr.Tab("Local (Stable Video Diffusion XT)"):
            with gr.Row():
                with gr.Column():
                    l_image = gr.Image(label="Input image", type="pil", height=320)
                    l_resolution = gr.Radio(
                        ["1024x576", "768x432"],
                        value="1024x576",
                        label="Resolution (use 768x432 if you OOM)",
                    )
                    with gr.Row():
                        l_frames = gr.Slider(14, 25, value=25, step=1, label="Frames")
                        l_fps = gr.Slider(4, 12, value=7, step=1, label="FPS")
                    with gr.Row():
                        l_motion = gr.Slider(
                            1, 255, value=127, step=1,
                            label="Motion strength (higher = more motion)",
                        )
                        l_noise = gr.Slider(
                            0.0, 0.2, value=0.02, step=0.01,
                            label="Noise aug (higher = more deviation)",
                        )
                    l_seed = gr.Number(value=42, precision=0, label="Seed")
                    with gr.Row():
                        l_run = gr.Button("Generate", variant="primary")
                        l_unload = gr.Button("Unload model")
                with gr.Column():
                    l_video = gr.Video(label="Result", height=420)
                    l_status = gr.Markdown()
            gr.Markdown(
                "First run downloads the SVD-XT weights (~9 GB). "
                "Generation takes ~2-5 min on an RTX 3060 12GB at 1024x576."
            )
            l_run.click(
                run_local,
                inputs=[l_image, l_seed, l_frames, l_fps, l_motion, l_noise, l_resolution],
                outputs=l_video,
            )
            l_unload.click(unload_local, outputs=l_status)

        # ----- API tab -----
        with gr.Tab("API (Replicate)"):
            with gr.Row():
                with gr.Column():
                    a_image = gr.Image(label="Input image", type="pil", height=320)
                    a_prompt = gr.Textbox(
                        label="Prompt",
                        lines=4,
                        placeholder="Describe the motion and scene...",
                    )
                    a_neg = gr.Textbox(label="Negative prompt", lines=2)
                    a_model = gr.Dropdown(
                        choices=list(API_MODELS.keys()),
                        value="Wan 2.5 I2V (Alibaba)",
                        label="Model",
                    )
                    a_duration = gr.Radio([3, 5, 10], value=5, label="Duration (seconds)")
                    a_token = gr.Textbox(
                        label="Replicate API token (or set REPLICATE_API_TOKEN env var)",
                        type="password",
                        value=os.environ.get("REPLICATE_API_TOKEN", ""),
                    )
                    a_run = gr.Button("Generate", variant="primary")
                with gr.Column():
                    a_video = gr.Video(label="Result", height=420)
            gr.Markdown(
                "Get a token at https://replicate.com/account/api-tokens. "
                "Pricing varies by model - Wan 2.5 480p/5s is roughly $0.25."
            )
            a_run.click(
                run_api,
                inputs=[a_image, a_prompt, a_neg, a_model, a_duration, a_token],
                outputs=a_video,
            )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
