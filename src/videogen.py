"""
Open Fantasia — local video generation with Wan2.1.

Text-to-video generation using the diffusers WanPipeline (Wan-AI/Wan2.1-T2V-1.3B).
The 1.3B model needs only ~8.19 GB VRAM, fits on consumer GPUs, and produces a
5-second 480P clip on an RTX 4090 in about 4 minutes.

Mirrors the structure of imagegen.py so server.py can treat video as another
media sibling. Output is encoded to MP4 (H.264) via imageio-ffmpeg so it plays
in any browser <video> tag.
"""

import os
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Model & defaults ─────────────────────────────────────────────────────────
DEFAULT_VIDEO_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"

# Quality presets: (width, height, num_frames, num_inference_steps)
# Wan2.1 T2V uses 16 fps internally; 81 frames ≈ 5s. 480P is the recommended
# sweet spot for the 1.3B model.
VIDEO_QUALITY_PRESETS = {
    "low":  (480, 480, 49, 20),   # ~3s clip, faster
    "mid":  (480, 832, 81, 30),   # ~5s clip, 480P portrait (recommended)
    "high": (480, 832, 81, 50),   # ~5s clip, more steps = better quality
}

VIDEO_MODEL_ALIASES = {
    "wan":       "Wan-AI/Wan2.1-T2V-1.3B",
    "wan13":     "Wan-AI/Wan2.1-T2V-1.3B",
    "wan2.1":    "Wan-AI/Wan2.1-T2V-1.3B",
    "wan-1.3b":  "Wan-AI/Wan2.1-T2V-1.3B",
}


def is_wan(model_id: str) -> bool:
    """Return True if the model id points to a Wan video model."""
    return "wan" in model_id.lower()


def resolve_video_size(quality=None, width=None, height=None, steps=None, num_frames=None):
    """Resolve video dimensions from quality keyword or explicit values."""
    if quality:
        q = quality.lower()
        if q not in VIDEO_QUALITY_PRESETS:
            raise ValueError(
                f"Unknown video quality '{q}'. Choose from: {', '.join(VIDEO_QUALITY_PRESETS)}"
            )
        w, h, nf, s = VIDEO_QUALITY_PRESETS[q]
        return (
            width or w,
            height or h,
            num_frames or nf,
            steps if steps is not None else s,
        )
    return (
        width or 480,
        height or 832,
        num_frames or 81,
        steps if steps is not None else 30,
    )


def get_video_pipeline(model_id=None):
    """Load the Wan2.1 text-to-video pipeline.

    Args:
        model_id: HuggingFace model ID or alias (defaults to Wan2.1-T2V-1.3B).

    Returns:
        (pipe, device) — the loaded WanPipeline and the compute device.
    """
    hf_token = os.environ.get("HF_TOKEN")

    if model_id is None:
        model_id = DEFAULT_VIDEO_MODEL
    model_id = VIDEO_MODEL_ALIASES.get(model_id, model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    token = hf_token or True

    print(f"Loading video model: {model_id} on {device}")

    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=token,
    )
    pipe = pipe.to(device)

    # Reduce peak memory during the long text-encoder + DiT forward passes.
    # enable_model_cpu_offload moves submodules to/from GPU automatically.
    try:
        pipe.enable_model_cpu_offload()
        print("Video pipeline CPU offload enabled ✅")
    except Exception as e:
        print(f"CPU offload unavailable ({e}), keeping on {device}")

    return pipe, device


def generate_video(prompt, output="output.mp4", model_id=None,
                   quality=None, height=None, width=None, steps=None,
                   num_frames=None, seed=42, negative_prompt="",
                   guidance_scale=6.0, enhance=False, pipe=None, device=None):
    """Generate a short MP4 video from a text prompt using Wan2.1.

    Args:
        prompt: Text description of the video to generate.
        output: Output .mp4 file path.
        model_id: HuggingFace model ID or alias.
        quality: 'low' | 'mid' | 'high' preset.
        height/width: explicit resolution (overrides quality).
        steps: number of inference steps.
        num_frames: number of frames (16 fps → 81 ≈ 5s).
        seed: random seed for reproducibility.
        negative_prompt: negative prompt for guidance.
        guidance_scale: classifier-free guidance scale (Wan recommends 6.0).
        enhance: if True, prepend a cinematic quality booster to the prompt.
        pipe: optional pre-loaded WanPipeline (avoids reloading if already in VRAM).
        device: compute device matching the pre-loaded pipe.

    Returns:
        The output file path.
    """
    w, h, nf, s = resolve_video_size(
        quality=quality, width=width, height=height, steps=steps, num_frames=num_frames
    )

    if pipe is None:
        pipe, device = get_video_pipeline(model_id)

    if enhance:
        prompt = (
            "Create a visually striking video scene: "
            + prompt
            + ". Use smooth motion, strong contrasts, cinematic lighting, and clear details."
        )

    print(f"Generating {w}x{h} video, {nf} frames @ {s} steps...")
    generator = torch.Generator(device=device).manual_seed(seed)

    output_video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "bright colors, overexposed, static, blurred details, subtitles, style",
        height=h,
        width=w,
        num_frames=nf,
        num_inference_steps=s,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    # frames is a list of PIL images → encode to MP4
    _save_mp4(output_video, output, fps=16)
    print(f"Video saved: {output}")
    return output


def _save_mp4(frames, output: str, fps: int = 16):
    """Encode a list of PIL images into an H.264 MP4 via imageio-ffmpeg."""
    import numpy as np

    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio

    # imageio-ffmpeg ships a bundled ffmpeg binary; ensure it's registered.
    try:
        import imageio_ffmpeg
        imageio.plugins.ffmpeg.download()  # no-op if already present
    except Exception:
        pass

    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()
