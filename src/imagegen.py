import os
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from diffusers import FluxPipeline, StableDiffusionPipeline

# Quality presets: (width, height, steps)
QUALITY_PRESETS = {
    "low":  (256,  256,  8),   # min 8 steps for FLUX scheduler stability
    "mid":  (512,  512,  20),
    "high": (1024, 1024, 50),
}


def resolve_size(quality=None, width=None, height=None, steps=None):
    """Resolve dimensions from quality keyword or explicit values."""
    if quality:
        q = quality.lower()
        if q not in QUALITY_PRESETS:
            raise ValueError(f"Unknown quality '{q}'. Choose from: {', '.join(QUALITY_PRESETS)}")
        w, h, s = QUALITY_PRESETS[q]
        return w, h, steps if steps is not None else s
    return width or 512, height or 512, steps or 20


def get_pipeline(model_id=None):
    """Load image generation pipeline."""
    hf_token = os.environ.get("HF_TOKEN")
    if model_id is None:
        model_id = (
            "black-forest-labs/FLUX.1-schnell" if hf_token
            else "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    token = hf_token or True

    print(f"Loading model: {model_id} on {device}")
    if "FLUX" in model_id or "flux" in model_id:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
        pipe.safety_checker = None  # disable NSFW filter — local use only

    pipe = pipe.to(device)
    return pipe, device


def enhance_prompt(prompt):
    """Basic prompt enhancement for better image quality."""
    return (
        "Create a visually striking scene: "
        + prompt
        + ". Use strong contrasts, cinematic lighting, and clear details."
    )


def generate(prompt, output="output.png", model_id=None, enhance=True,
             quality=None, height=None, width=None, steps=None,
             guidance=7.5, seed=42):
    """Generate an image from a text prompt."""
    w, h, s = resolve_size(quality=quality, width=width, height=height, steps=steps)

    pipe, device = get_pipeline(model_id)

    if enhance:
        enhanced = enhance_prompt(prompt)
        print(f"Original: {prompt}")
        print(f"Enhanced: {enhanced}")
        prompt = enhanced

    print(f"Generating {w}x{h} at {s} steps...")
    generator = torch.Generator(device=device).manual_seed(seed)

    is_flux = hasattr(pipe, 'transformer')
    kwargs = dict(
        prompt=prompt,
        height=h,
        width=w,
        num_inference_steps=s,
        generator=generator,
    )
    if not is_flux:
        kwargs["guidance_scale"] = guidance

    image = pipe(**kwargs).images[0]
    image.save(output)
    print(f"Image saved: {output}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Open Fantasia — local image generator.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt.")
    parser.add_argument("--output", type=str, default="output.png", help="Output file path.")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model ID.")
    parser.add_argument("--no-enhance", action="store_true", help="Skip prompt enhancement.")
    parser.add_argument("--quality", type=str, choices=["low", "mid", "high"],
                        help="Quality preset: low (256px/4 steps), mid (512px/20 steps), high (1024px/50 steps).")
    parser.add_argument("--width", type=int, default=None, help="Override width (ignored if --quality set).")
    parser.add_argument("--height", type=int, default=None, help="Override height (ignored if --quality set).")
    parser.add_argument("--steps", type=int, default=None, help="Override inference steps.")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        prompt=args.prompt,
        output=args.output,
        model_id=args.model,
        enhance=not args.no_enhance,
        quality=args.quality,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )
