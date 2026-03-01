import os
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    StableDiffusionPipeline,
    Flux2KleinPipeline,
    GGUFQuantizationConfig,
)

# Quality presets: (width, height, steps)
QUALITY_PRESETS = {
    "low":  (512,  512,  4),
    "mid":  (768,  768,  8),
    "high": (1024, 1024, 20),
}

# Recommended model: BF16 schnell with torchao int8 quantization
# Stays in VRAM, uses native CUDA int8 ops — ~3-5x faster than BF16
DEFAULT_MODEL      = "black-forest-labs/FLUX.1-schnell"
FULL_MODEL_KLEIN   = "black-forest-labs/FLUX.2-klein-base-9B"
FULL_MODEL_SD15    = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# GGUF fallback (slower on most GPUs due to CPU dequant, kept for low-VRAM setups)
DEFAULT_GGUF_REPO  = "city96/FLUX.1-schnell-gguf"
DEFAULT_GGUF_FILE  = "flux1-schnell-Q4_K_S.gguf"


def is_gguf(model_id: str) -> bool:
    """Return True if the model identifier points to a GGUF file."""
    return model_id.endswith(".gguf") or ".gguf" in model_id


def resolve_size(quality=None, width=None, height=None, steps=None):
    """Resolve dimensions from quality keyword or explicit values."""
    if quality:
        q = quality.lower()
        if q not in QUALITY_PRESETS:
            raise ValueError(f"Unknown quality '{q}'. Choose from: {', '.join(QUALITY_PRESETS)}")
        w, h, s = QUALITY_PRESETS[q]
        return w, h, steps if steps is not None else s
    return width or 512, height or 512, steps or 4


def get_pipeline(model_id=None):
    """Load image generation pipeline. Supports HF model IDs and GGUF paths/repos."""
    hf_token = os.environ.get("HF_TOKEN")

    if model_id is None:
        model_id = DEFAULT_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    token  = hf_token or True

    print(f"Loading model: {model_id} on {device}")

    if is_gguf(model_id):
        # ── GGUF path: slower on most GPUs (CPU dequant), kept for low-VRAM setups ──
        if "/" in model_id and not os.path.exists(model_id):
            parts    = model_id.split("/")
            repo_id  = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        else:
            repo_id  = None
            filename = model_id

        quant_config = GGUFQuantizationConfig(compute_dtype=dtype)

        if repo_id:
            transformer = FluxTransformer2DModel.from_single_file(
                f"https://huggingface.co/{repo_id}/blob/main/{filename}",
                quantization_config=quant_config,
                torch_dtype=dtype,
                token=token,
            )
        else:
            transformer = FluxTransformer2DModel.from_single_file(
                filename,
                quantization_config=quant_config,
                torch_dtype=dtype,
            )

        pipe = FluxPipeline.from_pretrained(
            DEFAULT_MODEL,
            transformer=transformer,
            torch_dtype=dtype,
            token=token,
        )
        pipe = pipe.to(device)

    elif "klein" in model_id.lower():
        pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
        pipe = pipe.to(device)

    elif "FLUX" in model_id or "flux" in model_id:
        # ── FLUX BF16 + torchao int8 quantization — native CUDA, fast ──
        try:
            from torchao.quantization import autoquant
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
            pipe = pipe.to(device)
            pipe.transformer = autoquant(pipe.transformer, error_on_unseen=False)
            print("torchao autoquant applied to transformer ✅")
        except Exception as e:
            print(f"torchao unavailable ({e}), falling back to BF16")
            pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
            pipe = pipe.to(device)

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

    is_flux = hasattr(pipe, "transformer")
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
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID or GGUF path. Default: city96/FLUX.1-schnell-gguf/flux1-schnell-Q4_K_S.gguf")
    parser.add_argument("--no-enhance", action="store_true", help="Skip prompt enhancement.")
    parser.add_argument("--quality", type=str, choices=["low", "mid", "high"],
                        help="Quality preset: low (512px/4 steps), mid (768px/8 steps), high (1024px/20 steps).")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
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
