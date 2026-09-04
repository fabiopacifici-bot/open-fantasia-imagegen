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

# Turbo quality presets: CFG-free, 1-4 steps max
TURBO_QUALITY_PRESETS = {
    "low":  (512, 512, 1),
    "mid":  (512, 512, 2),
    "high": (768, 768, 4),
}

# Recommended model: BF16 schnell with torchao int8 quantization
# Stays in VRAM, uses native CUDA int8 ops — ~3-5x faster than BF16
DEFAULT_MODEL      = "black-forest-labs/FLUX.1-schnell"
FULL_MODEL_KLEIN   = "black-forest-labs/FLUX.2-klein-base-9B"
FULL_MODEL_SD15    = "stable-diffusion-v1-5/stable-diffusion-v1-5"
Z_IMG_MODEL        = "Zhibei-ai/Z-Img"
SD_TURBO_MODEL     = "stabilityai/sd-turbo"
SDXL_TURBO_MODEL   = "stabilityai/sdxl-turbo"

# GGUF fallback (slower on most GPUs due to CPU dequant, kept for low-VRAM setups)
DEFAULT_GGUF_REPO  = "city96/FLUX.1-schnell-gguf"
DEFAULT_GGUF_FILE  = "flux1-schnell-Q4_K_S.gguf"


def is_gguf(model_id: str) -> bool:
    """Return True if the model identifier points to a GGUF file."""
    return model_id.endswith(".gguf") or ".gguf" in model_id


def resolve_size(quality=None, width=None, height=None, steps=None):
    """Resolve dimensions from quality keyword or explicit values.

    Explicit width/height/steps override the quality preset when provided;
    otherwise the preset (or 512x512x4 default) is used.
    """
    if quality:
        q = quality.lower()
        if q not in QUALITY_PRESETS:
            raise ValueError(f"Unknown quality '{q}'. Choose from: {', '.join(QUALITY_PRESETS)}")
        w, h, s = QUALITY_PRESETS[q]
        return width or w, height or h, steps if steps is not None else s
    return width or 512, height or 512, steps or 4


MODEL_ALIASES = {
    # Standard HF models
    "turbo":       "stabilityai/sd-turbo",
    "turbo-xl":    "stabilityai/sdxl-turbo",
    "schnell":     "black-forest-labs/FLUX.1-schnell",
    "klein":       "black-forest-labs/FLUX.2-klein-base-9B",
    "sd15":        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "z-img":       "Zhibei-ai/Z-Img",
    # GGUF quantized variants
    "schnell-gguf": "city96/FLUX.1-schnell-gguf/flux1-schnell-Q4_K_S.gguf",
}


def get_pipeline(model_id=None, quant="autoquant"):
    """Load image generation pipeline. Supports HF model IDs and GGUF paths/repos.

    Args:
        model_id: HuggingFace model ID, GGUF path/repo, alias, or None (uses DEFAULT_MODEL).
        quant: Quantization mode — "none", "autoquant" (default), or "int4".
    """
    hf_token = os.environ.get("HF_TOKEN")

    if model_id is None:
        model_id = DEFAULT_MODEL

    model_id = MODEL_ALIASES.get(model_id, model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32
    token  = hf_token or True

    print(f"Loading model: {model_id} on {device} (quant={quant})")

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

    elif "z-img" in model_id.lower() or "zhibei" in model_id.lower():
        # ── Z-Img: 6B single-stream diffusion transformer, loaded as FluxPipeline ──
        print(f"Detected Z-Img model ({Z_IMG_MODEL}), loading with FluxPipeline...")
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=token)
        pipe = pipe.to(device)
        pipe = _apply_quant(pipe, quant, device)

    elif "FLUX" in model_id or "flux" in model_id:
        # ── FLUX BF16 + optional quantization ──
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
        pipe = pipe.to(device)
        pipe = _apply_quant(pipe, quant, device)

    elif "sd-turbo" in model_id.lower() or "sdxl-turbo" in model_id.lower():
        # ── SD-Turbo / SDXL-Turbo: CFG-free, 1-4 steps ──
        from diffusers import AutoPipelineForText2Image
        print(f"Detected turbo model ({model_id}), loading with AutoPipelineForText2Image...")
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        )
        pipe = pipe.to(device)

    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, token=token)
        pipe.safety_checker = None  # disable NSFW filter — local use only
        pipe = pipe.to(device)

    return pipe, device


def _apply_quant(pipe, quant: str, device: str):
    """Apply torchao quantization to the transformer in-place."""
    if quant == "autoquant":
        try:
            from torchao.quantization import autoquant
            pipe.transformer = autoquant(pipe.transformer, error_on_unseen=False)
            print("torchao autoquant applied to transformer ✅")
        except Exception as e:
            print(f"torchao autoquant unavailable ({e}), falling back to BF16")
    elif quant == "int4":
        try:
            import torchao
            from torchao.quantization import quantize_
            try:
                # torchao >= 0.4: Int4WeightOnlyQuantizedLinear via quantize_ API
                from torchao.quantization import int4_weight_only
                quantize_(pipe.transformer, int4_weight_only())
                print("torchao int4 weight-only quantization applied to transformer ✅")
            except ImportError:
                # Older torchao fallback
                from torchao.quantization.quant_api import Int4WeightOnlyQuantizedLinear
                torchao.quantization.quantize_(pipe.transformer, Int4WeightOnlyQuantizedLinear())
                print("torchao int4 weight-only quantization (legacy) applied to transformer ✅")
        except Exception as e:
            print(f"torchao int4 unavailable ({e}), falling back to BF16")
    elif quant == "none":
        print("Quantization disabled — running pure BF16")
    else:
        print(f"Unknown quant mode '{quant}', skipping quantization")
    return pipe


def enhance_prompt(prompt):
    """Basic prompt enhancement for better image quality."""
    return (
        "Create a visually striking scene: "
        + prompt
        + ". Use strong contrasts, cinematic lighting, and clear details."
    )


def generate(prompt, output="output.png", model_id=None, enhance=True,
             quality=None, height=None, width=None, steps=None,
             guidance=7.5, seed=42, quant="autoquant"):
    """Generate an image from a text prompt."""
    # Resolve model alias early to detect turbo
    resolved_model = MODEL_ALIASES.get(model_id, model_id) if model_id else None
    is_turbo = resolved_model and ("sd-turbo" in resolved_model.lower() or "sdxl-turbo" in resolved_model.lower())

    if is_turbo:
        # Use turbo presets; resolve_size falls back to QUALITY_PRESETS so we handle it manually
        preset_map = TURBO_QUALITY_PRESETS
        q = (quality or "low").lower()
        if q not in preset_map:
            raise ValueError(f"Unknown quality '{q}'. Choose from: {', '.join(preset_map)}")
        pw, ph, ps = preset_map[q]
        w = width or pw
        h = height or ph
        s = steps if steps is not None else ps
    else:
        w, h, s = resolve_size(quality=quality, width=width, height=height, steps=steps)

    pipe, device = get_pipeline(model_id, quant=quant)

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
    if is_turbo:
        kwargs["guidance_scale"] = 0.0
    elif not is_flux:
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
                        help="Model ID or GGUF path. Default: black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--no-enhance", action="store_true", help="Skip prompt enhancement.")
    parser.add_argument("--quality", type=str, choices=["low", "mid", "high"],
                        help="Quality preset: low (512px/4 steps), mid (768px/8 steps), high (1024px/20 steps).")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", type=str, default="autoquant", choices=["none", "autoquant", "int4"],
                        help="Quantization mode: none, autoquant (default), int4.")
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
        quant=args.quant,
    )
