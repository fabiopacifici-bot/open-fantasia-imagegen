import os
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from diffusers import FluxPipeline, StableDiffusionPipeline


def get_pipeline(model_id=None):
    """Load image generation pipeline. Defaults to FLUX.1-schnell if HF_TOKEN is set, else SD 2.1."""
    if model_id is None:
        hf_token = os.environ.get("HF_TOKEN")
        model_id = (
            "black-forest-labs/FLUX.1-schnell" if hf_token
            else "stabilityai/stable-diffusion-2-1"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    hf_token = os.environ.get("HF_TOKEN") or True

    if "FLUX" in model_id or "flux" in model_id:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, token=hf_token)

    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception as e:
            print(f"Warning: enable_model_cpu_offload() failed: {e}")
            pipe = pipe.to(device)
    else:
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
             height=512, width=512, steps=20, guidance=7.5, seed=42):
    """Generate an image from a text prompt."""
    pipe, device = get_pipeline(model_id)

    raw_prompt = prompt
    if enhance:
        prompt = enhance_prompt(prompt)
        print(f"Original: {raw_prompt}")
        print(f"Enhanced: {prompt}")

    generator = torch.Generator(device=device).manual_seed(seed)

    # FLUX pipelines don't use guidance_scale the same way
    is_flux = hasattr(pipe, 'transformer')
    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
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

    parser = argparse.ArgumentParser(description="Local HuggingFace image generator.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--output", type=str, default="output.png", help="Output image file path.")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model ID (optional).")
    parser.add_argument("--no-enhance", action="store_true", help="Skip prompt enhancement.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(
        prompt=args.prompt,
        output=args.output,
        model_id=args.model,
        enhance=not args.no_enhance,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )
