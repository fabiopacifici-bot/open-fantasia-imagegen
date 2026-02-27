import torch
try:
    from diffusers import Flux2KleinPipeline
except Exception:
    from diffusers import Flux2Pipeline as Flux2KleinPipeline

def enhance_prompt(prompt):
    # Basic enhancement logic; expand as needed
    enhanced = (
        "Create a visually striking scene: " +
        prompt +
        ". Use strong contrasts, cinematic lighting, and clear details."
    )
    return enhanced

device = "cuda" if torch.cuda.is_available() else "cpu"
# use bfloat16 on CUDA for memory savings, otherwise use float32 on CPU
dtype = torch.bfloat16 if device == "cuda" else torch.float32

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=dtype
)
try:
    if device == "cuda":
        pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU
except Exception as e:
    print(f"Warning: enable_model_cpu_offload() failed: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image generator using Flux pipeline.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate the image.")
    parser.add_argument("--output", type=str, default="flux-klein.png", help="Output image file path.")
    args = parser.parse_args()

    raw_prompt = args.prompt
    out = args.output

    prompt = enhance_prompt(raw_prompt)
    print(f"Original prompt: {raw_prompt}")
    print(f"Enhanced prompt: {prompt}")

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device=device).manual_seed(0),
    ).images[0]
    image.save(out)
    print(f"Image saved: {out}")
