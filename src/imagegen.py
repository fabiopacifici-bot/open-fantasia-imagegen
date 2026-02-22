import torch
from diffusers import Flux2KleinPipeline

def enhance_prompt(prompt):
    # Basic enhancement logic; expand as needed
    enhanced = (
        "Create a visually striking scene: " +
        prompt +
        ". Use strong contrasts, cinematic lighting, and clear details."
    )
    return enhanced

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-9B", torch_dtype=dtype
)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "A cat holding a sign that says hello world"
    out = sys.argv[2] if len(sys.argv) > 2 else "flux-klein.png"

    enhanced = enhance_prompt(prompt)
    print(f"Original prompt: {prompt}")
    print(f"Enhanced prompt: {enhanced}")
    image = pipe(
        enhanced,
        height=1024,
        width=1024,
        guidance_scale=4.0,
        num_inference_steps=50,
        generator=torch.Generator(device=device).manual_seed(0)
    ).images[0]
    image.save(out)
    print(f"Image saved: {out}")
