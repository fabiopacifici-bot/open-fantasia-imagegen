import torch
from diffusers import StableDiffusionPipeline

class LocalImageGen:
    def __init__(self, model_name="stabilityai/stable-diffusion-2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name)
        self.pipe.to(self.device)

    def generate(self, prompt, out_path):
        image = self.pipe(prompt).images[0]
        image.save(out_path)
        return out_path

if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a black cat sitting on the head of its human"
    out = sys.argv[2] if len(sys.argv) > 2 else "output.png"
    gen = LocalImageGen()
    gen.generate(prompt, out)
    print(f"Image saved: {out}")
