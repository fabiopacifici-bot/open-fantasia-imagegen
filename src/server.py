"""
Open Fantasia — persistent inference server.
Loads the model once, serves /generate requests instantly.

Usage:
    python src/server.py --model stable-diffusion-v1-5/stable-diffusion-v1-5
    python src/server.py  # uses SD 1.5 by default

POST /generate
    { "prompt": "...", "quality": "mid", "seed": 42, "enhance": true }
    Returns: PNG image bytes

GET /health
    Returns: { "model": "...", "device": "cuda" }
"""

import io
import os
import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from imagegen import get_pipeline, enhance_prompt, resolve_size, QUALITY_PRESETS

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Open Fantasia", version="1.0")

# Loaded once at startup
_pipe = None
_device = None
_model_id = None


def load_model(model_id: str):
    global _pipe, _device, _model_id
    _pipe, _device = get_pipeline(model_id)
    _model_id = model_id
    print(f"✅ Model ready: {_model_id} on {_device}")


# ── Request schema ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    quality: str = Field(default="mid", description="low | mid | high")
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float = 7.5
    seed: int = 42
    enhance: bool = True


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model": _model_id, "device": _device, "status": "ready"}


@app.post("/generate")
def generate(req: GenerateRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    w, h, s = resolve_size(
        quality=req.quality,
        width=req.width,
        height=req.height,
        steps=req.steps,
    )

    prompt = req.prompt
    if req.enhance:
        prompt = enhance_prompt(prompt)

    print(f"Generating [{req.quality}] {w}x{h} @ {s} steps — \"{prompt}\"")

    generator = torch.Generator(device=_device).manual_seed(req.seed)
    is_flux = hasattr(_pipe, "transformer")

    kwargs = dict(
        prompt=prompt,
        height=h,
        width=w,
        num_inference_steps=s,
        generator=generator,
    )
    if not is_flux:
        kwargs["guidance_scale"] = req.guidance

    image = _pipe(**kwargs).images[0]

    # Store generated images under the outputs/ directory to separate from working space
    out_dir = os.path.join(os.path.dirname(__file__), "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    import datetime
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(out_dir, filename)
    image.save(path)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Fantasia inference server")
    parser.add_argument(
        "--model",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="HuggingFace model ID to load",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    load_model(args.model)
    uvicorn.run(app, host=args.host, port=args.port)
