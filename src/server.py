"""
Open Fantasia — persistent inference server.
Loads the model once, serves /generate requests instantly.

Usage:
    python src/server.py --model stable-diffusion-v1-5/stable-diffusion-v1-5
    python src/server.py  # uses SD 1.5 by default

POST /generate
    { "prompt": "...", "quality": "mid", "seed": 42, "enhance": true, "quant": "autoquant" }
    Returns: PNG image bytes

GET /health
    Returns: { "model": "...", "device": "cuda", "quant": "autoquant" }

POST /edit   [v2 stub — VL understanding only, pixel editing coming v2.1]
    { "image": "/path/to/image.png", "instruction": "make the sky purple" }
    Returns: { "response": "...", "note": "..." }

GET /edit/status
    Returns: { "loaded": bool, "model": "Qwen2.5-VL-7B-Instruct" }
"""

import io
import os
import logging
import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from imagegen import get_pipeline, enhance_prompt, resolve_size, QUALITY_PRESETS

logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Open Fantasia", version="2.0")

# Loaded once at startup
_pipe = None
_device = None
_model_id = None
_quant_mode = "autoquant"

# Qwen2.5-VL edit model — loaded lazily on first /edit call
_edit_model = None
_edit_processor = None
_EDIT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_model(model_id: str, quant: str = "autoquant"):
    global _pipe, _device, _model_id, _quant_mode
    _pipe, _device = get_pipeline(model_id, quant=quant)
    _model_id = model_id
    _quant_mode = quant
    print(f"✅ Model ready: {_model_id} on {_device} (quant={_quant_mode})")


# ── Request schemas ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    quality: str = Field(default="mid", description="low | mid | high")
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float = 7.5
    seed: int = 42
    enhance: bool = True
    count: int = Field(default=1, ge=1, le=4, description="Number of images to generate (1-4)")
    quant: str = Field(default="autoquant", pattern="^(none|autoquant|int4)$",
                       description="Quantization mode: none | autoquant | int4")


class EditRequest(BaseModel):
    image: str = Field(description="Path to input image")
    instruction: str = Field(description="Edit instruction, e.g. 'make the sky purple'")
    max_new_tokens: int = 512


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model": _model_id, "device": _device, "status": "ready", "quant": _quant_mode}


@app.post("/generate")
def generate(req: GenerateRequest):
    if _pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # If quant in request differs from loaded model, warn (can't hot-swap)
    if req.quant != _quant_mode:
        logger.warning(
            f"Request quant={req.quant} but server loaded with quant={_quant_mode}. "
            "Restart server with --quant flag to change quantization."
        )

    w, h, s = resolve_size(
        quality=req.quality,
        width=req.width,
        height=req.height,
        steps=req.steps,
    )

    prompt = req.prompt
    if req.enhance:
        prompt = enhance_prompt(prompt)

    print(f"Generating [{req.quality}] {w}x{h} @ {s} steps x{req.count} — \"{prompt}\"")

    is_flux = hasattr(_pipe, "transformer")

    kwargs = dict(
        prompt=prompt,
        height=h,
        width=w,
        num_inference_steps=s,
    )
    if not is_flux:
        kwargs["guidance_scale"] = req.guidance
    out_dir = os.path.expanduser("~/.openclaw/media/fantasia")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    import datetime

    paths = []
    for i in range(req.count):
        gen_i = torch.Generator(device=_device).manual_seed(req.seed + i)
        kwargs["generator"] = gen_i
        img = _pipe(**kwargs).images[0]
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
        path = os.path.join(out_dir, filename)
        img.save(path)
        paths.append(path)

    # Return last image as PNG (all are saved to disk)
    buf = io.BytesIO()
    from PIL import Image as PILImage
    PILImage.open(paths[-1]).save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png", headers={"X-Saved-Paths": ",".join(paths)})


# ── Edit endpoint (v2 stub — Qwen2.5-VL visual understanding) ─────────────────
@app.get("/edit/status")
def edit_status():
    return {"loaded": _edit_model is not None, "model": _EDIT_MODEL_ID}


@app.post("/edit")
async def edit_image(req: EditRequest):
    """
    v2 stub: Uses Qwen2.5-VL-7B-Instruct for visual understanding of edit instructions.
    Returns a text description/analysis of the requested edit.
    NOTE: This is NOT pixel-level editing. Diffusion inpainting (true pixel editing) is planned for v2.1.
    """
    global _edit_model, _edit_processor

    # Validate input image exists
    if not os.path.exists(req.image):
        raise HTTPException(status_code=400, detail=f"Image not found: {req.image}")

    # Lazy load Qwen2.5-VL
    if _edit_model is None:
        if _pipe is not None:
            logger.warning(
                "FLUX pipeline is loaded — attempting to load Qwen2.5-VL alongside it. "
                "This may cause OOM on GPUs with < 24GB VRAM."
            )
        logger.info(f"Loading {_EDIT_MODEL_ID} in 4-bit (bitsandbytes)...")
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_4bit=True)
            _edit_processor = AutoProcessor.from_pretrained(_EDIT_MODEL_ID, trust_remote_code=True)
            _edit_model = AutoModelForVision2Seq.from_pretrained(
                _EDIT_MODEL_ID,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto",
            )
            logger.info(f"✅ {_EDIT_MODEL_ID} loaded in 4-bit")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load edit model: {e}")

    # Build chat message with image + instruction
    try:
        from PIL import Image as PILImage
        image = PILImage.open(req.image).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": req.instruction},
                ],
            }
        ]

        # Apply Qwen2.5-VL chat template
        text = _edit_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _edit_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        # Move inputs to model device
        device = next(_edit_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = _edit_model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        response_text = _edit_processor.decode(generated, skip_special_tokens=True)

        return {
            "response": response_text,
            "model": _EDIT_MODEL_ID,
            "note": (
                "v2 stub: This is visual understanding (VL), not pixel-level editing. "
                "True diffusion inpainting is planned for v2.1."
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit inference failed: {e}")


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
    parser.add_argument(
        "--quant",
        default="autoquant",
        choices=["none", "autoquant", "int4"],
        help="Quantization mode for the generation model (default: autoquant)",
    )
    args = parser.parse_args()

    load_model(args.model, quant=args.quant)
    uvicorn.run(app, host=args.host, port=args.port)
