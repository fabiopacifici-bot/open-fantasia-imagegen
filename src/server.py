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
    Returns: { "active": "flux"|"qwen-vl"|"qwen-edit"|"none", "flux_loaded": bool, "vl_loaded": bool,
               "model": "...", "device": "cuda", "status": "ready" }

POST /edit — pixel-level image edit via Qwen Image Edit fp8 + Lightning LoRA
    Input: {"image": "path", "instruction": "...", "steps": 4, "cfg": 1.0, "seed": 42}
    Returns: PNG image bytes

GET /edit/status
    Returns: { "loaded": bool, "model": "Qwen-Image-Edit-fp8 + Lightning-4step LoRA" }
"""

import gc
import io
import os
import sys
import math
import logging
import argparse
import threading
import datetime

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
_quant_mode = "none"

# Generation lock — prevents concurrent requests from stacking (returns 503 if busy)
_generate_lock = threading.Lock()

# Qwen2.5-VL edit model — loaded lazily on first /edit call (legacy stub)
_vl_model = None
_vl_processor = None
_EDIT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Qwen Image Edit model — loaded lazily on first /edit call
_edit_model = None
_edit_clip  = None
_edit_vae   = None
_QWEN_EDIT_UNET = "/mnt/e/models/diffusion_models/qwen_image_edit_fp8_e4m3fn.safetensors"
_QWEN_EDIT_CLIP = "/mnt/e/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
_QWEN_EDIT_VAE  = "/mnt/e/models/vae/qwen_image_vae.safetensors"
_QWEN_EDIT_LORA = "/mnt/e/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors"


def load_model(model_id: str, quant: str = "none"):
    global _pipe, _device, _model_id, _quant_mode
    _pipe, _device = get_pipeline(model_id, quant=quant)
    _model_id = model_id
    _quant_mode = quant
    print(f"✅ Model ready: {_model_id} on {_device} (quant={_quant_mode})")


def warmup_model():
    """Run a single low-res warmup inference to compile CUDA kernels at startup."""
    global _pipe, _device
    if _pipe is None or _device != "cuda":
        return
    print("🔥 Warming up CUDA kernels (1 step, 256x256)...")
    try:
        with torch.no_grad():
            gen = torch.Generator(device=_device).manual_seed(0)
            is_flux = hasattr(_pipe, "transformer")
            kwargs = dict(prompt="warmup", height=256, width=256, num_inference_steps=1, generator=gen)
            if not is_flux:
                kwargs["guidance_scale"] = 1.0
            _pipe(**kwargs)
        print("✅ Warmup complete — inference ready")
    except Exception as e:
        print(f"⚠️  Warmup failed (non-fatal): {e}")


def _unload_flux():
    """Offload FLUX pipeline from GPU and free VRAM."""
    global _pipe
    if _pipe is not None:
        logger.info("Swapping FLUX → Qwen edit for edit request")
        _pipe.to("cpu")
        del _pipe
        _pipe = None
        torch.cuda.empty_cache()
        gc.collect()


def _unload_vl():
    """Offload Qwen2.5-VL from GPU and free VRAM."""
    global _vl_model, _vl_processor
    if _vl_model is not None:
        logger.info("Swapping Qwen2.5-VL → FLUX for generate request")
        _vl_model.to("cpu")
        del _vl_model
        del _vl_processor
        _vl_model = None
        _vl_processor = None
        torch.cuda.empty_cache()
        gc.collect()


def _unload_edit():
    """Offload Qwen Image Edit model from GPU and free VRAM."""
    global _edit_model, _edit_clip, _edit_vae
    if _edit_model is not None:
        logger.info("Unloading Qwen Image Edit model")
        _edit_model = None
        _edit_clip = None
        _edit_vae = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Qwen Image Edit unloaded")


def _load_edit():
    """Load Qwen Image Edit fp8 + Lightning LoRA via ComfyUI loaders."""
    global _edit_model, _edit_clip, _edit_vae

    comfyui_path = "/mnt/d/compy/ComfyUI"
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    import comfy.sd
    import comfy.utils
    import comfy.lora
    from comfy.sd import CLIPType

    logger.info(f"Loading Qwen edit UNET from {_QWEN_EDIT_UNET}")
    model = comfy.sd.load_diffusion_model(_QWEN_EDIT_UNET)

    logger.info(f"Applying Lightning LoRA from {_QWEN_EDIT_LORA}")
    lora_sd = comfy.utils.load_torch_file(_QWEN_EDIT_LORA, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    patches = comfy.lora.load_lora(lora_sd, key_map)
    model.add_patches(patches, strength_patch=1.0)

    logger.info(f"Loading Qwen edit CLIP from {_QWEN_EDIT_CLIP}")
    clip = comfy.sd.load_clip(ckpt_paths=[_QWEN_EDIT_CLIP], clip_type=CLIPType.QWEN_IMAGE)

    logger.info(f"Loading Qwen edit VAE from {_QWEN_EDIT_VAE}")
    vae_sd = comfy.utils.load_torch_file(_QWEN_EDIT_VAE, safe_load=True)
    vae = comfy.sd.VAE(sd=vae_sd)

    _edit_model = model
    _edit_clip  = clip
    _edit_vae   = vae
    logger.info("✅ Qwen Image Edit loaded (fp8 + Lightning 4-step LoRA)")


def _run_edit(pil_image, instruction: str, steps: int = 4, strength: float = 0.75, cfg: float = 1.0, seed: int = 42):
    """Run pixel-level image editing via Qwen Image Edit + Lightning LoRA. Returns PIL.Image."""
    import math as _math
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import node_helpers
    import numpy as _np
    from PIL import Image as PILImage

    # Scale to ~1MP
    total_px = 1024 * 1024
    w, h = pil_image.size
    scale = _math.sqrt(total_px / (w * h))
    new_w = round(w * scale / 8) * 8
    new_h = round(h * scale / 8) * 8
    pil_image = pil_image.resize((new_w, new_h), PILImage.LANCZOS)

    img_np = _np.array(pil_image.convert("RGB")).astype(_np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1,H,W,3]

    # Encode reference image to latent
    ref_latent = _edit_vae.encode(img_tensor[:, :, :, :3])

    # Positive conditioning
    images_for_clip = [img_tensor[:, :, :, :3]]
    tokens_pos = _edit_clip.tokenize(instruction, images=images_for_clip)
    cond_pos = _edit_clip.encode_from_tokens_scheduled(tokens_pos)
    cond_pos = node_helpers.conditioning_set_values(
        cond_pos, {"reference_latents": [ref_latent]}, append=True
    )

    # Negative conditioning
    tokens_neg = _edit_clip.tokenize("", images=[])
    cond_neg = _edit_clip.encode_from_tokens_scheduled(tokens_neg)

    # Latent noise
    latent = _edit_vae.encode(img_tensor[:, :, :, :3])
    noise = comfy.sample.prepare_noise(latent, seed, None)

    samples_out = comfy.sample.sample(
        model=_edit_model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="simple",
        positive=cond_pos,
        negative=cond_neg,
        latent_image=latent,
        denoise=strength,
        seed=seed,
    )

    decoded = _edit_vae.decode(samples_out)
    decoded_np = decoded[0].clamp(0, 1).cpu().numpy()
    return PILImage.fromarray((decoded_np * 255).astype("uint8"))

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
    image: str = Field(description="Path to input image (fantasia/ or inbound/ dir)")
    instruction: str = Field(description="Edit instruction, e.g. 'make the person look like a cyberpunk robot'")
    steps: int = Field(default=4, ge=1, le=50, description="Sampling steps")
    strength: float = Field(default=0.75, ge=0.1, le=1.0, description="Edit strength (0.1=subtle, 1.0=full redraw)")
    cfg: float = Field(default=1.0, ge=0.5, le=10.0, description="CFG scale")
    seed: int = Field(default=42, description="Random seed")

    @property
    def safe_image_path(self) -> str:
        """Resolve and validate image path is within allowed directories."""
        allowed_dirs = [
            os.path.realpath(os.path.expanduser("~/.openclaw/media/fantasia")),
            os.path.realpath(os.path.expanduser("~/.openclaw/media/inbound")),
        ]
        resolved = os.path.realpath(os.path.expanduser(self.image))
        if not any(resolved.startswith(d + os.sep) or resolved == d for d in allowed_dirs):
            raise ValueError(f"Image path not allowed: must be within fantasia/ or inbound/")
        return resolved


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    active = "none"
    if _pipe is not None:
        active = "flux"
    elif _vl_model is not None:
        active = "qwen-vl"
    elif _edit_model is not None:
        active = "qwen-edit"
    busy = not _generate_lock.acquire(blocking=False)
    if not busy:
        _generate_lock.release()
    return {
        "active": active,
        "flux_loaded": _pipe is not None,
        "vl_loaded": _vl_model is not None,
        "edit_loaded": _edit_model is not None,
        "model": _model_id,
        "device": _device or "cuda",
        "status": "busy" if busy else "ready",
        "quant": _quant_mode,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    global _pipe

    # Return 503 immediately if another generation is in progress
    acquired = _generate_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy — another generation is in progress. Try again shortly.")

    try:
        return _do_generate(req)
    finally:
        _generate_lock.release()


def _do_generate(req: GenerateRequest):
    global _pipe

    # Swap out Qwen models if loaded
    _unload_vl()
    _unload_edit()

    # Load FLUX if not loaded
    if _pipe is None:
        if _model_id is None:
            raise HTTPException(status_code=503, detail="Model not configured — restart server with --model")
        load_model(_model_id, quant=_quant_mode)

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


# ── Edit endpoint — Qwen Image Edit fp8 + Lightning LoRA ─────────────────────
@app.get("/edit/status")
def edit_status():
    return {"loaded": _edit_model is not None, "model": "Qwen-Image-Edit-fp8 + Lightning-4step LoRA"}


@app.post("/edit")
async def edit_image(req: EditRequest):
    """
    Pixel-level image editing via Qwen Image Edit fp8 + Lightning 4-step LoRA.
    Uses ComfyUI's sampling pipeline (euler + simple scheduler).
    Returns edited PNG bytes; also saves to ~/.openclaw/media/fantasia/<timestamp>_edit.png.
    """
    global _edit_model, _edit_clip, _edit_vae

    # Validate input image path
    try:
        safe_path = req.safe_image_path
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not os.path.exists(safe_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {safe_path}")

    # Swap out other models (16GB VRAM — can't coexist)
    _unload_flux()
    _unload_vl()

    # Lazy load Qwen Image Edit
    if _edit_model is None:
        _load_edit()

    try:
        from PIL import Image as PILImage

        pil_image = PILImage.open(safe_path).convert("RGB")
        result_image = _run_edit(
            pil_image,
            instruction=req.instruction,
            steps=req.steps,
            strength=req.strength,
            cfg=req.cfg,
            seed=req.seed,
        )

        # Save result
        out_dir = os.path.expanduser("~/.openclaw/media/fantasia")
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(out_dir, f"{timestamp}_edit.png")
        result_image.save(save_path)

        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"X-Saved-Path": save_path},
        )

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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--quant",
        default="none",
        choices=["none", "autoquant", "int4"],
        help="Quantization mode for the generation model (default: none — BF16, fastest cold-start)",
    )
    args = parser.parse_args()

    load_model(args.model, quant=args.quant)
    warmup_model()
    uvicorn.run(app, host=args.host, port=args.port)
