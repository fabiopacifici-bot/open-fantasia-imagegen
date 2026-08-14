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
    Returns: { "active": "flux"|"qwen-vl"|"none", "flux_loaded": bool, "vl_loaded": bool,
               "model": "...", "device": "cuda", "status": "ready" }

    Input: {"image": "path", "instruction": "...", "steps": 4, "cfg": 1.0, "seed": 42}
    Returns: PNG image bytes

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

import psutil
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from typing import Optional
from pydantic import BaseModel, Field

from imagegen import (
    get_pipeline,
    enhance_prompt,
    resolve_size,
    QUALITY_PRESETS,
    MODEL_ALIASES,
)

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

# ── System RAM guard ─────────────────────────────────────────────────────────
# Reject a generation when free system RAM (plus estimated headroom) would drop
# below this threshold. Stops model/page-cache spill from starving the OS.
MIN_FREE_RAM_GB = float(os.environ.get("FANTASIA_MIN_FREE_RAM_GB", "4"))

# Hard caps on a single generation request — guard against accidental OOM from
# huge resolutions / step counts / batch sizes.
MAX_WIDTH = int(os.environ.get("FANTASIA_MAX_WIDTH", "1024"))
MAX_HEIGHT = int(os.environ.get("FANTASIA_MAX_HEIGHT", "1024"))
MAX_STEPS = int(os.environ.get("FANTASIA_MAX_STEPS", "20"))
MAX_COUNT = int(os.environ.get("FANTASIA_MAX_COUNT", "4"))

# Enable pipeline CPU offload so activations stay on the GPU instead of spilling
# into system RAM during generation. Off by default (small perf cost) — set to 1
# to turn on if your system still runs low on RAM.
ENABLE_CPU_OFFLOAD = os.environ.get("FANTASIA_CPU_OFFLOAD", "0") == "1"

# Qwen2.5-VL model — loaded lazily on first /vl call
_vl_model = None
_vl_processor = None
_EDIT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Qwen Image Edit pipeline — loaded lazily on first /edit call
_edit_pipe = None
_EDIT_PIPE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


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
            kwargs = dict(
                prompt="warmup",
                height=256,
                width=256,
                num_inference_steps=1,
                generator=gen,
            )
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
    """Offload Qwen Image Edit pipeline from GPU and free VRAM."""
    global _edit_pipe
    if _edit_pipe is not None:
        logger.info("Unloading Qwen Image Edit pipeline")
        _edit_pipe.to("cpu")
        del _edit_pipe
        _edit_pipe = None
        torch.cuda.empty_cache()
        gc.collect()


def _unload_flux_for_swap():
    global _pipe
    if _pipe is not None:
        _pipe.to("cpu")
        del _pipe
        _pipe = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("VRAM freed for model swap")


def _resolve_model_alias(model_str: str) -> str:
    return MODEL_ALIASES.get(model_str, model_str)


def _free_ram_gb() -> float:
    """Return available system RAM in GB (psutil.available — includes reclaimable cache)."""
    try:
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        # Fallback: don't block generation if psutil is unavailable
        return float("inf")


def _check_ram_guard() -> None:
    """Raise HTTP 503 if free system RAM would drop below the safety floor."""
    free = _free_ram_gb()
    if free < MIN_FREE_RAM_GB:
        raise HTTPException(
            status_code=503,
            detail=(
                f"System RAM too low to generate safely ({free:.1f} GB free, "
                f"need >= {MIN_FREE_RAM_GB:.1f} GB). Free up memory and retry."
            ),
        )


def _enable_offload(pipe) -> None:
    """Best-effort CPU offload for pipelines that support it, to stop RAM spill."""
    if not ENABLE_CPU_OFFLOAD:
        return
    try:
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            logger.info("CPU offload enabled on pipeline")
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            logger.info("Sequential CPU offload enabled on pipeline")
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
    except Exception as e:
        logger.warning(f"Could not enable CPU offload: {e}")


def _load_edit():
    """Load QwenImageEditPipeline from 2511 cache (full BF16, CPU offload for 16GB VRAM)."""
    global _edit_pipe
    from diffusers import QwenImageEditPipeline

    logger.info(f"Loading QwenImageEditPipeline from {_EDIT_PIPE_MODEL_ID}...")
    _edit_pipe = QwenImageEditPipeline.from_pretrained(
        _EDIT_PIPE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    _edit_pipe.enable_model_cpu_offload()
    logger.info("✅ Qwen Image Edit pipeline ready (CPU offload enabled)")


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
    count: int = Field(
        default=1, ge=1, le=4, description="Number of images to generate (1-4)"
    )
    quant: str = Field(
        default="autoquant",
        pattern="^(none|autoquant|int4)$",
        description="Quantization mode: none | autoquant | int4",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model alias or full HF ID; None = use startup default",
    )


@app.post("/generate")
def generate(req: GenerateRequest):
    global _pipe

    # Return 503 immediately if another generation is in progress
    acquired = _generate_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="Server busy — another generation is in progress. Try again shortly.",
        )

    try:
        return _do_generate(req)
    finally:
        _generate_lock.release()


def _do_generate(req: GenerateRequest):
    global _pipe

    # Reject early if the system is running low on free RAM
    _check_ram_guard()

    # Enforce hard caps on a single generation
    if (req.width or 512) > MAX_WIDTH or (req.height or 512) > MAX_HEIGHT:
        raise HTTPException(
            status_code=400,
            detail=f"Resolution too large (max {MAX_WIDTH}x{MAX_HEIGHT}px).",
        )
    if (req.steps or 4) > MAX_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many inference steps (max {MAX_STEPS}).",
        )
    if req.count > MAX_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images per request (max {MAX_COUNT}).",
        )

    # Swap out Qwen models if loaded
    _unload_vl()

    # Resolve requested model (alias or full ID)
    requested_id = _resolve_model_alias(req.model) if req.model else _model_id

    if requested_id is None:
        raise HTTPException(
            status_code=503, detail="Model not configured — restart server with --model"
        )

    # Hot-swap if different model requested
    if _pipe is not None and requested_id != _model_id:
        logger.info(f"Hot-swapping model: {_model_id} → {requested_id}")
        _unload_flux_for_swap()

    # Load if not loaded (initial load or after swap)
    if _pipe is None:
        load_model(requested_id, quant=_quant_mode)
        _enable_offload(_pipe)

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

    print(f'Generating [{req.quality}] {w}x{h} @ {s} steps x{req.count} — "{prompt}"')

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
    return Response(
        content=buf.read(),
        media_type="image/png",
        headers={"X-Saved-Paths": ",".join(paths)},
    )


@app.get("/health")
def health():
    active = "none"
    if _pipe is not None:
        active = "flux"
    elif _vl_model is not None:
        active = "qwen-vl"
    elif _edit_pipe is not None:
        active = "qwen-edit"
    busy = not _generate_lock.acquire(blocking=False)
    if not busy:
        _generate_lock.release()
    return {
        "active": active,
        "flux_loaded": _pipe is not None,
        "vl_loaded": _vl_model is not None,
        "edit_loaded": _edit_pipe is not None,
        "model": _model_id,
        "device": _device or "cuda",
        "status": "busy" if busy else "ready",
        "quant": _quant_mode,
    }


# ── Edit endpoint — Qwen Image Edit 2511 ─────────────────────────────────────
class EditRequest(BaseModel):
    image: str = Field(description="Absolute path to input image")
    prompt: str = Field(description="Edit instruction")
    negative_prompt: str = Field(default=" ", description="Negative prompt")
    steps: int = Field(default=40, ge=1, le=80)
    true_cfg_scale: float = Field(default=4.0, ge=0.5, le=10.0)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=5.0)
    seed: int = Field(default=0)
    count: int = Field(default=1, ge=1, le=4)

    @property
    def safe_image_path(self) -> str:
        allowed = [
            os.path.expanduser("~/.openclaw/media/fantasia"),
            os.path.expanduser("~/.openclaw/media/inbound"),
        ]
        abs_path = os.path.realpath(self.image)
        if not any(abs_path.startswith(d) for d in allowed):
            raise ValueError(f"Image path not in allowed dirs: {abs_path}")
        return abs_path


@app.post("/edit")
async def edit_image(req: EditRequest):
    """
    DISABLED — Qwen Image Edit endpoint is blocked to prevent OOM/system crash.
    The heavy model load caused system instability. Use /generate with sd-turbo instead.
    """
    raise HTTPException(
        status_code=503,
        detail="Image edit endpoint is disabled. Qwen Image Edit model is too resource-intensive for this system.",
    )


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
        help="Quantization mode (default: none)",
    )
    args = parser.parse_args()

    # Store the configured model as default — loaded lazily on first /generate request
    # These are module-level globals, assigned directly (no global declaration needed here)
    import sys as _sys
    _mod = _sys.modules[__name__]
    _mod._model_id = args.model
    _mod._quant_mode = args.quant
    print(f"[fantasia] Lazy mode: model '{args.model}' will load on first /generate request.")
    uvicorn.run(app, host=args.host, port=args.port)
