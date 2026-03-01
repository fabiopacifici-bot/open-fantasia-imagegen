---
name: open-fantasia-imagegen
description: Local image generation with FLUX or Stable Diffusion. Generate images from text prompts on your GPU.
metadata:
  version: 1.2.0
---

# Open Fantasia — Image Generation Skill

## Slash Command
`/fantasia <prompt> [--quality low|mid|high] [--model schnell|klein|sd15|turbo|turbo-xl] [--count 1-4] [--raw|--enhance]`

## Description
Generate images locally on GPU using the persistent Open Fantasia inference server.
Model stays loaded in VRAM — near-instant generation after first startup.
Default model: **FLUX.1-schnell BF16** (~34GB, or use torchao autoquant to reduce VRAM usage).

## First-Time Setup
```bash
cd repositories/open-fantasia-imagegen
bash setup.sh
```
This installs dependencies, installs and starts the server as a systemd service.
The model (`black-forest-labs/FLUX.1-schnell`) is downloaded automatically by diffusers on first run (~34GB BF16, or use `--model` flag for alternatives).

---

## Usage Examples
```
/fantasia a samurai cat at sunset
/fantasia cyberpunk city at night --quality high
/fantasia quick sketch of a robot --quality low
/fantasia portrait of a woman --raw
/fantasia 3 variations of a forest --count 3
/fantasia a dragon --model klein
```

## Flags
| Flag            | Behaviour |
|-----------------|-----------|
| `--raw`         | Prompt sent directly to the model — no enhancement. |
| `--enhance`     | (default) AI polishes your prompt before generation. |
| `--count N`     | Generate N images (1–4). Each uses a different seed. |
| `--model X`     | Select model: `schnell` (default BF16), `klein`, `sd15`, `z-img`, `turbo`, `turbo-xl`. |
| `--quality X`   | `low` / `mid` / `high` (see presets below). |
| `--quant X`     | Quantization: `autoquant` (default), `int4`, or `none`. |

## Quality Presets
| Flag      | Resolution  | Steps | Speed (RTX 4090) |
|-----------|-------------|-------|-------------------|
| `--quality low`  | 512×512   | 4   | ~2-3 min          |
| `--quality mid`  | 768×768   | 8   | ~5-8 min          |
| `--quality high` | 1024×1024 | 20  | ~13+ min          |

Default: `low`

---

## Agent Instructions

### When `/fantasia setup` is called:
Run `bash setup.sh` in the repo directory and report the result. If the server starts successfully, confirm with a health check.

### When `/fantasia` is called WITHOUT a prompt:
Send this formatted message with inline buttons:

> 🪄 **Open Fantasia** — Local Image Generator
>
> Generate images on your GPU. Near-instant after first load.
>
> **Usage:** `/fantasia <prompt> [--quality low|mid|high] [--model schnell|klein|sd15] [--count 1-4] [--raw|--enhance]`
>
> **Available models:**
> - ⚡ FLUX.1-schnell BF16 — best quality (default, ~34GB / autoquant available)
> - 🌸 FLUX.2-klein — lighter experimental
> - 🎨 SD 1.5 — classic fallback
> - 🔷 Z-Img — 6B single-stream diffusion (Zhibei-ai/Z-Img)
> - ⚡ SD-Turbo — 1-3s, 2GB VRAM, ideal for quick drafts
> - ⚡ SDXL-Turbo — 2-5s, 6GB VRAM, better quality
>
> **Quality presets:**
> - 🔹 Low — 512×512, 4 steps (~2-3 min RTX 4090)
> - 🔷 Mid — 768×768, 8 steps (~5-8 min)
> - 🔶 High — 1024×1024, 20 steps (~13+ min)

Buttons (3 rows):

Row 1 — Model:
- `⚡ FLUX schnell` → callback_data: `fantasia_model_schnell`
- `🌸 FLUX klein` → callback_data: `fantasia_model_klein`
- `🎨 SD 1.5` → callback_data: `fantasia_model_sd15`

Row 2 — Quality:
- `🔹 Low` → callback_data: `fantasia_quality_low`
- `🔷 Mid` → callback_data: `fantasia_quality_mid`
- `🔶 High` → callback_data: `fantasia_quality_high`

Row 3 — Count:
- `1️⃣` → callback_data: `fantasia_count_1`
- `2️⃣` → callback_data: `fantasia_count_2`
- `3️⃣` → callback_data: `fantasia_count_3`
- `4️⃣` → callback_data: `fantasia_count_4`

### ⚠️ ALWAYS spawn generation as a background subagent
Never poll generation synchronously in the main session — it blocks the chat.
Use `sessions_spawn` with `runtime="subagent"`, `mode="run"` for all generation requests.
The subagent handles the curl, waits for the image, sends it via Telegram, then exits.
Main session stays responsive throughout.

### When `/fantasia` is called WITH a prompt:
1. Extract everything after `/fantasia` as the prompt
2. Check for `--quality low|mid|high`; strip it; default `low`
3. Check for `--count N` (1-4); strip it; default `1`
4. Check for `--model schnell|klein|sd15`; strip it; default `schnell`
   - `schnell` → `black-forest-labs/FLUX.1-schnell`
   - `klein`   → `black-forest-labs/FLUX.2-klein-base-9B`
   - `sd15`    → `stable-diffusion-v1-5/stable-diffusion-v1-5`
   - `z-img`   → `Zhibei-ai/Z-Img`
   - `turbo`   → `stabilityai/sd-turbo`
   - `turbo-xl`→ `stabilityai/sdxl-turbo`
5. Check for `--quant none|autoquant|int4`; strip it; default `autoquant`
6. Check for `--raw`: set `enhance: false`; else `enhance: true` (default)
6. Check server health: `GET http://localhost:8765/health`
   - If down: reply "🚫 Fantasia offline — restarting..." → `systemctl --user restart fantasia.service` → wait 20s → retry
   - If model mismatch: update `~/.config/systemd/user/fantasia.service` ExecStart with correct `--model`, daemon-reload, restart, wait 20s
7. POST to `http://localhost:8765/generate`:
   ```json
   {"prompt": "<prompt>", "quality": "<quality>", "enhance": <bool>, "count": <N>, "quant": "<quant>"}
   ```
   Server saves all images to `~/.openclaw/media/fantasia/<timestamp>_<i>.png`
8. Send each image via Telegram with caption: `🎨 /fantasia — <short prompt preview> [<i>/<N>]`

---

## Server Setup

### First time (recommended)
```bash
cd repositories/open-fantasia-imagegen
bash setup.sh
```

### Manual
```bash
systemctl --user enable fantasia.service
systemctl --user start fantasia.service
systemctl --user status fantasia.service
```

Service file: `~/.config/systemd/user/fantasia.service`

### Manual fallback (no systemd)
```bash
cd repositories/open-fantasia-imagegen
.venv/bin/python src/server.py --model black-forest-labs/FLUX.1-schnell
```

## Requirements
- Python 3.10+
- CUDA GPU with 16GB+ VRAM (recommended for FLUX BF16; use torchao autoquant for lower VRAM)
- HF_TOKEN environment variable (free HuggingFace account)
- ~34GB disk space for FLUX.1-schnell BF16 (one-time download on first run)

## Output Location
`~/.openclaw/media/fantasia/`
On Windows WSL: `\\wsl$\kali-linux\home\<user>\.openclaw\media\fantasia\`

---

## Troubleshooting
- **Slow generation:** BF16 FLUX is compute-heavy. Use torchao autoquant or `--quality low` for faster results.
- **OOM error:** Switch to `--quality low` or use `--model sd15`
- **Connection refused:** `systemctl --user restart fantasia.service`; check logs: `journalctl --user -u fantasia.service -n 50`
- **First run slow:** Model downloads from HuggingFace on first run (~34GB), then loads from disk each restart (~10-20s)

---

## Quantization Modes

| Mode         | Description | VRAM savings | Speed |
|--------------|-------------|--------------|-------|
| `autoquant`  | torchao auto-selects best int8 kernel (default) | ~40% | fastest |
| `int4`       | torchao int4 weight-only, maximum VRAM savings | ~60% | fast, minor quality loss |
| `none`       | Pure BF16, no quantization | 0% | baseline |

### CLI usage
```bash
python src/imagegen.py --prompt "a sunset" --quant int4
python src/server.py --model black-forest-labs/FLUX.1-schnell --quant int4
```

### Server API
Pass `"quant"` in the generate request (note: quant is set at server startup; request field logs a warning if it differs from loaded mode):
```json
{"prompt": "a sunset", "quant": "int4"}
```

---

## Turbo Models (SD-Turbo / SDXL-Turbo)

⚡ Turbo models are **CFG-free** (`guidance_scale=0.0`) and use a maximum of 4 steps — ideal for quick drafts.

| Alias      | Full Model ID              | VRAM  | Speed      |
|------------|---------------------------|-------|------------|
| `turbo`    | stabilityai/sd-turbo       | ~2GB  | ⚡ 1-3s    |
| `turbo-xl` | stabilityai/sdxl-turbo     | ~6GB  | ⚡ 2-5s    |

### Turbo Quality Presets
| Preset | Resolution | Steps |
|--------|------------|-------|
| `low`  | 512×512    | 1     |
| `mid`  | 512×512    | 2     |
| `high` | 768×768    | 4     |

> **Note:** Turbo models use `guidance_scale=0.0` automatically — no CFG needed. Max 4 steps.

### Usage
```bash
/fantasia a quick sketch --model turbo --quality mid
/fantasia a portrait --model turbo-xl --quality high
```

---

## Z-Img Model

**Z-Img** is a 6B single-stream diffusion transformer. HuggingFace ID: `Zhibei-ai/Z-Img`

Loaded via `FluxPipeline` (same single-stream transformer architecture). Supports the same quality presets and quantization options as FLUX.

### Usage
```bash
/fantasia a dragon --model z-img
python src/imagegen.py --prompt "a dragon" --model Zhibei-ai/Z-Img
```

The server/CLI auto-detects `z-img` or `zhibei` in the model ID (case-insensitive).

---

## /edit Endpoint — Visual Understanding (v2 stub)

> ⚠️ **v2 stub**: This uses Qwen2.5-VL-7B-Instruct for **visual understanding**, not pixel-level editing.
> True diffusion inpainting (pixel editing) is planned for **v2.1**.

The model is loaded **lazily** — only when the first `/edit` request arrives. Loaded in 4-bit via bitsandbytes to minimize VRAM.

### POST /edit
```json
{
  "image": "/path/to/image.png",
  "instruction": "make the sky purple",
  "max_new_tokens": 512
}
```
Returns:
```json
{
  "response": "The sky in the image could be changed to a vibrant purple by...",
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "note": "v2 stub: This is visual understanding (VL), not pixel-level editing. True diffusion inpainting is planned for v2.1."
}
```

### GET /edit/status
```json
{"loaded": false, "model": "Qwen/Qwen2.5-VL-7B-Instruct"}
```

### Notes
- Requires `bitsandbytes` installed (already in requirements.txt)
- If FLUX is loaded simultaneously, warns about potential OOM on GPUs < 24GB
- Images must be local file paths (not URLs)
