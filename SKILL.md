---
name: open-fantasia-imagegen
description: Local image generation with FLUX or Stable Diffusion. Generate images from text prompts on your GPU.
metadata:
  version: 1.2.0
---

# Open Fantasia — Image Generation Skill

## Slash Command
`/fantasia <prompt> [--quality low|mid|high] [--model schnell|klein|sd15] [--count 1-4] [--raw|--enhance]`

## Description
Generate images locally on GPU using the persistent Open Fantasia inference server.
Model stays loaded in VRAM — near-instant generation after first startup.
Default model: **FLUX.1-schnell Q4_K_S GGUF** (~6.6GB, fast on 8GB+ VRAM).

## First-Time Setup
```bash
cd repositories/open-fantasia-imagegen
bash setup.sh
```
This downloads the recommended GGUF model (~6.6GB), installs dependencies, and starts the server as a systemd service.

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
| `--model X`     | Select model: `schnell` (default GGUF), `klein`, `sd15`. |
| `--quality X`   | `low` / `mid` / `high` (see presets below). |

## Quality Presets (GGUF model)
| Flag      | Resolution  | Steps | Speed (RTX 4090 Mobile) |
|-----------|-------------|-------|--------------------------|
| `--quality low`  | 512×512   | 4   | ~5-10s  |
| `--quality mid`  | 768×768   | 8   | ~15-25s |
| `--quality high` | 1024×1024 | 20  | ~40-60s |

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
> - ⚡ FLUX.1-schnell GGUF — fast quantized (default, ~6.6GB)
> - 🌸 FLUX.2-klein — lighter experimental
> - 🎨 SD 1.5 — classic fallback
>
> **Quality presets (GGUF):**
> - 🔹 Low — 512×512, 4 steps (~5-10s)
> - 🔷 Mid — 768×768, 8 steps (~15-25s)
> - 🔶 High — 1024×1024, 20 steps (~40-60s)

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
   - `schnell` → `city96/FLUX.1-schnell-gguf/flux1-schnell-Q4_K_S.gguf`
   - `klein`   → `black-forest-labs/FLUX.2-klein-base-9B`
   - `sd15`    → `stable-diffusion-v1-5/stable-diffusion-v1-5`
5. Check for `--raw`: set `enhance: false`; else `enhance: true` (default)
6. Check server health: `GET http://localhost:8765/health`
   - If down: reply "🚫 Fantasia offline — restarting..." → `systemctl --user restart fantasia.service` → wait 20s → retry
   - If model mismatch: update `~/.config/systemd/user/fantasia.service` ExecStart with correct `--model`, daemon-reload, restart, wait 20s
7. POST to `http://localhost:8765/generate`:
   ```json
   {"prompt": "<prompt>", "quality": "<quality>", "enhance": <bool>, "count": <N>}
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
.venv/bin/python src/server.py --model city96/FLUX.1-schnell-gguf/flux1-schnell-Q4_K_S.gguf
```

## Requirements
- Python 3.10+
- CUDA GPU with 8GB+ VRAM (16GB recommended for mid/high quality)
- HF_TOKEN environment variable (free HuggingFace account)
- ~6.6GB disk space for default GGUF model (one-time download)

## Output Location
`~/.openclaw/media/fantasia/`
On Windows WSL: `\\wsl$\kali-linux\home\<user>\.openclaw\media\fantasia\`

---

## Troubleshooting
- **Slow generation:** Ensure GGUF model is loaded (check `/health` response). BF16 models are slow on 16GB VRAM.
- **OOM error:** Switch to `--quality low` or use `--model sd15`
- **Connection refused:** `systemctl --user restart fantasia.service`; check logs: `journalctl --user -u fantasia.service -n 50`
- **First run slow:** Model loads from disk once (~10-20s), then stays in VRAM
