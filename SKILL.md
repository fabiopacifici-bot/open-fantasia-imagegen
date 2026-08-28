---
name: open-fantasia-imagegen
description: Local image AND video generation with FLUX, Stable Diffusion, and Wan2.1. Generate images or short videos from text prompts on your GPU.
metadata:
  version: 1.4.0
  commands:
    - /fantasia
---

# Open Fantasia — Image & Video Generation Skill

## Slash Command
`/fantasia <prompt> [--quality low|mid|high] [--model schnell|klein|sd15|turbo] [--count 1-4] [--raw|--enhance]`

## Description
Generate images and short videos locally on GPU using the persistent Open Fantasia inference server.
Models stay loaded in VRAM — near-instant generation after first startup.
- Default image model: **FLUX.1-schnell BF16** (~16GB VRAM, ~33GB disk).
- Default video model: **Wan-AI/Wan2.1-T2V-1.3B** (~8.19GB VRAM, 480P, ~4min/5s clip on an RTX 4090).

## First-Time Setup
```bash
cd repositories/open-fantasia-imagegen
cp .env.example .env   # add your HF_TOKEN
bash setup.sh
```
This installs dependencies, downloads the recommended model, and starts the server as a systemd service.

---

## Video Generation (`POST /video`)

Generate a short MP4 clip from a text prompt using Wan2.1:

```json
POST http://localhost:8765/video
{
  "prompt": "a cat walking through a neon city at night",
  "quality": "mid",        // low | mid | high
  "steps": 30,
  "num_frames": 81,        // 16fps; 81 ≈ 5s
  "guidance_scale": 6.0,
  "seed": 42,
  "enhance": false,
  "model": null            // null = Wan-AI/Wan2.1-T2V-1.3B
}
```

**Response:** raw `video/mp4` bytes with `X-Saved-Paths` header.
Saved to `~/.openclaw/media/fantasia/videos/<timestamp>.mp4`.

| Field | Default | Notes |
|-------|---------|-------|
| `quality` | `mid` | `low` (480×480, 49f, 20 steps) · `mid` (480×832, 81f, 30 steps) · `high` (480×832, 81f, 50 steps) |
| `steps` | 30 | Inference steps (max 20 via env cap) |
| `num_frames` | 81 | 16 fps → 81 ≈ 5s |
| `guidance_scale` | 6.0 | Wan recommends 6.0 |

**Model aliases:** `wan`, `wan13`, `wan2.1`, `wan-1.3b` → `Wan-AI/Wan2.1-T2V-1.3B`

**VRAM note:** Video generation is heavy. The server auto-unloads the image/Qwen pipelines before loading Wan2.1 to free VRAM, and re-loads them on the next image request.

---

## Usage Examples
```
/fantasia a samurai cat at sunset
/fantasia cyberpunk city at night --quality high
/fantasia quick sketch of a robot --quality low
/fantasia portrait of a woman --raw
/fantasia 3 variations of a forest --count 3
/fantasia a dragon --model klein
/fantasia fast test --model turbo
```

## Flags
| Flag            | Behaviour |
|-----------------|-----------|
| `--raw`         | Prompt sent directly to the model — no enhancement. |
| `--enhance`     | (default) AI polishes your prompt before generation. |
| `--count N`     | Generate N images (1–4). Each uses a different seed. |
| `--model X`     | Select model: `schnell` (default), `klein`, `sd15`, `turbo`. |
| `--quality X`   | `low` / `mid` / `high` (see presets below). |

## Models
| Flag | Model | VRAM | Notes |
|------|-------|------|-------|
| `schnell` (default) | `black-forest-labs/FLUX.1-schnell` | ~16GB | Best quality, BF16 precision |
| `klein` | `black-forest-labs/FLUX.2-klein-base-9B` | ~16GB | Experimental, lighter |
| `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | ~4GB | Fast, classic |
| `turbo` | `stabilityai/sd-turbo` | ~4GB | Near-instant, ~1-3s at 512×512 |

## Quality Presets
| Flag      | Resolution  | Steps | Approx. Speed       |
|-----------|-------------|-------|---------------------|
| `--quality low`  | 512×512   | 4   | ~1-5 min (GPU dependent)  |
| `--quality mid`  | 768×768   | 8   | ~3-10 min (GPU dependent) |
| `--quality high` | 1024×1024 | 20  | ~10-20 min (GPU dependent)|

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
> **Usage:** `/fantasia <prompt> [--quality low|mid|high] [--model schnell|klein|sd15|turbo] [--count 1-4] [--raw|--enhance]`
>
> **Available models:**
> - ⚡ FLUX.1-schnell BF16 — best quality (default, ~16GB VRAM)
> - 🌸 FLUX.2-klein — lighter experimental
> - 🎨 SD 1.5 — classic fallback (~4GB VRAM)
> - ⚡ SD-Turbo — near-instant (~1-3s, ~4GB VRAM)

Buttons (3 rows):

Row 1 — Model:
- `⚡ FLUX schnell` → callback_data: `fantasia_model_schnell`
- `🌸 FLUX klein` → callback_data: `fantasia_model_klein`
- `🎨 SD 1.5` → callback_data: `fantasia_model_sd15`
- `⚡ Turbo` → callback_data: `fantasia_model_turbo`

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
4. Check for `--model schnell|klein|sd15|turbo`; strip it; default `schnell`
   - `schnell` → `black-forest-labs/FLUX.1-schnell`
   - `klein`   → `black-forest-labs/FLUX.2-klein-base-9B`
   - `sd15`    → `stable-diffusion-v1-5/stable-diffusion-v1-5`
   - `turbo`   → `stabilityai/sd-turbo`
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
cp .env.example .env   # add your HF_TOKEN
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
- CUDA GPU with 8GB+ VRAM (16GB recommended for FLUX mid/high quality)
- `HF_TOKEN` in `.env` (free HuggingFace account — see `.env.example`)
- ~33GB disk space for default BF16 model (one-time download); ~4GB for sd15/turbo

## Output Location
`~/.openclaw/media/fantasia/`
On Windows WSL: `\\wsl$\kali-linux\home\<user>\.openclaw\media\fantasia\`

---

## Troubleshooting
- **Slow generation:** Use `--model turbo` or `--quality low` for faster results
- **OOM error:** Switch to `--model sd15` or `--model turbo` (~4GB VRAM)
- **Connection refused:** `systemctl --user restart fantasia.service`; check logs: `journalctl --user -u fantasia.service -n 50`
- **First run slow:** Model loads from disk once (~10-20s), then stays in VRAM
