---
name: open-fantasia-imagegen
description: Local image generation with FLUX or Stable Diffusion. Generate images from text prompts on your GPU.
metadata:
  version: 1.0.0
---

# Open Fantasia — Image Generation Skill

## Slash Command
`/fantasia <prompt> [--quality low|mid|high]`

## Description
Generate images locally on GPU using the persistent Open Fantasia inference server.
Model stays loaded in VRAM — near-instant generation after first startup.

## When called without prompt
Show this help menu with inline buttons for quality selection.

---

## Usage Examples
```
/fantasia a samurai cat at sunset
/fantasia cyberpunk city at night --quality high
/fantasia quick sketch of a robot --quality low
```

## Quality Presets
| Flag      | Resolution | Steps | Speed     |
|-----------|------------|-------|-----------|
| `--quality low`  | 256×256  | 4   | ~1s       |
| `--quality mid`  | 512×512  | 20  | ~5-10s    |
| `--quality high` | 1024×1024| 50  | ~30-60s   |

Default: `mid`

---

## Agent Instructions

### When `/fantasia` is called WITHOUT a prompt:
Send this formatted message with inline buttons:

> 🪄 **Open Fantasia** — Local Image Generator
> 
> Generate images on your GPU. Near-instant after first load.
> 
> **Usage:** `/fantasia <prompt> [--quality low|mid|high]`
> 
> **Quality presets:**
> - 🔹 Low — 256×256, 4 steps (~1s)
> - 🔷 Mid — 512×512, 20 steps (~5s)
> - 🔶 High — 1024×1024, 50 steps (~30s)
> 
> **Examples:**
> - `/fantasia a calm forest at dawn`
> - `/fantasia cyberpunk skyline --quality high`
> - `/fantasia quick cat sketch --quality low`

Then send inline buttons:
- `Low 🔹` | `Mid 🔷` | `High 🔶` (callback_data: `fantasia_quality:low`, `fantasia_quality:mid`, `fantasia_quality:high`)

### When `/fantasia` is called WITH a prompt:
1. Extract everything after `/fantasia` as the prompt
2. Check for `--quality low|mid|high` flag; strip it from prompt; default `mid`
3. Check server health: `GET http://localhost:8765/health`
   - If down: reply "🚫 Open Fantasia server is offline. Start it with: `cd repositories/open-fantasia-imagegen && .venv/bin/python src/server.py`"
4. POST to `http://localhost:8765/generate`:
   ```json
   {"prompt": "<extracted prompt>", "quality": "<quality>", "enhance": true}
   ```
5. Save response PNG to `repositories/open-fantasia-imagegen/outputs/fantasia_<timestamp>.png`
6. Send via Telegram with caption: `🎨 /fantasia — <short prompt preview>`

---

## Server Setup (run once)
```bash
cd repositories/open-fantasia-imagegen
.venv/bin/python src/server.py --model stable-diffusion-v1-5/stable-diffusion-v1-5
```
Server persists at `localhost:8765`. Restart if machine reboots.

## Output Location
Generated images are stored in:
`repositories/open-fantasia-imagegen/outputs/`

---

## Troubleshooting
- **Connection refused:** Server not running — start it with the command above
- **CUDA out of memory:** Use `--quality low` or reduce steps
- **Slow first run:** Model loading from disk (~30s) — subsequent runs are instant
