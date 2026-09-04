# 🪄 Open Fantasia — Local Image & Video Generation

Generate images locally on your GPU using FLUX.1-schnell or Stable Diffusion, and short
videos with Wan2.1 — no external API, no cost per image or clip.

[![Hippocratic License HL3-LAW-MIL-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-LAW-MIL-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/law-mil-sv.html)


---

## Gallery

| Prompt | Result |
|--------|--------|
| `a cat astronaut on the moon` | ![cat astronaut](assets/demo-cat-astronaut.png) |
| `a samurai cat at sunset` | ![samurai cat](assets/demo-samurai-cat.png) |
| `cyberpunk city at night` | ![cyberpunk city](assets/demo-cyberpunk-city.png) |
| `a dragon breathing fire over a medieval castle` | ![dragon](assets/demo-dragon.png) |

*Generated locally with FLUX.1-schnell, low quality preset. Speed varies by GPU.*

---

## Video Gallery

Short clips generated locally with **Wan2.1** (Wan-AI/Wan2.1-T2V-1.3B, `mid` quality):

| Prompt | Result |
|--------|--------|
| `a horse standing on a rocky beach at sunset` | <video src="assets/videos/demo-horse-beach-sunset.mp4" controls width="320"></video> |
| `a black cat walking along the shoreline in backlight` | <video src="assets/videos/demo-cat-shoreline-backlit.mp4" controls width="320"></video> |
| `a black cat standing in shallow surf, yellow-green eyes` | <video src="assets/videos/demo-cat-shallow-surf.mp4" controls width="320"></video> |
| `a glowing blue whale shark swimming in dark water` | <video src="assets/videos/demo-blue-whale-shark.mp4" controls width="320"></video> |
| `a neon sign glowing pink-purple at night` | <video src="assets/videos/demo-neon-sign.mp4" controls width="320"></video> |
| `a golden light streak flowing on a dark navy background` | <video src="assets/videos/demo-golden-light-streak.mp4" controls width="320"></video> |
| `a flowing gold ribbon on a dark background` | <video src="assets/videos/demo-gold-ribbon.mp4" controls width="320"></video> |
| `a small camera robot sitting among lush green plants` | <video src="assets/videos/demo-camera-robot.mp4" controls width="320"></video> |

*Generated locally with Wan2.1, `mid` quality preset. ~5s clips at 480P.*

---

## Features

- 🚀 **Local generation** — no API keys, no cloud, no per-image cost
- ⚡ **FLUX.1-schnell** — fast distilled model, great quality
- 🌸 **FLUX.2-klein** — lighter experimental model
- 🎨 **Stable Diffusion 1.5** — classic fallback, runs on 6GB VRAM
- ⚡ **SD-Turbo** — near-instant generation (~1-3s at 512×512)
- 🎬 **Wan2.1 text-to-video** — short MP4 clips from text (~8GB VRAM)
- 🔢 **Batch generation** — generate 1–4 images per prompt (different seeds)
- 💬 **OpenClaw slash command** — `/fantasia <prompt>` with inline buttons
- 🔁 **Persistent server** — model stays in VRAM, near-instant after first load

---

## Video Generation (Wan2.1)

Generate a short MP4 clip from a text prompt using `Wan-AI/Wan2.1-T2V-1.3B`
(~8.19 GB VRAM, 480P, ~4 min for a 5s clip on an RTX 4090):

```bash
curl -X POST http://localhost:8765/video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat walking through a neon city at night","quality":"mid"}'
```

Returns raw `video/mp4` bytes. Saved to `~/.openclaw/media/fantasia/videos/<timestamp>.mp4`.

| Field | Default | Notes |
|-------|---------|-------|
| `quality` | `mid` | `low` (480×480, 49f, 20 steps) · `mid` (480×832, 81f, 30 steps) · `high` (480×832, 81f, 50 steps) |
| `steps` | 30 | Inference steps |
| `num_frames` | 81 | 16 fps → 81 ≈ 5s |
| `guidance_scale` | 6.0 | Wan recommends 6.0 |
| `seed` | 42 | Reproducibility |

**Model aliases:** `wan`, `wan13`, `wan2.1`, `wan-1.3b` → `Wan-AI/Wan2.1-T2V-1.3B`

> **VRAM note:** The server auto-unloads the image/Qwen pipelines before loading Wan2.1
to free VRAM, and reloads them on the next image request.

---

## Requirements

- Python 3.10+
- CUDA GPU — **8GB VRAM minimum** (16GB recommended for FLUX)
- [HuggingFace account](https://huggingface.co) + `HF_TOKEN` in `.env` (see `.env.example`)
- ~16GB disk space for FLUX.1-schnell (downloaded once)

---

## Quick Start

```bash
git clone https://github.com/fabiopacifici-bot/open-fantasia-imagegen
cd open-fantasia-imagegen
cp .env.example .env   # add your HF_TOKEN
bash setup.sh
```

`setup.sh` will:
1. Create a Python venv and install dependencies
2. Download the recommended model (~16GB, one-time)
3. Install and enable the systemd service
4. Start the server at `http://localhost:8765`

---

## Usage

### Via OpenClaw slash command (recommended)

```
/fantasia a samurai cat at sunset
/fantasia cyberpunk city --quality high
/fantasia quick sketch --quality low --count 3
/fantasia portrait --model klein
/fantasia setup        ← first-time setup
```

**Inline buttons:** Call `/fantasia` with no prompt to get model/quality/count selection buttons.

### Direct API

```bash
# Generate an image
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat in space", "quality": "low", "enhance": true, "count": 1}'

# Health check
curl http://localhost:8765/health
```

### CLI

```bash
.venv/bin/python src/imagegen.py \
  --prompt "a samurai cat at sunset" \
  --quality low \
  --output output.png
```

---

## Quality Presets

| Preset | Resolution | Steps | Approx. Speed       |
|--------|-----------|-------|---------------------|
| `low`  | 512×512   | 4     | ~1-5 min (GPU dependent)  |
| `mid`  | 768×768   | 8     | ~3-10 min (GPU dependent) |
| `high` | 1024×1024 | 20    | ~10-20 min (GPU dependent)|

> **Note:** Speed varies significantly by GPU. FLUX runs in BF16 precision. Use torchao autoquant to reduce VRAM usage and improve throughput.

---

## Models

| Flag | Model | VRAM | Notes |
|------|-------|------|-------|
| `schnell` (default) | `black-forest-labs/FLUX.1-schnell` | ~16GB | Best quality, BF16 precision |
| `klein` | `black-forest-labs/FLUX.2-klein-base-9B` | ~16GB | Experimental, lighter |
| `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | ~4GB | Fast, classic |
| `turbo` | `stabilityai/sd-turbo` | ~4GB | Near-instant, ~1-3s at 512×512 |

---

## Server Management

```bash
# Start / stop / restart
systemctl --user start fantasia.service
systemctl --user stop fantasia.service
systemctl --user restart fantasia.service

# Status & logs
systemctl --user status fantasia.service
journalctl --user -u fantasia.service -f
```

Output images are saved to `~/.openclaw/media/fantasia/`.

---

## Project Structure

```
open-fantasia-imagegen/
├── src/
│   ├── server.py       # FastAPI inference server
│   └── imagegen.py     # Pipeline loader + generation logic
├── assets/             # README demo images
├── setup.sh            # First-time setup script
├── .env.example        # Environment variable template
├── requirements.txt
└── .specs/             # Plans, docs, debugging notes
```

---

## License

MIT

## Roadmap

### v1.3 — Turbo mode ✅
- `sd-turbo` available as `--model turbo` (stabilityai/sd-turbo)
- ~1-3s generation at 512×512 on CUDA

### v2.0 — Image Editing
- Instruction-based image editing via **Qwen2.5-VL**
- New endpoint: `POST /edit` — `{ "image": "<path>", "instruction": "make the sky purple" }`
- Slash command: `/fantasia edit <image> <instruction>`
