# open-fantasia-imagegen

Local HuggingFace-powered image generation for OpenClaw. Supports FLUX.1 and Stable Diffusion models for high-quality text-to-image generation on your own hardware.

## Quick Start (Slash Command)

Once the server is running, generate images directly from Telegram or webchat:

```
/fantasia Generate an image of the man on the moon
/fantasia a cat civilization in a fantasy city --quality high
/fantasia cyberpunk street at night --quality low
```

Supported quality flags: `--quality low | mid | high` (default: `mid`)

---

## Requirements

### Runtime
- Python 3.10+
- pip / virtualenv
- CUDA GPU recommended (CPU fallback supported, but slow)

### Dependencies
Install via `pip install -r requirements.txt`:

| Package         | Purpose                                       |
|-----------------|-----------------------------------------------|
| `diffusers`     | HuggingFace diffusion pipeline                |
| `torch`         | Deep learning framework                       |
| `transformers`  | Model tokenizers and components               |
| `accelerate`    | GPU memory optimisation / CPU offload         |
| `Pillow`        | Image saving                                  |
| `numpy`         | Array operations                              |
| `python-dotenv` | Loads `.env` credentials                      |
| `fastapi`       | REST API server                               |
| `uvicorn`       | ASGI server for FastAPI                       |

### Environment Variables

| Variable    | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `HF_TOKEN`  | HuggingFace token (required for gated models like FLUX.1)               |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN if needed
```

---

## Persistent Inference Server (Recommended)

Start the server once — model loads into VRAM and stays there. Every subsequent request is near-instant.

```bash
python src/server.py --model stable-diffusion-v1-5/stable-diffusion-v1-5
# or with FLUX (requires HF_TOKEN and 8GB+ VRAM):
python src/server.py --model black-forest-labs/FLUX.1-schnell
```

Server runs on `http://localhost:8765` by default.

### API

**Health check:**
```bash
curl http://localhost:8765/health
# {"model":"...","device":"cuda","status":"ready"}
```

**Generate image:**
```bash
curl -X POST http://localhost:8765/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a dragon over a mountain lake","quality":"mid"}' \
  -o image.png
```

**Request body:**

| Field      | Type    | Default | Description                          |
|------------|---------|---------|--------------------------------------|
| `prompt`   | string  | required| Text prompt                          |
| `quality`  | string  | `mid`   | `low` / `mid` / `high`               |
| `enhance`  | bool    | `true`  | Auto-enhance the prompt              |
| `seed`     | int     | `42`    | Random seed                          |
| `guidance` | float   | `7.5`   | Guidance scale (SD models only)      |
| `width`    | int     | null    | Override width (ignores quality)     |
| `height`   | int     | null    | Override height (ignores quality)    |
| `steps`    | int     | null    | Override inference steps             |

---

## Quality Presets

| Preset | Resolution | Steps | Use case              |
|--------|------------|-------|-----------------------|
| `low`  | 256×256    | 4     | Quick drafts          |
| `mid`  | 512×512    | 20    | Balanced (default)    |
| `high` | 1024×1024  | 50    | Final quality         |

---

## One-shot CLI (no server)

```bash
python src/imagegen.py --prompt "a red dragon flying over a mountain lake" --quality mid
python src/imagegen.py --prompt "cyberpunk city" --quality high --output city.png
python src/imagegen.py --prompt "simple cat sketch" --no-enhance --quality low
```

### CLI Options

| Option        | Default         | Description                        |
|---------------|-----------------|------------------------------------|
| `--prompt`    | required        | Text prompt                        |
| `--output`    | `output.png`    | Output image path                  |
| `--model`     | auto            | HuggingFace model ID               |
| `--quality`   | —               | `low` / `mid` / `high` preset      |
| `--no-enhance`| off             | Skip automatic prompt enhancement  |
| `--height`    | 512             | Image height in pixels             |
| `--width`     | 512             | Image width in pixels              |
| `--steps`     | 20              | Inference steps                    |
| `--guidance`  | 7.5             | Guidance scale (SD only)           |
| `--seed`      | 42              | Random seed                        |

---

## Project Structure

```
open-fantasia-imagegen/
├── README.md
├── requirements.txt
├── src/
│   ├── imagegen.py       # Core generator + one-shot CLI
│   └── server.py         # Persistent FastAPI inference server
├── skill/
│   └── SKILL.md          # OpenClaw skill integration docs
├── tests/
│   └── test_imagegen.py  # Unit tests
└── flux/                 # FLUX reference submodule
```
