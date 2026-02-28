# open-fantasia-imagegen

Local HuggingFace-powered image generation for OpenClaw. Supports FLUX.1 and Stable Diffusion models for high-quality text-to-image generation on your own hardware.

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
| `python-dotenv` | Loads `.env` credentials                     |

### Environment Variables
Copy `.env.example` to `.env` and fill in your values.

| Variable    | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `HF_TOKEN`  | HuggingFace token (required for gated models like FLUX.1)               |

Get your token at: https://huggingface.co/settings/tokens

**Without `HF_TOKEN`:** falls back to `stabilityai/stable-diffusion-2-1` (open access).  
**With `HF_TOKEN`:** uses `black-forest-labs/FLUX.1-schnell` by default.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
```

## Usage

```bash
# Basic
python src/imagegen.py --prompt "a red dragon flying over a mountain lake"

# Custom model, output, resolution
python src/imagegen.py \
  --prompt "a cyberpunk city at night" \
  --model black-forest-labs/FLUX.1-schnell \
  --output city.png \
  --width 1024 --height 1024 \
  --steps 20

# Skip prompt enhancement
python src/imagegen.py --prompt "simple cat sketch" --no-enhance
```

## CLI Options

| Option        | Default                   | Description                      |
|---------------|---------------------------|----------------------------------|
| `--prompt`    | required                  | Text prompt                      |
| `--output`    | `output.png`              | Output image path                |
| `--model`     | auto (see above)          | HuggingFace model ID             |
| `--no-enhance`| off                       | Skip automatic prompt enhancement|
| `--height`    | 512                       | Image height in pixels           |
| `--width`     | 512                       | Image width in pixels            |
| `--steps`     | 20                        | Inference steps                  |
| `--guidance`  | 7.5                       | Guidance scale (SD only)         |
| `--seed`      | 42                        | Random seed for reproducibility  |

## Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
open-fantasia-imagegen/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   └── imagegen.py       # Core generator + CLI
├── skill/
│   └── SKILL.md          # OpenClaw skill integration docs
├── tests/
│   └── test_imagegen.py  # Unit tests
└── flux/                 # FLUX reference submodule (Black Forest Labs)
```
