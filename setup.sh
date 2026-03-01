#!/usr/bin/env bash
# Open Fantasia — First-time setup script
# Downloads the recommended GGUF model and configures the systemd service.
# Usage: bash setup.sh [--model <gguf-file>] [--force]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
SERVICE_FILE="$HOME/.config/systemd/user/fantasia.service"
MEDIA_DIR="$HOME/.openclaw/media/fantasia"

# Recommended GGUF model — Q4_K_S: ~6.6GB, fast on 8GB+ VRAM
DEFAULT_GGUF_REPO="city96/FLUX.1-schnell-gguf"
DEFAULT_GGUF_FILE="flux1-schnell-Q4_K_S.gguf"
GGUF_MODEL="$DEFAULT_GGUF_REPO/$DEFAULT_GGUF_FILE"

FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) GGUF_MODEL="$2"; shift 2 ;;
    --force) FORCE=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "🪄 Open Fantasia Setup"
echo "========================"

# 1. Check HF_TOKEN
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "⚠️  HF_TOKEN not set. Some models require authentication."
  echo "   Set it with: export HF_TOKEN=hf_..."
  echo "   Continuing without it (public models only)..."
fi

# 2. Create venv and install deps
if [[ ! -d "$VENV" ]] || [[ "$FORCE" == true ]]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv "$VENV"
fi

echo "📦 Installing/updating dependencies..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt"
"$VENV/bin/pip" install -q gguf  # required for GGUF quantization

# 3. Download GGUF model via huggingface_hub
echo "⬇️  Downloading model: $GGUF_MODEL"
echo "   (This is a one-time download, ~6.6GB for Q4_K_S)"

GGUF_PARTS=(${GGUF_MODEL//\// })
HF_REPO="${GGUF_PARTS[0]}/${GGUF_PARTS[1]}"
HF_FILE="${GGUF_PARTS[2]}"

"$VENV/bin/python" - <<EOF
from huggingface_hub import hf_hub_download
import os
token = os.environ.get("HF_TOKEN") or True
print(f"Downloading {HF_FILE} from {HF_REPO}...")
path = hf_hub_download(repo_id="$HF_REPO", filename="$HF_FILE", token=token)
print(f"✅ Saved to: {path}")
EOF

# 4. Create media output dir
mkdir -p "$MEDIA_DIR"
echo "📁 Output directory: $MEDIA_DIR"

# 5. Install systemd service
echo "🔧 Installing systemd service..."
mkdir -p "$(dirname "$SERVICE_FILE")"

cat > "$SERVICE_FILE" <<SVCEOF
[Unit]
Description=Open Fantasia Image Generation Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$VENV/bin/python src/server.py --model $GGUF_MODEL
Restart=on-failure
RestartSec=5
Environment=HF_TOKEN=${HF_TOKEN:-}

[Install]
WantedBy=default.target
SVCEOF

systemctl --user daemon-reload
systemctl --user enable fantasia.service
systemctl --user restart fantasia.service

echo ""
echo "✅ Open Fantasia is ready!"
echo "   Server: http://localhost:8765"
echo "   Health: curl http://localhost:8765/health"
echo "   Images: $MEDIA_DIR"
echo ""
echo "   Try it: /fantasia a cat in space"
