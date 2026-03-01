#!/usr/bin/env bash
# Open Fantasia — First-time setup script
# Installs dependencies and configures the systemd service.
# The model (black-forest-labs/FLUX.1-schnell) is downloaded automatically by diffusers on first run.
# Usage: bash setup.sh [--force]

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
SERVICE_FILE="$HOME/.config/systemd/user/fantasia.service"
MEDIA_DIR="$HOME/.openclaw/media/fantasia"

FORCE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
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

# 3. Create media output dir
mkdir -p "$MEDIA_DIR"
echo "📁 Output directory: $MEDIA_DIR"

# 4. Install systemd service
echo "🔧 Installing systemd service..."
mkdir -p "$(dirname "$SERVICE_FILE")"

cat > "$SERVICE_FILE" <<SVCEOF
[Unit]
Description=Open Fantasia Image Generation Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$VENV/bin/python src/server.py --model black-forest-labs/FLUX.1-schnell
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
echo "   The model (black-forest-labs/FLUX.1-schnell) will be downloaded automatically"
echo "   by diffusers on first run (~34GB BF16). Use --model flag to specify an alternative."
echo ""
echo "   Try it: /fantasia a cat in space"
