#!/usr/bin/env bash
# Open Fantasia — cross-platform install script (Linux/macOS/WSL).
#
#   bash install.sh [--yes] [--no-server] [--no-skill]
#
# Does three things:
#   1. Creates a venv, installs requirements, and (optionally) starts the
#      Fantasia server on :8765.
#   2. Discovers OpenClaw and kernel-evolving skill workspaces.
#   3. Installs the skills/fantasia skill into each discovered workspace
#      AFTER explicit user approval (unless --yes).
#
# Windows (PowerShell) users: use install.ps1 instead.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
MEDIA_DIR="$HOME/.openclaw/media/fantasia"
SKILL_SRC="$REPO_DIR/skills/fantasia"

ASSUME_YES=false
START_SERVER=true
INSTALL_SKILL=true
for arg in "$@"; do
  case "$arg" in
    --yes) ASSUME_YES=true ;;
    --no-server) START_SERVER=false ;;
    --no-skill) INSTALL_SKILL=false ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

confirm() {
  # confirm <prompt> -> true/false. Skips prompt with --yes.
  if [[ "$ASSUME_YES" == true ]]; then
    return 0
  fi
  local prompt="$1"
  read -r -p "$prompt [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]]
}

echo "🪄 Open Fantasia Installer"
echo "=========================="

# ── 1. Dependencies + server ────────────────────────────────────────────────
if [[ "$START_SERVER" == true ]]; then
  echo "📦 Setting up virtual environment..."
  if [[ ! -d "$VENV" ]]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$REPO_DIR/requirements.txt"
  mkdir -p "$MEDIA_DIR"

  if command -v systemctl >/dev/null 2>&1 && systemctl --user >/dev/null 2>&1; then
    echo "🔧 Installing systemd user service..."
    SERVICE_FILE="$HOME/.config/systemd/user/fantasia.service"
    mkdir -p "$(dirname "$SERVICE_FILE")"
    cat > "$SERVICE_FILE" <<SVCEOF
[Unit]
Description=Open Fantasia Image Generation Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO_DIR
ExecStart=$VENV/bin/python server/server.py --model black-forest-labs/FLUX.1-schnell
Restart=on-failure
RestartSec=5
Environment=HF_TOKEN=\${HF_TOKEN:-}

[Install]
WantedBy=default.target
SVCEOF
    systemctl --user daemon-reload
    systemctl --user enable fantasia.service
    systemctl --user restart fantasia.service
  else
    echo "⚠️  systemd not available — starting server in background instead."
    nohup "$VENV/bin/python" server/server.py --model black-forest-labs/FLUX.1-schnell \
      > /tmp/fantasia_server.log 2>&1 &
    echo "   Server PID $! — log: /tmp/fantasia_server.log"
  fi

  echo "✅ Server installed. Health: curl http://localhost:8765/health"
fi

# ── 2. Discover agent workspaces ────────────────────────────────────────────
if [[ "$INSTALL_SKILL" == true ]]; then
  declare -a TARGETS=()

  # OpenClaw workspaces: ~/.openclaw/workspace*/skills/
  for ws in "$HOME"/.openclaw/workspace*/skills; do
    [[ -d "$ws" ]] && TARGETS+=("$ws")
  done
  # kernel-evolving private ecosystem
  if [[ -d "$HOME/.kernel/ecosystem/private/skills" ]]; then
    TARGETS+=("$HOME/.kernel/ecosystem/private/skills")
  fi
  # kernel-evolving workspace skills
  if [[ -d "$HOME/.kernel-evolving/workspace/skills" ]]; then
    TARGETS+=("$HOME/.kernel-evolving/workspace/skills")
  fi

  # De-dupe while preserving order
  declare -a UNIQ=()
  for t in "${TARGETS[@]}"; do
    found=false
    for u in "${UNIQ[@]}"; do [[ "$u" == "$t" ]] && found=true; done
    [[ "$found" == false ]] && UNIQ+=("$t")
  done

  if [[ ${#UNIQ[@]} -eq 0 ]]; then
    echo "ℹ️  No OpenClaw/kernel-evolving skill workspaces discovered. Skipping skill install."
  else
    echo ""
    echo "🔎 Discovered skill workspaces:"
    for i in "${!UNIQ[@]}"; do
      echo "   [$((i+1))] ${UNIQ[$i]}"
    done
    if confirm "Install the 'fantasia' skill into all of the above? "; then
      for ws in "${UNIQ[@]}"; do
        mkdir -p "$ws/fantasia"
        cp -R "$SKILL_SRC/." "$ws/fantasia/"
        echo "   ✅ Installed -> $ws/fantasia"
      done
      echo "🎉 Skill installed. Agents can now use /fantasia."
    else
      echo "⏭️  Skipped skill install (no approval)."
    fi
  fi
fi

echo ""
echo "✅ Open Fantasia install complete."
echo "   CLI:  $REPO_DIR/fantasia.py"
echo "   Try:  python3 fantasia.py health"
