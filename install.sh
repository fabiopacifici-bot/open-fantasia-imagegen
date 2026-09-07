#!/usr/bin/env bash
# Open Fantasia — cross-platform install script (Linux/macOS/WSL).
#
#   bash install.sh [--yes] [--no-server] [--no-skill]
#
# Does three things:
#   1. Reuses setup.sh to create the venv, install dependencies, and start the
#      Fantasia server on :8765.
#   2. Discovers OpenClaw and kernel-evolving skill workspaces.
#   3. Installs the skills/fantasia skill into each discovered workspace
#      AFTER explicit user approval (unless --yes).
#
# Windows (PowerShell) users: use install.ps1 instead.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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

# ── 1. Dependencies + server — delegate to setup.sh (single source of truth) ──
if [[ "$START_SERVER" == true ]]; then
  echo "📦 Running setup.sh (venv, deps, server)..."
  bash "$REPO_DIR/setup.sh"
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
