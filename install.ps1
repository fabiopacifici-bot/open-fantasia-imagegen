# Open Fantasia — Windows (PowerShell) install script.
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 [-Yes] [-NoServer] [-NoSkill] [-NoCli]
#
# Does four things:
#   1. Creates a venv, installs requirements, and (optionally) starts the
#      Fantasia server on :8765.
#   1b. Installs the global `fantasia` command on PATH.
#   2. Discovers OpenClaw and kernel-evolving skill workspaces.
#   3. Installs the skills/fantasia skill into each discovered workspace
#      AFTER explicit user approval (unless -Yes).
#
# Linux/macOS/WSL users: use install.sh instead.

param(
    [switch]$Yes,
    [switch]$NoServer,
    [switch]$NoSkill,
    [switch]$NoCli
)

$ErrorActionPreference = "Stop"
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $RepoDir ".venv"
$MediaDir = Join-Path $HOME ".openclaw\media\fantasia"
$SkillSrc = Join-Path $RepoDir "skills\fantasia"

function Confirm-Step([string]$Prompt) {
    if ($Yes) { return $true }
    $ans = Read-Host "$Prompt [y/N]"
    return ($ans -eq "y" -or $ans -eq "Y")
}

Write-Host "🪄 Open Fantasia Installer"
Write-Host "=========================="

# ── 1. Dependencies + server ────────────────────────────────────────────────
if (-not $NoServer) {
    Write-Host "📦 Setting up virtual environment..."
    if (-not (Test-Path $Venv)) {
        python -m venv $Venv
    }
    $Pip = Join-Path $Venv "Scripts\pip.exe"
    & $Pip install -q --upgrade pip
    & $Pip install -q -r (Join-Path $RepoDir "requirements.txt")
    New-Item -ItemType Directory -Force -Path $MediaDir | Out-Null

    # Start server in background (no systemd on Windows).
    $Python = Join-Path $Venv "Scripts\python.exe"
    $ServerPy = Join-Path $RepoDir "server\server.py"
    $LogFile = Join-Path $env:TEMP "fantasia_server.log"
    Write-Host "🚀 Starting Fantasia server in background..."
    Start-Process -FilePath $Python -ArgumentList $ServerPy, "--model", "black-forest-labs/FLUX.1-schnell" `
        -RedirectStandardOutput $LogFile -RedirectStandardError $LogFile -WindowStyle Hidden
    Write-Host "   Log: $LogFile"
    Write-Host "   Health: curl http://localhost:8765/health"
}

# ── 1b. Global CLI on PATH — run `fantasia image ..` from anywhere ─────────
if (-not $NoCli) {
    Write-Host "🔗 Installing global 'fantasia' command..."
    $CliPy = Join-Path $RepoDir "fantasia.py"
    $BinDir = Join-Path $HOME ".local\bin"
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $CliBat = Join-Path $BinDir "fantasia.cmd"
    # .cmd shim so `fantasia` resolves on PATH like any terminal command
    @("@echo off", "\"$CliPy\" %*") | Set-Content -Path $CliBat -Encoding Ascii
    Write-Host "   ✅ Installed -> $CliBat"
    Write-Host "   ⚠️  Add $BinDir to your PATH if it isn't already."
    Write-Host "   Try: fantasia health"
}

# ── 2. Discover agent workspaces ────────────────────────────────────────────
if (-not $NoSkill) {
    $Targets = New-Object System.Collections.Generic.List[string]

    # OpenClaw workspaces: ~/.openclaw/workspace*/skills/
    Get-ChildItem -Path $HOME -Directory -Filter ".openclaw*" -ErrorAction SilentlyContinue | ForEach-Object {
        $skills = Join-Path $_.FullName "workspace\skills"
        if (Test-Path $skills) { $Targets.Add($skills) }
    }
    # kernel-evolving private ecosystem
    $ke = Join-Path $HOME ".kernel\ecosystem\private\skills"
    if (Test-Path $ke) { $Targets.Add($ke) }
    # kernel-evolving workspace skills
    $kew = Join-Path $HOME ".kernel-evolving\workspace\skills"
    if (Test-Path $kew) { $Targets.Add($kew) }

    $Uniq = $Targets | Select-Object -Unique

    if ($Uniq.Count -eq 0) {
        Write-Host "ℹ️  No OpenClaw/kernel-evolving skill workspaces discovered. Skipping skill install."
    } else {
        Write-Host ""
        Write-Host "🔎 Discovered skill workspaces:"
        $i = 1
        foreach ($t in $Uniq) {
            Write-Host "   [$i] $t"
            $i++
        }
        if (Confirm-Step "Install the 'fantasia' skill into all of the above? ") {
            foreach ($ws in $Uniq) {
                $dest = Join-Path $ws "fantasia"
                New-Item -ItemType Directory -Force -Path $dest | Out-Null
                Copy-Item -Path (Join-Path $SkillSrc "*") -Destination $dest -Recurse -Force
                Write-Host "   ✅ Installed -> $dest"
            }
            Write-Host "🎉 Skill installed. Agents can now use /fantasia."
        } else {
            Write-Host "⏭️  Skipped skill install (no approval)."
        }
    }
}

Write-Host ""
Write-Host "✅ Open Fantasia install complete."
Write-Host "   CLI:  $RepoDir\fantasia.py"
Write-Host "   Try:  fantasia health" (or python fantasia.py health)
