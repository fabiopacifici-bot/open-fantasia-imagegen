# Plan: ClawHub Release — GGUF support + setup command

## Goal
Publish open-fantasia-imagegen to ClawHub with:
- Fast generation via quantized (GGUF/FP8) FLUX model
- `/fantasia setup` command for first-time installation
- Clean skill UX (model buttons, count, quality)

## Acceptance Criteria
- Server loads GGUF model via llama.cpp / diffusers GGUF backend
- Generation at low quality completes in <30s on RTX 4090 Mobile
- `/fantasia setup` downloads recommended model + installs deps
- Works on a fresh OpenClaw install with just HF_TOKEN set

## Steps

### Phase 1 — GGUF/quantized model support
- [ ] Research best diffusers GGUF loader for FLUX.1-schnell
- [ ] Update `imagegen.py` to detect GGUF model path and use correct loader
- [ ] Update systemd service to point to GGUF model
- [ ] Verify generation speed improvement
- [ ] Update QUALITY_PRESETS (low steps back to 4 if GGUF is faster)

### Phase 2 — Setup command
- [ ] Add `setup.sh` script: checks HF_TOKEN, downloads model, installs venv, enables systemd service
- [ ] Update SKILL.md with `/fantasia setup` instructions
- [ ] Test on clean environment

### Phase 3 — ClawHub publish
- [ ] Read clawhub skill, prepare skill package
- [ ] Bump version, update README
- [ ] Publish via clawhub CLI
