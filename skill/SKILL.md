# open-fantasia-imagegen Skill Integration

## Description
A local HuggingFace-based image generation skill for OpenClaw. Enables models like Qwen or Flux to create images from prompts on your own hardware.

## Usage
- Install requirements: `pip install -r requirements.txt`
- Run: `python src/imagegen.py "prompt" out.png`
- Integrate via OpenClaw skill system (see README)

## API & CLI Options
- Prompt: input text
- Output: image file
- Model selection: stabilityai/stable-diffusion-2 default; can extend to Qwen, Flux, etc.

## Extension
- Add prompt engineering, batch generation, custom output formats
- Connect with agent tools for automation and advanced workflows

---
_This SKILL.md is a starter for agentic image gen integration._
