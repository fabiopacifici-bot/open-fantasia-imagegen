# open-fantasia-imagegen

A local HuggingFace-powered image generator integration for OpenClaw. Supports Flux/Qwen models and others for creative, high-quality images from text prompts.

## Features
- Run Qwen, Flux, or any supported HuggingFace image model locally (no API key required)
- Python-based CLI for direct prompt-to-image workflow
- Skill wrapper for OpenClaw agent integration
- Robust error handling, caching, logging
- Easy setup with requirements.txt, venv support

## Project Structure
```
open-fantasia-imagegen/
├── README.md
├── requirements.txt
├── src/
│   └── imagegen.py
├── skill/
│   └── SKILL.md
├── tests/
│   └── test_imagegen.py
└── examples/
    └── sample_prompt_and_image.png
```

## Implementation Plan

1. **Model & Dependency Setup**
   - Select HuggingFace model (e.g. Qwen, Flux in diffusers)
   - List required pip packages (diffusers, transformers, torch, etc.)
   - Write setup guide in README
2. **Core Generator**
   - Build `imagegen.py` for prompt-to-image inference
   - Support prompt, model selection, output path
   - Add error handling, logging
3. **CLI Interface**
   - Add bash/python CLI wrapper (src/cli.py or imagegen.py)
   - Allow prompt and output options
4. **Skill Integration**
   - Create SKILL.md to document agent integration
   - Provide example OpenClaw skill usage
   - Document how to call locally (no API key needed)
5. **Testing**
   - Basic unit tests in tests/test_imagegen.py
   - Sample prompt in examples/
6. **Documentation**
   - Comprehensive README setup and integration guide
   - Troubleshooting, FAQ

## Next Steps
- Scaffold folders and starter files
- Fill out README and requirements
- Build src/imagegen.py (minimal version)
- Draft SKILL.md for OpenClaw

---

_This repo kickstarts local imagegen for your OpenClaw agent. Update steps as models and workflows evolve!_
