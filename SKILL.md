---
name: open-fantasia-imagegen
description: Agent-safe CLI for local image/video generation through Open Fantasia server.
metadata:
  version: 2.0.0
  commands:
    - /fantasia
---

# Open Fantasia — Agent Skill (CLI-first)

## Purpose
Give every agent one stable command surface for image/video generation:

- `scripts/fantasia image ...`
- `scripts/fantasia video ...`
- `scripts/fantasia health`
- `scripts/fantasia models`
- `scripts/fantasia setup`

This removes fragile ad-hoc curl logic from weaker agents.

## Folder Organization
- `src/` — server + Python CLI implementation (`fantasia_cli.py`)
- `scripts/` — executable wrappers for agents (`fantasia`)
- `tests/` — unit tests for server + CLI payload behavior
- `assets/` — demo/reference media
- `outputs/` — local ad-hoc outputs

## Slash Command
`/fantasia <prompt> [--quality low|mid|high] [--model <alias>] [--count 1-4] [--raw]`

## Agent Operating Rules
1. **No direct curl** unless debugging the CLI itself.
2. For generation requests, call `scripts/fantasia ...`.
3. For long generations, spawn a background subagent.
4. Always check health first for interactive requests:
   - `scripts/fantasia health`
5. Use model aliases via:
   - `scripts/fantasia models`

## Agent Examples
### Image
```bash
scripts/fantasia image --prompt "a robot cat in the colosseum" --quality low --count 2
```

### Video (5s default)
```bash
scripts/fantasia video --prompt "a robot cat walking in the colosseum" --quality mid --seconds 5
```

### Health
```bash
scripts/fantasia health
```

## Setup
```bash
scripts/fantasia setup
```

## Output Location
- Images: `~/.openclaw/media/fantasia/`
- Videos: `~/.openclaw/media/fantasia/videos/`
