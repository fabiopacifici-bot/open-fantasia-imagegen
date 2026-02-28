# Open Fantasia — Image Generation Skill

## Trigger
Slash command: `/fantasia <prompt> [--quality low|mid|high]`

## Description
Generate images locally on GPU using the persistent Open Fantasia inference server.
Model stays loaded in VRAM — near-instant generation after first startup.

## Usage Examples
```
/fantasia Generate an image of the man on the moon
/fantasia a cat civilization in a fantasy medieval city --quality high
/fantasia quick sketch of a robot --quality low
/fantasia photorealistic sunset over the ocean --quality mid
```

## How It Works
1. Parse the prompt and optional `--quality` flag from the message
2. Check if server is alive at `http://localhost:8765/health`
3. POST to `http://localhost:8765/generate` with prompt + quality
4. Receive PNG bytes → save to workspace → send via Telegram

## Quality Presets
| Flag      | Resolution | Steps | Speed     |
|-----------|------------|-------|-----------|
| `--quality low`  | 256×256  | 4   | ~1s       |
| `--quality mid`  | 512×512  | 20  | ~5-10s    |
| `--quality high` | 1024×1024| 50  | ~30-60s   |

Default: `mid`

## Server Setup (run once)
```bash
cd repositories/open-fantasia-imagegen
.venv/bin/python src/server.py --model stable-diffusion-v1-5/stable-diffusion-v1-5
```
Server persists at `localhost:8765`. Restart if machine reboots.

## Agent Instructions
When `/fantasia` is invoked:
1. Extract everything after `/fantasia` as the prompt
2. Check for `--quality low|mid|high` flag; strip it from prompt; default `mid`
3. POST to `http://localhost:8765/generate`:
   ```json
   {"prompt": "<extracted prompt>", "quality": "<quality>", "enhance": true}
   ```
4. Save response PNG to `/home/pacificDev/.openclaw/workspace/fantasia_out.png`
5. Send via Telegram to user with a short caption
6. If server is down (connection refused), reply: "Open Fantasia server is offline. Start it with: `.venv/bin/python src/server.py` in the open-fantasia-imagegen repo."
