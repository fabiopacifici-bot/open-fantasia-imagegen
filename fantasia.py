#!/usr/bin/env python3
"""Open Fantasia agent-friendly CLI wrapper.

CLI goals:
- Stable subcommands for agents (`image`, `video`, `health`, `models`, `setup`)
- Thin HTTP wrapper around the local server on :8765
- Machine-parseable JSON output for orchestration
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = os.environ.get("FANTASIA_BASE_URL", "http://127.0.0.1:8765")

# CLI lives at repo root (next to server/, skills/, setup.sh).
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.expanduser("~/.openclaw/media/fantasia")


def _http_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, method=method, data=data, headers=headers)

    try:
        with request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
            saved_paths = resp.headers.get("X-Saved-Paths", "")
            if "application/json" in content_type:
                out = json.loads(body)
            else:
                out = {"raw": body}
            if saved_paths:
                out["saved_paths"] = [p for p in saved_paths.split(",") if p]
            out["status_code"] = resp.status
            return out
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": raw}
        parsed["status_code"] = e.code
        raise RuntimeError(json.dumps(parsed, ensure_ascii=False)) from e
    except error.URLError as e:
        raise RuntimeError(f"Server unreachable at {url}: {e}") from e


def seconds_to_frames(seconds: int) -> int:
    """Convert seconds to frames at 16fps (server convention), min 1."""
    return max(1, seconds * 16)


def build_image_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "quality": args.quality,
        "count": args.count,
        "seed": args.seed,
        "enhance": not args.raw,
        "model": args.model,
    }


def build_video_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "prompt": args.prompt,
        "quality": args.quality,
        "seed": args.seed,
        "enhance": not args.raw,
        "model": args.model,
    }
    if args.seconds is not None:
        payload["num_frames"] = seconds_to_frames(args.seconds)
    return payload


def cmd_health(args: argparse.Namespace) -> int:
    data = _http_json("GET", f"{args.base_url}/health")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def cmd_image(args: argparse.Namespace) -> int:
    payload = build_image_payload(args)
    data = _http_json("POST", f"{args.base_url}/generate", payload)
    print(json.dumps({
        "ok": True,
        "type": "image",
        "saved_paths": data.get("saved_paths", []),
        "quality": args.quality,
        "count": args.count,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    payload = build_video_payload(args)
    data = _http_json("POST", f"{args.base_url}/video", payload)
    print(json.dumps({
        "ok": True,
        "type": "video",
        "saved_paths": data.get("saved_paths", []),
        "quality": args.quality,
        "seconds": args.seconds,
    }, ensure_ascii=False, indent=2))
    return 0


# Lightweight alias maps (kept dependency-free so the CLI runs on any python3
# without torch/diffusers). Keep in sync with src/imagegen.py MODEL_ALIASES and
# src/videogen.py VIDEO_MODEL_ALIASES.
IMAGE_ALIASES = {
    "turbo": "stabilityai/sd-turbo",
    "turbo-xl": "stabilityai/sdxl-turbo",
    "schnell": "black-forest-labs/FLUX.1-schnell",
    "klein": "black-forest-labs/FLUX.2-klein-base-9B",
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "z-img": "Zhibei-ai/Z-Img",
    "schnell-gguf": "city96/FLUX.1-schnell-gguf/flux1-schnell-Q4_K_S.gguf",
}

VIDEO_ALIASES = {
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan13": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan2.1": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan-1.3b": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "wan-diffusers": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}


def cmd_models(_: argparse.Namespace) -> int:
    print(json.dumps({
        "image_aliases": IMAGE_ALIASES,
        "video_aliases": VIDEO_ALIASES,
    }, ensure_ascii=False, indent=2))
    return 0


def cmd_setup(_: argparse.Namespace) -> int:
    setup_script = os.path.join(ROOT_DIR, "setup.sh")
    if not os.path.exists(setup_script):
        print(json.dumps({"ok": False, "error": f"setup.sh not found at {setup_script}"}, ensure_ascii=False))
        return 1
    proc = subprocess.run(["bash", setup_script], cwd=ROOT_DIR, capture_output=True, text=True)
    print(json.dumps({
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }, ensure_ascii=False, indent=2))
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fantasia", description="Open Fantasia CLI")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Fantasia server base URL")

    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("image", help="Generate image(s)")
    p_img.add_argument("--prompt", required=True)
    p_img.add_argument("--quality", choices=["low", "mid", "high"], default="low")
    p_img.add_argument("--model", default=None)
    p_img.add_argument("--count", type=int, default=1)
    p_img.add_argument("--seed", type=int, default=42)
    p_img.add_argument("--raw", action="store_true", help="Disable prompt enhancement")
    p_img.set_defaults(func=cmd_image)

    p_vid = sub.add_parser("video", help="Generate one video")
    p_vid.add_argument("--prompt", required=True)
    p_vid.add_argument("--quality", choices=["low", "mid", "high"], default="mid")
    p_vid.add_argument("--model", default=None)
    p_vid.add_argument("--seed", type=int, default=42)
    p_vid.add_argument("--seconds", type=int, default=5)
    p_vid.add_argument("--raw", action="store_true", help="Disable prompt enhancement")
    p_vid.set_defaults(func=cmd_video)

    p_health = sub.add_parser("health", help="Server health")
    p_health.set_defaults(func=cmd_health)

    p_models = sub.add_parser("models", help="List aliases")
    p_models.set_defaults(func=cmd_models)

    p_setup = sub.add_parser("setup", help="Run setup.sh")
    p_setup.set_defaults(func=cmd_setup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
