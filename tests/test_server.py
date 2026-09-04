"""
Tests for the Open Fantasia server endpoints, hot-swap logic, and resolve_size
edge cases. Addresses GitHub issue #13:

  * /generate endpoint (200 PNG on success, 503-when-busy, hard caps, 500 on
    unknown quality since resolve_size raises an unhandled ValueError)
  * /health endpoint (ready vs busy, active-model detection)
  * /edit endpoint (disabled -> always 503)
  * resolve_size edge cases (unknown quality ValueError, explicit overrides)
  * model hot-swap behaviour (different model id triggers unload + reload)

Uses FastAPI TestClient with a fully mocked pipeline, so no GPU/model is needed.
"""

import os
import sys
import unittest
from unittest.mock import patch

from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server
from fastapi.testclient import TestClient


def make_fake_image():
    """A minimal stand-in for a PIL image that records .save()/file writes."""
    img = type("FakeImage", (), {})()
    img.save = lambda *a, **k: None
    return img


class FakePipe:
    """A fake SD-style pipeline (no .transformer attr) yielding one image."""

    def __init__(self):
        self.images = {"single": [make_fake_image()]}

    def __call__(self, **kwargs):
        return type("Res", (), {"images": [make_fake_image()]})()


def make_fake_pipe():
    """Return a callable fake SD-style pipeline (no .transformer attr)."""
    return FakePipe()


class TestServerGenerateEndpoint(unittest.TestCase):
    def setUp(self):
        server._pipe = None
        server._device = "cpu"
        server._model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        server._quant_mode = "none"
        server._vl_model = None
        server._edit_pipe = None

    def _run_generate(self, **payload):
        payload.setdefault("prompt", "a cat")
        with TestClient(server.app) as client:
            return client.post("/generate", json=payload)

    def test_generate_returns_png_on_success(self):
        server._pipe = make_fake_pipe()
        # The server imports PIL locally inside _do_generate, so patch the real
        # PIL.Image.open to return a fake image (no real file is written).
        with patch.object(server, "_check_ram_guard"), \
             patch.object(server, "enhance_prompt", side_effect=lambda p: p), \
             patch.object(PILImage, "open", return_value=make_fake_image()):
            resp = self._run_generate(enhance=False)

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        self.assertIn("X-Saved-Paths", resp.headers)

    def test_generate_503_when_busy_lock_held(self):
        """Second concurrent request must fail fast with 503 (no queueing)."""
        server._pipe = make_fake_pipe()
        acquired = server._generate_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            resp = self._run_generate(enhance=False)
        finally:
            server._generate_lock.release()
        self.assertEqual(resp.status_code, 503)

    def test_generate_returns_500_on_invalid_quality(self):
        """resolve_size raises ValueError for unknown quality -> unhandled 500."""
        server._pipe = make_fake_pipe()
        # Disable the ram guard so the only failure is the bad quality value.
        # raise_server_exceptions=False lets the unhandled ValueError surface as
        # a real 500 response instead of being re-raised by the TestClient.
        with patch.object(server, "_check_ram_guard"):
            with TestClient(server.app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/generate",
                    json={"prompt": "a cat", "quality": "ultra", "enhance": False},
                )
        self.assertEqual(resp.status_code, 500)

    def test_generate_rejects_oversized_resolution(self):
        """Hard cap on resolution -> explicit 400 before any pipeline call."""
        server._pipe = make_fake_pipe()
        resp = self._run_generate(width=99999, height=99999, enhance=False)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Resolution too large", resp.json()["detail"])


class TestServerHealthEndpoint(unittest.TestCase):
    def setUp(self):
        server._pipe = None
        server._vl_model = None
        server._edit_pipe = None
        server._model_id = None
        server._device = "cuda"
        server._quant_mode = "none"

    def _get_health(self):
        with TestClient(server.app) as client:
            return client.get("/health").json()

    def test_health_ready_when_idle(self):
        data = self._get_health()
        self.assertEqual(data["status"], "ready")
        self.assertFalse(data["flux_loaded"])
        self.assertEqual(data["active"], "none")

    def test_health_reports_busy_when_lock_held(self):
        acquired = server._generate_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            data = self._get_health()
        finally:
            server._generate_lock.release()
        self.assertEqual(data["status"], "busy")

    def test_health_detects_flux_loaded(self):
        server._pipe = make_fake_pipe()
        data = self._get_health()
        self.assertEqual(data["active"], "flux")
        self.assertTrue(data["flux_loaded"])


class TestServerEditEndpoint(unittest.TestCase):
    def test_edit_always_503_disabled(self):
        with TestClient(server.app) as client:
            resp = client.post(
                "/edit",
                json={"image": "/tmp/input.png", "prompt": "make it sunny"},
            )
        self.assertEqual(resp.status_code, 503)
        self.assertIn("disabled", resp.json()["detail"].lower())


class TestResolveSizeEdgeCases(unittest.TestCase):
    def test_unknown_quality_raises(self):
        from imagegen import resolve_size
        with self.assertRaises(ValueError):
            resolve_size(quality="ultra")

    def test_known_quality_presets(self):
        from imagegen import resolve_size
        self.assertEqual(resolve_size(quality="low")[:2], (512, 512))
        self.assertEqual(resolve_size(quality="mid")[:2], (768, 768))
        self.assertEqual(resolve_size(quality="high")[:2], (1024, 1024))

    def test_explicit_dims_override_quality(self):
        from imagegen import resolve_size
        self.assertEqual(resolve_size(quality="mid", width=640, height=480),
                         (640, 480, 8))

    def test_steps_override_preset(self):
        from imagegen import resolve_size
        self.assertEqual(resolve_size(quality="low", steps=12), (512, 512, 12))

    def test_no_args_uses_defaults(self):
        from imagegen import resolve_size
        self.assertEqual(resolve_size(), (512, 512, 4))


class TestHotSwap(unittest.TestCase):
    def setUp(self):
        server._pipe = None
        server._device = "cpu"
        server._model_id = None
        server._quant_mode = "none"
        server._vl_model = None
        server._edit_pipe = None

    def test_hot_swap_unloads_and_reloads_when_model_differs(self):
        """A /generate for a different model id must trigger _unload_flux_for_swap
        then a fresh load_model."""
        server._pipe = make_fake_pipe()
        server._model_id = "model-a"

        with patch.object(server, "_unload_flux_for_swap",
                          side_effect=lambda: setattr(server, "_pipe", None)) as mock_unload, \
             patch.object(server, "load_model",
                          side_effect=lambda model_id, *a, **k: (
                              setattr(server, "_pipe", make_fake_pipe()),
                              setattr(server, "_model_id", model_id),
                          )) as mock_load, \
             patch.object(server, "_check_ram_guard"), \
             patch.object(server, "enhance_prompt", side_effect=lambda p: p), \
             patch.object(PILImage, "open", return_value=make_fake_image()):
            with TestClient(server.app) as client:
                resp = client.post("/generate", json={
                    "prompt": "a cat", "model": "model-b", "enhance": False,
                })

        self.assertEqual(resp.status_code, 200)
        mock_unload.assert_called_once()
        mock_load.assert_called_once()
        self.assertEqual(server._model_id, "model-b")

    def test_same_model_does_not_unload(self):
        """Requesting the already-loaded model id must NOT hot-swap."""
        server._pipe = make_fake_pipe()
        server._model_id = "model-a"

        with patch.object(server, "_unload_flux_for_swap") as mock_unload, \
             patch.object(server, "load_model") as mock_load, \
             patch.object(server, "_check_ram_guard"), \
             patch.object(server, "enhance_prompt", side_effect=lambda p: p), \
             patch.object(PILImage, "open", return_value=make_fake_image()):
            with TestClient(server.app) as client:
                resp = client.post("/generate", json={
                    "prompt": "a cat", "model": "model-a", "enhance": False,
                })

        self.assertEqual(resp.status_code, 200)
        mock_unload.assert_not_called()
        mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
