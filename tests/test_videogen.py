import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestResolveVideoSize(unittest.TestCase):
    def test_quality_preset_mid(self):
        from videogen import resolve_video_size
        w, h, nf, s = resolve_video_size(quality="mid")
        self.assertEqual((w, h, nf, s), (480, 832, 81, 30))

    def test_quality_preset_low(self):
        from videogen import resolve_video_size
        w, h, nf, s = resolve_video_size(quality="low")
        self.assertEqual((w, h, nf, s), (480, 480, 49, 20))

    def test_unknown_quality_raises(self):
        from videogen import resolve_video_size
        with self.assertRaises(ValueError):
            resolve_video_size(quality="ultra")

    def test_explicit_values_override_quality(self):
        from videogen import resolve_video_size
        w, h, nf, s = resolve_video_size(quality="mid", width=640, height=360, steps=12, num_frames=33)
        self.assertEqual((w, h, nf, s), (640, 360, 33, 12))


class TestWanDetection(unittest.TestCase):
    def test_is_wan(self):
        from videogen import is_wan
        self.assertTrue(is_wan("Wan-AI/Wan2.1-T2V-1.3B"))
        self.assertTrue(is_wan("wan"))
        self.assertFalse(is_wan("black-forest-labs/FLUX.1-schnell"))


class TestGenerateVideoDryRun(unittest.TestCase):
    @patch("videogen.get_video_pipeline")
    def test_generate_calls_pipeline_and_saves_mp4(self, mock_get_pipeline):
        mock_pipe = MagicMock()
        mock_frame = MagicMock()
        mock_frame.convert.return_value = mock_frame
        # .frames[0] returns a list of PIL images
        mock_pipe.return_value.frames = [[mock_frame] * 3]
        mock_get_pipeline.return_value = (mock_pipe, "cpu")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name

        try:
            from videogen import generate_video
            with patch("videogen._save_mp4") as mock_save:
                generate_video(prompt="a cat walking", output=out, enhance=False)
                mock_save.assert_called_once()
                # Ensure the writer wrote the frames
                self.assertEqual(mock_save.call_args[0][0], [mock_frame] * 3)
        finally:
            if os.path.exists(out):
                os.unlink(out)

    @patch("videogen.get_video_pipeline")
    def test_preloaded_pipe_avoids_reload(self, mock_get_pipeline):
        mock_pipe = MagicMock()
        mock_frame = MagicMock()
        mock_frame.convert.return_value = mock_frame
        mock_pipe.return_value.frames = [[mock_frame] * 3]
        mock_get_pipeline.return_value = (mock_pipe, "cpu")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name

        try:
            from videogen import generate_video
            with patch("videogen._save_mp4"):
                generate_video(
                    prompt="a cat walking", output=out, enhance=False,
                    pipe=mock_pipe, device="cpu",
                )
            # get_video_pipeline should NOT be called when a pipe is pre-supplied
            mock_get_pipeline.assert_not_called()
        finally:
            if os.path.exists(out):
                os.unlink(out)


class TestSaveMp4(unittest.TestCase):
    def test_save_mp4_accepts_numpy_arrays(self):
        """Wan2.1 returns numpy arrays (H,W,C uint8), not PIL images — must encode."""
        import numpy as np
        from videogen import _save_mp4
        import tempfile
        frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name
        try:
            _save_mp4(frames, out, fps=16)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_save_mp4_accepts_pil_images(self):
        """PIL images still work (convert path)."""
        from PIL import Image
        from videogen import _save_mp4
        import tempfile
        frames = [Image.new("RGB", (16, 16)) for _ in range(3)]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out = f.name
        try:
            _save_mp4(frames, out, fps=16)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)
        finally:
            if os.path.exists(out):
                os.unlink(out)


if __name__ == "__main__":
    unittest.main()
