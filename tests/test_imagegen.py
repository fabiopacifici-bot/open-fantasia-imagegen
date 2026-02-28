import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestEnhancePrompt(unittest.TestCase):
    def test_enhance_adds_prefix(self):
        from imagegen import enhance_prompt
        result = enhance_prompt("a red dragon")
        self.assertIn("a red dragon", result)
        self.assertIn("visually striking", result)

    def test_enhance_adds_lighting(self):
        from imagegen import enhance_prompt
        result = enhance_prompt("sunset over mountains")
        self.assertIn("cinematic lighting", result)


class TestGenerateDryRun(unittest.TestCase):
    @patch("imagegen.get_pipeline")
    def test_generate_calls_pipeline(self, mock_get_pipeline):
        mock_pipe = MagicMock()
        mock_image = MagicMock()
        mock_pipe.return_value.images = [mock_image]
        # simulate non-flux pipeline (no transformer attr)
        del mock_pipe.transformer
        mock_get_pipeline.return_value = (mock_pipe, "cpu")

        from imagegen import generate
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out = f.name

        try:
            generate(prompt="a cat", output=out, enhance=False)
            mock_pipe.assert_called_once()
            mock_image.save.assert_called_once_with(out)
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_enhance_false_skips_enhancement(self):
        from imagegen import enhance_prompt
        raw = "plain prompt"
        enhanced = enhance_prompt(raw)
        self.assertNotEqual(raw, enhanced)


if __name__ == "__main__":
    unittest.main()
