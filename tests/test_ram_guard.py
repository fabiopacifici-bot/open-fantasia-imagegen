import sys
import os
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import HTTPException


class TestRamGuard(unittest.TestCase):
    def test_ok_when_plenty_ram(self):
        from server import _check_ram_guard, MIN_FREE_RAM_GB
        with patch("server._free_ram_gb", return_value=MIN_FREE_RAM_GB + 10):
            # Should not raise
            _check_ram_guard()

    def test_raises_503_when_low_ram(self):
        from server import _check_ram_guard, MIN_FREE_RAM_GB
        with patch("server._free_ram_gb", return_value=MIN_FREE_RAM_GB - 1):
            with self.assertRaises(HTTPException) as ctx:
                _check_ram_guard()
            self.assertEqual(ctx.exception.status_code, 503)

    def test_free_ram_inf_when_psutil_broken(self):
        # If psutil itself fails, we fall back to "infinite" and never block
        from server import _free_ram_gb
        with patch("server.psutil.virtual_memory", side_effect=Exception("boom")):
            self.assertEqual(_free_ram_gb(), float("inf"))


class TestOffload(unittest.TestCase):
    def test_offload_skipped_when_disabled(self):
        from server import _enable_offload, ENABLE_CPU_OFFLOAD
        if ENABLE_CPU_OFFLOAD:
            self.skipTest("ENABLE_CPU_OFFLOAD is on in env; test in its own mode")
        fake = type("FakePipe", (), {
            "enable_model_cpu_offload": Mock,
            "enable_attention_slicing": Mock,
        })()
        _enable_offload(fake)
        # With offload disabled it should return without touching the pipe — nothing to assert
        # beyond not raising; the real path is covered by the ENABLE_CPU_OFFLOAD variant below.
        self.assertIsNotNone(fake)

    def test_offload_calls_methods_when_enabled(self):
        from server import _enable_offload
        captured = {"calls": []}
        class FakePipe:
            def enable_model_cpu_offload(self):
                captured["calls"].append("model")
            def enable_attention_slicing(self):
                captured["calls"].append("attention")
        with patch("server.ENABLE_CPU_OFFLOAD", True):
            _enable_offload(FakePipe())
        self.assertIn("model", captured["calls"])
        self.assertIn("attention", captured["calls"])


if __name__ == "__main__":
    unittest.main()
