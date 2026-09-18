import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fantasia as cli


class TestFantasiaCliPayloads(unittest.TestCase):
    def test_seconds_to_frames(self):
        self.assertEqual(cli.seconds_to_frames(5), 80)
        self.assertEqual(cli.seconds_to_frames(0), 1)

    def test_build_image_payload(self):
        args = type("Args", (), {
            "prompt": "robot cat",
            "quality": "low",
            "count": 2,
            "seed": 7,
            "raw": False,
            "model": "turbo",
        })()
        payload = cli.build_image_payload(args)
        self.assertEqual(payload["prompt"], "robot cat")
        self.assertEqual(payload["count"], 2)
        self.assertTrue(payload["enhance"])

    def test_build_video_payload_with_seconds(self):
        args = type("Args", (), {
            "prompt": "cat in colosseum",
            "quality": "mid",
            "seed": 42,
            "raw": True,
            "model": "wan",
            "seconds": 6,
        })()
        payload = cli.build_video_payload(args)
        self.assertFalse(payload["enhance"])
        self.assertEqual(payload["num_frames"], 96)


if __name__ == "__main__":
    unittest.main()
