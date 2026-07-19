import sys
import unittest
from pathlib import Path

import numpy as np


DEPLOYMENT = Path(__file__).resolve().parents[1] / "deployment"
sys.path.insert(0, str(DEPLOYMENT))

from keypoints import decode_keypoint_heatmaps  # noqa: E402


class KeypointDecodingTests(unittest.TestCase):
    def test_decodes_heatmap_centers_into_box_coordinates(self):
        heatmap = np.zeros((2, 4), dtype=np.float32)
        heatmap[1, 2] = 3.5

        decoded = decode_keypoint_heatmaps(heatmap, [10, 20, 50, 60])

        np.testing.assert_allclose(decoded, [[35, 50, 3.5]])

    def test_decodes_multiple_keypoints(self):
        heatmaps = np.zeros((2, 2, 2), dtype=np.float32)
        heatmaps[0, 0, 0] = 1
        heatmaps[1, 1, 1] = 2

        decoded = decode_keypoint_heatmaps(heatmaps, [0, 0, 20, 20])

        np.testing.assert_allclose(decoded, [[5, 5, 1], [15, 15, 2]])
        self.assertEqual(decoded.dtype, np.float32)

    def test_rejects_invalid_shape(self):
        with self.assertRaises(ValueError):
            decode_keypoint_heatmaps(np.zeros((1, 2, 3, 4)), [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
