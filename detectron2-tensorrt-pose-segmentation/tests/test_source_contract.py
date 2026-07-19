import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTests(unittest.TestCase):
    def test_no_developer_absolute_paths_remain(self):
        sources = list((ROOT / "configs").glob("*.yaml"))
        sources += list((ROOT / "deployment").glob("*.py"))
        combined = "\n".join(path.read_text() for path in sources)
        self.assertNotIn("/home/fariborz", combined)

    def test_graph_exports_keypoint_dimension(self):
        source = (ROOT / "deployment" / "create_onnx.py").read_text()
        self.assertIn("self.num_keypoints", source)
        self.assertIn('keypoint_heatmaps.name = "detection_keypoint_heatmaps"', source)
        self.assertIn("box_head_outputs + [mask_head_output, keypoint_heatmaps]", source)


if __name__ == "__main__":
    unittest.main()
