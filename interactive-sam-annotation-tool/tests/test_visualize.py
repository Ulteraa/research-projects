import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sam_annotation_tool.visualize import render_document  # noqa: E402


class VisualizationTests(unittest.TestCase):
    def test_renders_nested_image_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "images" / "nested" / "sample.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (40, 30), "white").save(image_path)
            data = {
                "categories": [{"id": 1, "name": "object"}],
                "images": [
                    {"id": 1, "file_name": "nested/sample.png", "width": 40, "height": 30}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]],
                        "bbox": [5, 5, 15, 15],
                        "area": 225,
                        "iscrowd": 0,
                    }
                ],
            }
            output = root / "rendered"
            self.assertEqual(render_document(data, root / "images", output, None), 1)
            self.assertTrue((output / "nested" / "sample.png").is_file())


if __name__ == "__main__":
    unittest.main()
