from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AnnotationToolTest(unittest.TestCase):
    def test_coco_conversion_and_validation(self) -> None:
        coco = {
            "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 80}],
            "categories": [{"id": 1, "name": "person"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "iscrowd": 0,
                    "bbox": [10, 8, 40, 48],
                    "segmentation": [[10, 8, 50, 8, 50, 56, 10, 56]],
                    "keypoints": [20, 16, 2] * 17,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            coco_path = temporary_path / "annotations.json"
            labels_path = temporary_path / "labels"
            coco_path.write_text(json.dumps(coco), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/convert_coco_joint.py"),
                    "--coco-json",
                    str(coco_path),
                    "--output-dir",
                    str(labels_path),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/validate_joint_labels.py"),
                    "--labels",
                    str(labels_path),
                    "--keypoints",
                    "17",
                    "--classes",
                    "1",
                ],
                check=True,
            )

            label = (labels_path / "sample.txt").read_text(encoding="utf-8")
            self.assertTrue(label.startswith("id: 0 bbox:"))
            self.assertIn("keypoints:", label)
            self.assertIn("segmentations:", label)


if __name__ == "__main__":
    unittest.main()
