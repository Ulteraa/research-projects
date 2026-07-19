import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sam_annotation_tool.coco import (  # noqa: E402
    CocoDataset,
    bbox_from_polygons,
    polygon_area,
    validate_document,
)


CATEGORIES = [{"id": 1, "name": "object", "supercategory": "object"}]


class GeometryTests(unittest.TestCase):
    def test_polygon_area(self):
        self.assertEqual(polygon_area([0, 0, 4, 0, 0, 3]), 6.0)

    def test_bbox_uses_xywh_order(self):
        self.assertEqual(bbox_from_polygons([[1, 2, 5, 2, 5, 6, 1, 6]]), [1, 2, 4, 4])


class CocoDatasetTests(unittest.TestCase):
    def test_persists_integer_ids_and_correct_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.json"
            dataset = CocoDataset.open(output, CATEGORIES)
            image_id = dataset.add_image("nested/image.jpg", width=640, height=480)
            annotation_id = dataset.add_annotation(
                image_id=image_id,
                category_id=1,
                polygons=[[10, 20, 30, 20, 30, 40, 10, 40]],
                source="manual",
            )
            dataset.save()

            saved = json.loads(output.read_text())
            self.assertEqual(saved["images"][0]["width"], 640)
            self.assertEqual(saved["images"][0]["height"], 480)
            self.assertIsInstance(saved["images"][0]["id"], int)
            self.assertEqual(saved["annotations"][0]["id"], annotation_id)
            self.assertEqual(saved["annotations"][0]["bbox"], [10, 20, 20, 20])
            self.assertEqual(saved["annotations"][0]["attributes"]["source"], "manual")

            resumed = CocoDataset.open(output, CATEGORIES)
            self.assertEqual(resumed.add_image("nested/image.jpg", 640, 480), image_id)

    def test_undo_is_scoped_to_the_current_image(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = CocoDataset.open(Path(directory) / "annotations.json", CATEGORIES)
            first = dataset.add_image("one.jpg", 100, 100)
            second = dataset.add_image("two.jpg", 100, 100)
            for image_id in (first, second, first):
                dataset.add_annotation(
                    image_id=image_id,
                    category_id=1,
                    polygons=[[0, 0, 10, 0, 10, 10]],
                )
            removed = dataset.remove_last_annotation(first)
            self.assertEqual(removed["image_id"], first)
            self.assertEqual(len(dataset.annotations_for_image(second)), 1)

    def test_rejects_category_changes_when_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "annotations.json"
            dataset = CocoDataset.open(output, CATEGORIES)
            dataset.save()
            with self.assertRaises(ValueError):
                CocoDataset.open(output, [{"id": 2, "name": "different"}])

    def test_validator_finds_broken_reference(self):
        data = {
            "images": [],
            "categories": CATEGORIES,
            "annotations": [
                {
                    "id": 1,
                    "image_id": 99,
                    "category_id": 1,
                    "segmentation": [[0, 0, 1, 0, 1, 1]],
                    "area": 1,
                    "bbox": [0, 0, 1, 1],
                }
            ],
        }
        errors, _ = validate_document(data)
        self.assertTrue(any("unknown image" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
