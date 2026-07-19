#!/usr/bin/env python3
"""Convert COCO polygon/keypoint annotations to the YOLOv8+ joint format."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def polygon_area(flat_polygon: list[float]) -> float:
    points = list(zip(flat_polygon[0::2], flat_polygon[1::2]))
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def normalize_keypoints(keypoints: list[float], width: int, height: int) -> list[float]:
    normalized: list[float] = []
    for index in range(0, len(keypoints), 3):
        normalized.extend(
            [keypoints[index] / width, keypoints[index + 1] / height, keypoints[index + 2]]
        )
    return normalized


def select_polygon(segmentation: Any) -> list[float] | None:
    if not isinstance(segmentation, list):
        return None  # COCO RLE is intentionally not converted by this utility.
    polygons = [polygon for polygon in segmentation if isinstance(polygon, list) and len(polygon) >= 6]
    return max(polygons, key=polygon_area) if polygons else None


def format_numbers(values: list[float]) -> str:
    return " ".join(f"{value:.8g}" for value in values)


def convert(coco_json: Path, output_dir: Path, include_crowd: bool = False) -> None:
    with coco_json.open(encoding="utf-8") as handle:
        data = json.load(handle)

    images = {image["id"]: image for image in data["images"]}
    category_ids = sorted(category["id"] for category in data["categories"])
    category_map = {category_id: index for index, category_id in enumerate(category_ids)}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_instances = 0
    skipped_instances = 0

    for image_id, image in images.items():
        width, height = image["width"], image["height"]
        lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            if annotation.get("iscrowd", 0) and not include_crowd:
                skipped_instances += 1
                continue

            polygon = select_polygon(annotation.get("segmentation"))
            keypoints = annotation.get("keypoints", [])
            if polygon is None or not keypoints or len(keypoints) % 3:
                skipped_instances += 1
                continue

            x, y, box_width, box_height = annotation["bbox"]
            bbox = [
                (x + box_width / 2.0) / width,
                (y + box_height / 2.0) / height,
                box_width / width,
                box_height / height,
            ]
            normalized_polygon = [
                value / (width if index % 2 == 0 else height)
                for index, value in enumerate(polygon)
            ]
            normalized_keypoints = normalize_keypoints(keypoints, width, height)
            class_id = category_map[annotation["category_id"]]
            lines.append(
                f"id: {class_id} bbox: {format_numbers(bbox)} "
                f"keypoints: {format_numbers(normalized_keypoints)} "
                f"segmentations: {format_numbers(normalized_polygon)}"
            )
            written_instances += 1

        label_name = Path(image["file_name"]).with_suffix(".txt").name
        (output_dir / label_name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Wrote {written_instances} instances to {output_dir}")
    print(f"Skipped {skipped_instances} unsupported or incomplete instances")
    print(f"Category mapping (COCO -> contiguous): {category_map}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-crowd", action="store_true")
    args = parser.parse_args()
    convert(args.coco_json, args.output_dir, args.include_crowd)


if __name__ == "__main__":
    main()
