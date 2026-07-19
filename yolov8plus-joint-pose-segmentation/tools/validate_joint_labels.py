#!/usr/bin/env python3
"""Validate YOLOv8+ joint label files before training."""

from __future__ import annotations

import argparse
from pathlib import Path


TOKENS = ("id:", "bbox:", "keypoints:", "segmentations:")


def values_between(line: str, start: str, end: str | None) -> list[str]:
    value = line.split(start, 1)[1]
    if end is not None:
        value = value.split(end, 1)[0]
    return value.split()


def validate_line(line: str, keypoint_count: int, class_count: int) -> list[str]:
    errors: list[str] = []
    if any(token not in line for token in TOKENS):
        return [f"expected tokens in order: {' '.join(TOKENS)}"]
    if not (
        line.index("id:") < line.index("bbox:") < line.index("keypoints:") < line.index("segmentations:")
    ):
        return ["tokens are not in the expected order"]

    try:
        class_values = values_between(line, "id:", "bbox:")
        bbox = [float(value) for value in values_between(line, "bbox:", "keypoints:")]
        keypoints = [float(value) for value in values_between(line, "keypoints:", "segmentations:")]
        polygon = [float(value) for value in values_between(line, "segmentations:", None)]
        if len(class_values) != 1 or not class_values[0].isdigit():
            errors.append("class ID must be one non-negative integer")
        else:
            class_id = int(class_values[0])
            if class_id >= class_count:
                errors.append(f"class ID {class_id} must be less than nc={class_count}")
        if len(bbox) != 4 or any(value < 0 or value > 1 for value in bbox):
            errors.append("bbox must contain four normalized values")
        if len(keypoints) != keypoint_count * 3:
            errors.append(f"expected {keypoint_count * 3} keypoint values, found {len(keypoints)}")
        else:
            for index in range(0, len(keypoints), 3):
                if not 0 <= keypoints[index] <= 1 or not 0 <= keypoints[index + 1] <= 1:
                    errors.append("keypoint x/y values must be normalized")
                    break
                if keypoints[index + 2] not in (0, 1, 2):
                    errors.append("keypoint visibility values must be 0, 1, or 2")
                    break
        if len(polygon) < 6 or len(polygon) % 2 or any(value < 0 or value > 1 for value in polygon):
            errors.append("segmentation must contain at least three normalized x/y pairs")
    except (IndexError, ValueError) as error:
        errors.append(str(error))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--keypoints", type=int, default=17)
    parser.add_argument("--classes", type=int, default=1)
    args = parser.parse_args()

    failures = 0
    files = sorted(args.labels.glob("*.txt"))
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for error in validate_line(line, args.keypoints, args.classes):
                failures += 1
                print(f"{path}:{line_number}: {error}")

    if failures:
        raise SystemExit(f"Validation failed with {failures} error(s).")
    print(f"Validated {len(files)} label files.")


if __name__ == "__main__":
    main()
