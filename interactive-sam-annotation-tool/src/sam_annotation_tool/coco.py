"""COCO polygon conversion, persistence, and validation utilities."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def polygon_area(polygon: Sequence[float]) -> float:
    """Return the shoelace area of a flattened ``[x1, y1, ...]`` polygon."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def bbox_from_polygons(polygons: Sequence[Sequence[float]]) -> list[float]:
    """Return an XYWH box enclosing all polygons."""
    points = [np.asarray(polygon, dtype=np.float64).reshape(-1, 2) for polygon in polygons]
    points = [polygon for polygon in points if len(polygon) >= 3]
    if not points:
        raise ValueError("At least one polygon with three points is required")
    merged = np.concatenate(points, axis=0)
    xmin, ymin = merged.min(axis=0)
    xmax, ymax = merged.max(axis=0)
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def polygons_to_mask(
    polygons: Sequence[Sequence[float]], height: int, width: int
) -> np.ndarray:
    """Rasterize COCO polygons into a boolean mask."""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    contours = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            contours.append(np.rint(points).astype(np.int32))
    if contours:
        cv2.fillPoly(mask, contours, 1)
    return mask.astype(bool)


def mask_to_geometry(
    mask: np.ndarray, *, min_area: float = 10.0, simplify: float = 1.0
) -> tuple[list[list[float]], list[float], float]:
    """Convert a binary mask to COCO polygons, XYWH box, and pixel area.

    External contours are retained. COCO polygon encoding cannot represent
    holes, so holes are filled; use RLE instead if hole fidelity is required.
    """
    import cv2

    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {binary.shape}")
    if not binary.any():
        raise ValueError("Cannot annotate an empty mask")

    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(0.0, float(simplify))
        if epsilon:
            contour = cv2.approxPolyDP(contour, epsilon, closed=True)
        flattened = contour.reshape(-1, 2).astype(float).reshape(-1).tolist()
        if len(flattened) >= 6:
            polygons.append(flattened)

    if not polygons:
        raise ValueError("No contour remained after area filtering")

    ys, xs = np.nonzero(binary)
    xmin, xmax = int(xs.min()), int(xs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    bbox = [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]
    return polygons, bbox, float(binary.sum())


class CocoDataset:
    """Small resumable COCO dataset writer for an interactive session."""

    def __init__(self, output: Path, data: dict):
        self.output = Path(output)
        self.data = data

    @classmethod
    def open(cls, output: Path, categories: Sequence[dict]) -> "CocoDataset":
        output = Path(output)
        normalized = _normalize_categories(categories)
        if output.exists():
            data = json.loads(output.read_text(encoding="utf-8"))
            errors, _ = validate_document(data)
            if errors:
                raise ValueError("Existing COCO file is invalid: " + "; ".join(errors[:5]))
            existing = _normalize_categories(data.get("categories", []))
            if existing != normalized:
                raise ValueError("Existing COCO categories do not match the requested categories")
        else:
            data = {
                "info": {
                    "description": "Interactive SAM annotation dataset",
                    "version": "1.0",
                },
                "licenses": [],
                "categories": normalized,
                "images": [],
                "annotations": [],
            }
        return cls(output, data)

    def add_image(self, file_name: str, width: int, height: int) -> int:
        for image in self.data["images"]:
            if image["file_name"] == file_name:
                if image["width"] != width or image["height"] != height:
                    raise ValueError(f"Dimension mismatch for existing image {file_name}")
                return int(image["id"])

        image_id = _next_id(self.data["images"])
        self.data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
            }
        )
        return image_id

    def add_annotation(
        self,
        *,
        image_id: int,
        category_id: int,
        polygons: Sequence[Sequence[float]],
        bbox: Sequence[float] | None = None,
        area: float | None = None,
        source: str = "manual",
    ) -> int:
        self._require_references(image_id, category_id)
        normalized = [[float(value) for value in polygon] for polygon in polygons]
        normalized = [polygon for polygon in normalized if len(polygon) >= 6]
        if not normalized:
            raise ValueError("At least one valid polygon is required")

        bbox = list(bbox) if bbox is not None else bbox_from_polygons(normalized)
        area = float(area) if area is not None else sum(polygon_area(p) for p in normalized)
        annotation_id = _next_id(self.data["annotations"])
        self.data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": int(image_id),
                "category_id": int(category_id),
                "segmentation": normalized,
                "area": area,
                "bbox": [float(value) for value in bbox],
                "iscrowd": 0,
                "attributes": {"source": source},
            }
        )
        return annotation_id

    def add_mask(
        self,
        *,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
        source: str,
        min_area: float,
        simplify: float,
    ) -> int:
        polygons, bbox, area = mask_to_geometry(
            mask, min_area=min_area, simplify=simplify
        )
        return self.add_annotation(
            image_id=image_id,
            category_id=category_id,
            polygons=polygons,
            bbox=bbox,
            area=area,
            source=source,
        )

    def annotations_for_image(self, image_id: int) -> list[dict]:
        return [
            annotation
            for annotation in self.data["annotations"]
            if int(annotation["image_id"]) == int(image_id)
        ]

    def remove_last_annotation(self, image_id: int) -> dict | None:
        for index in range(len(self.data["annotations"]) - 1, -1, -1):
            if int(self.data["annotations"][index]["image_id"]) == int(image_id):
                return self.data["annotations"].pop(index)
        return None

    def erase_box(
        self,
        *,
        image_id: int,
        box_xyxy: Sequence[float],
        min_area: float,
        simplify: float,
    ) -> int:
        image = next(item for item in self.data["images"] if item["id"] == image_id)
        height, width = int(image["height"]), int(image["width"])
        x0, y0, x1, y1 = _clipped_box(box_xyxy, width, height)
        changed = 0
        for annotation in list(self.annotations_for_image(image_id)):
            mask = polygons_to_mask(annotation["segmentation"], height, width)
            before = int(mask.sum())
            mask[y0:y1, x0:x1] = False
            if int(mask.sum()) == before:
                continue
            changed += 1
            if not mask.any():
                self.data["annotations"].remove(annotation)
                continue
            try:
                polygons, bbox, area = mask_to_geometry(
                    mask, min_area=min_area, simplify=simplify
                )
            except ValueError:
                self.data["annotations"].remove(annotation)
                continue
            annotation["segmentation"] = polygons
            annotation["bbox"] = bbox
            annotation["area"] = area
            annotation.setdefault("attributes", {})["edited_with"] = "erase_box"
        return changed

    def annotation_mask(self, annotation: dict) -> np.ndarray:
        image_id = int(annotation["image_id"])
        image = next(item for item in self.data["images"] if int(item["id"]) == image_id)
        return polygons_to_mask(
            annotation["segmentation"], int(image["height"]), int(image["width"])
        )

    def save(self) -> None:
        errors, _ = validate_document(self.data)
        if errors:
            raise ValueError("Refusing to write invalid COCO data: " + "; ".join(errors[:5]))
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.output.parent, delete=False
        ) as handle:
            json.dump(self.data, handle, indent=2)
            handle.write("\n")
            temp_name = handle.name
        os.replace(temp_name, self.output)

    def snapshot(self) -> dict:
        return deepcopy(self.data)

    def _require_references(self, image_id: int, category_id: int) -> None:
        if image_id not in {int(item["id"]) for item in self.data["images"]}:
            raise ValueError(f"Unknown image_id {image_id}")
        if category_id not in {int(item["id"]) for item in self.data["categories"]}:
            raise ValueError(f"Unknown category_id {category_id}")


def validate_document(data: dict, image_root: Path | None = None) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a COCO polygon document."""
    errors: list[str] = []
    warnings: list[str] = []
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            errors.append(f"{key} must be a list")
    if errors:
        return errors, warnings

    image_ids = _unique_ids(data["images"], "image", errors)
    category_ids = _unique_ids(data["categories"], "category", errors)
    _unique_ids(data["annotations"], "annotation", errors)
    images = {int(item["id"]): item for item in data["images"] if "id" in item}

    for image in data["images"]:
        if int(image.get("width", 0)) <= 0 or int(image.get("height", 0)) <= 0:
            errors.append(f"image {image.get('id')} has invalid dimensions")
        if image_root is not None and not (Path(image_root) / image.get("file_name", "")).is_file():
            errors.append(f"image file is missing: {image.get('file_name')}")

    for annotation in data["annotations"]:
        annotation_id = annotation.get("id")
        image_id = int(annotation.get("image_id", -1))
        category_id = int(annotation.get("category_id", -1))
        if image_id not in image_ids:
            errors.append(f"annotation {annotation_id} references unknown image {image_id}")
        if category_id not in category_ids:
            errors.append(f"annotation {annotation_id} references unknown category {category_id}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(float(x) < 0 for x in bbox[2:]):
            errors.append(f"annotation {annotation_id} has invalid bbox")
        if float(annotation.get("area", -1)) < 0:
            errors.append(f"annotation {annotation_id} has invalid area")
        polygons = annotation.get("segmentation")
        if not isinstance(polygons, list) or not polygons:
            errors.append(f"annotation {annotation_id} has no polygons")
            continue
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                errors.append(f"annotation {annotation_id} has an invalid polygon")
                break
        if image_id in images and isinstance(bbox, list) and len(bbox) == 4:
            width, height = images[image_id]["width"], images[image_id]["height"]
            if bbox[0] + bbox[2] > width + 1 or bbox[1] + bbox[3] > height + 1:
                warnings.append(f"annotation {annotation_id} bbox extends beyond its image")
    return errors, warnings


def _normalize_categories(categories: Sequence[dict]) -> list[dict]:
    normalized = []
    for category in categories:
        if "id" not in category or "name" not in category:
            raise ValueError("Every category needs id and name")
        item = {
            "id": int(category["id"]),
            "name": str(category["name"]),
            "supercategory": str(category.get("supercategory", "object")),
        }
        normalized.append(item)
    if not normalized or len({item["id"] for item in normalized}) != len(normalized):
        raise ValueError("Categories must be non-empty and have unique IDs")
    return sorted(normalized, key=lambda item: item["id"])


def _next_id(items: Iterable[dict]) -> int:
    return max((int(item["id"]) for item in items), default=0) + 1


def _unique_ids(items: Sequence[dict], label: str, errors: list[str]) -> set[int]:
    ids = [int(item.get("id", -1)) for item in items]
    if len(ids) != len(set(ids)):
        errors.append(f"{label} IDs must be unique")
    if any(value < 0 for value in ids):
        errors.append(f"{label} IDs must be non-negative integers")
    return set(ids)


def _clipped_box(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    x0, x1 = sorted((int(round(box[0])), int(round(box[2]))))
    y0, y1 = sorted((int(round(box[1])), int(round(box[3]))))
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Erase box is empty after clipping")
    return x0, y0, x1, y1
