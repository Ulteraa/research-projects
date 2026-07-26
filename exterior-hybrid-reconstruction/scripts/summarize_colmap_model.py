from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import pycolmap


def value_or_call(value: Any) -> Any:
    """Support PyCOLMAP APIs exposed as either properties or methods."""
    return value() if callable(value) else value


def track_length(point3d: Any) -> int:
    track = point3d.track

    if hasattr(track, "length"):
        return int(value_or_call(track.length))

    if hasattr(track, "elements"):
        return len(track.elements)

    return len(track)


def image_point3d_count(image: Any) -> int:
    if hasattr(image, "num_points3D"):
        return int(value_or_call(image.num_points3D))

    if hasattr(image, "points2D"):
        return sum(
            1
            for point2d in image.points2D
            if value_or_call(point2d.has_point3D)
        )

    return 0


def safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def safe_median(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def summarize(model_dir: Path) -> dict[str, Any]:
    reconstruction = pycolmap.Reconstruction(str(model_dir))

    images = list(reconstruction.images.values())
    points3d = list(reconstruction.points3D.values())

    observations_per_image = [
        image_point3d_count(image)
        for image in images
    ]

    track_lengths = [
        track_length(point)
        for point in points3d
    ]

    reprojection_errors = []
    for point in points3d:
        error = float(point.error)

        # VGGT feed-forward exports may not have BA-updated errors.
        if math.isfinite(error) and error >= 0:
            reprojection_errors.append(error)

    return {
        "model_dir": str(model_dir),
        "num_cameras": int(reconstruction.num_cameras()),
        "num_registered_images": int(reconstruction.num_reg_images()),
        "num_points3D": int(reconstruction.num_points3D()),
        "mean_observations_per_image": safe_mean(observations_per_image),
        "median_observations_per_image": safe_median(observations_per_image),
        "mean_track_length": safe_mean(track_lengths),
        "median_track_length": safe_median(track_lengths),
        "mean_point_reprojection_error_px": safe_mean(reprojection_errors),
        "median_point_reprojection_error_px": safe_median(
            reprojection_errors
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a COLMAP sparse reconstruction."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"COLMAP model directory not found: {args.model_dir}"
        )

    result = summarize(args.model_dir)

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
