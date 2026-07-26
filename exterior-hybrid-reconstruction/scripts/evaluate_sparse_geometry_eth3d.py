from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]

GT_MODEL = (
    ROOT
    / "data/eth3d_courtyard_full/courtyard"
    / "dslr_calibration_undistorted"
)

GT_SCAN_SAMPLE = (
    ROOT
    / "results/milestone1_12views"
    / "eth3d_scan_global_sample.ply"
)

MODEL_PATHS = {
    "vggt_feedforward": (
        ROOT / "outputs/vggt_feedforward_warm/sparse"
    ),
    "vggt_ba": (
        ROOT / "outputs/vggt_ba/sparse"
    ),
    "colmap": (
        ROOT / "outputs/colmap/sparse/0"
    ),
}


def value_or_call(value: Any) -> Any:
    return value() if callable(value) else value


def canonical_name(name: str) -> str:
    basename = Path(name).name
    return re.sub(r"^\d+_", "", basename)


def reconstruction_image_map(
    reconstruction: pycolmap.Reconstruction,
) -> dict[str, tuple[int, Any]]:
    return {
        canonical_name(image.name): (int(image_id), image)
        for image_id, image in reconstruction.images.items()
    }


def camera_center(image: Any) -> np.ndarray:
    return np.asarray(
        value_or_call(image.projection_center),
        dtype=np.float64,
    ).reshape(3)


def reconstruction_xyz(
    reconstruction: pycolmap.Reconstruction,
) -> np.ndarray:
    return np.asarray(
        [
            np.asarray(point.xyz, dtype=np.float64)
            for point in reconstruction.points3D.values()
        ],
        dtype=np.float64,
    )


def estimate_sim3(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama Sim(3): target ~= scale * R * source + translation."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_variance = float(
        np.mean(np.sum(source_centered**2, axis=1))
    )

    if source_variance < 1e-15:
        raise ValueError("Degenerate source camera configuration")

    covariance = (
        target_centered.T @ source_centered
    ) / source.shape[0]

    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt

    scale = float(
        np.sum(
            singular_values * np.diag(correction)
        ) / source_variance
    )

    translation = (
        target_mean - scale * rotation @ source_mean
    )

    return scale, rotation, translation


def transform_xyz(
    xyz: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (
        scale * (rotation @ xyz.T).T
        + translation.reshape(1, 3)
    )


def read_binary_xyz_ply(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None

        while True:
            line = handle.readline()

            if not line:
                raise RuntimeError(
                    f"No end_header found in {path}"
                )

            text = line.decode("ascii").strip()

            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])

            if text == "end_header":
                offset = handle.tell()
                break

    if vertex_count is None:
        raise RuntimeError(
            f"No vertex count found in {path}"
        )

    mapped = np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=offset,
        shape=(vertex_count, 3),
    )

    return np.asarray(mapped, dtype=np.float64)


def write_binary_xyz_ply(
    path: Path,
    xyz: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype="<f4")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        xyz.tofile(handle)


def voxel_downsample(
    xyz: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    if len(xyz) == 0:
        return xyz

    voxel_keys = np.floor(
        xyz / voxel_size
    ).astype(np.int64)

    _, first_indices = np.unique(
        voxel_keys,
        axis=0,
        return_index=True,
    )

    return xyz[np.sort(first_indices)]


def crop_points(
    xyz: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float]:
    inside = np.all(
        (xyz >= lower) & (xyz <= upper),
        axis=1,
    )

    fraction_inside = float(np.mean(inside))

    return xyz[inside], fraction_inside


def gt_sparse_region(
    reconstruction: pycolmap.Reconstruction,
    selected_image_ids: set[int],
    lower_percentile: float,
    upper_percentile: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    visible_xyz = []

    for point in reconstruction.points3D.values():
        elements = value_or_call(point.track.elements)

        visible = any(
            int(value_or_call(element.image_id))
            in selected_image_ids
            for element in elements
        )

        if visible:
            visible_xyz.append(
                np.asarray(point.xyz, dtype=np.float64)
            )

    visible_xyz = np.asarray(
        visible_xyz,
        dtype=np.float64,
    )

    if len(visible_xyz) == 0:
        raise RuntimeError(
            "No GT sparse points found for selected images"
        )

    lower = (
        np.percentile(
            visible_xyz,
            lower_percentile,
            axis=0,
        )
        - margin
    )

    upper = (
        np.percentile(
            visible_xyz,
            upper_percentile,
            axis=0,
        )
        + margin
    )

    return lower, upper, len(visible_xyz)


def distance_statistics(
    distances: np.ndarray,
) -> dict[str, float]:
    return {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(
            np.sqrt(np.mean(distances**2))
        ),
        "p90_m": float(np.percentile(distances, 90)),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
    }


def threshold_metrics(
    prediction_to_gt: np.ndarray,
    gt_to_prediction: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for threshold in thresholds:
        precision = float(
            np.mean(prediction_to_gt <= threshold)
        )
        recall = float(
            np.mean(gt_to_prediction <= threshold)
        )

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        output[f"{threshold:.3f}_m"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Common voxel size in ETH3D metric coordinates.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--lower_percentile",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--upper_percentile",
        type=float,
        default=99.5,
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.02,0.05,0.10,0.20",
    )
    args = parser.parse_args()

    thresholds = [
        float(value)
        for value in args.thresholds.split(",")
    ]

    output_dir = (
        ROOT
        / "results/milestone1_12views"
        / "sparse_geometry_diagnostic"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_reconstruction = pycolmap.Reconstruction(
        str(GT_MODEL)
    )
    gt_images = reconstruction_image_map(
        gt_reconstruction
    )

    reference_reconstruction = pycolmap.Reconstruction(
        str(MODEL_PATHS["vggt_feedforward"])
    )
    reference_names = sorted(
        reconstruction_image_map(
            reference_reconstruction
        )
    )

    selected_names = [
        name
        for name in reference_names
        if name in gt_images
    ]

    selected_gt_ids = {
        gt_images[name][0]
        for name in selected_names
    }

    lower, upper, sparse_region_point_count = (
        gt_sparse_region(
            gt_reconstruction,
            selected_gt_ids,
            args.lower_percentile,
            args.upper_percentile,
            args.margin,
        )
    )

    print("Evaluation bounds:")
    print("  lower:", lower)
    print("  upper:", upper)

    gt_scan = read_binary_xyz_ply(
        GT_SCAN_SAMPLE
    )
    gt_crop, gt_fraction_inside = crop_points(
        gt_scan,
        lower,
        upper,
    )
    gt_eval = voxel_downsample(
        gt_crop,
        args.voxel_size,
    )

    if len(gt_eval) == 0:
        raise RuntimeError(
            "GT crop is empty"
        )

    write_binary_xyz_ply(
        output_dir / "eth3d_gt_crop_voxelized.ply",
        gt_eval,
    )

    gt_tree = cKDTree(gt_eval)

    report: dict[str, Any] = {
        "diagnostic_type": (
            "Sparse geometry diagnostic; "
            "not official ETH3D dense-MVS evaluation"
        ),
        "selected_images": selected_names,
        "voxel_size_m": args.voxel_size,
        "thresholds_m": thresholds,
        "crop": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "gt_sparse_points_used_for_bounds": (
                sparse_region_point_count
            ),
            "gt_scan_sample_points": int(len(gt_scan)),
            "gt_scan_fraction_inside_crop": (
                gt_fraction_inside
            ),
            "gt_points_inside_crop": int(len(gt_crop)),
            "gt_voxelized_points": int(len(gt_eval)),
        },
        "models": {},
    }

    for key, model_path in MODEL_PATHS.items():
        print(f"\nEvaluating {key}")

        reconstruction = pycolmap.Reconstruction(
            str(model_path)
        )
        estimated_images = reconstruction_image_map(
            reconstruction
        )

        common_names = sorted(
            set(selected_names)
            & set(estimated_images)
            & set(gt_images)
        )

        source_centers = np.asarray(
            [
                camera_center(
                    estimated_images[name][1]
                )
                for name in common_names
            ],
            dtype=np.float64,
        )

        target_centers = np.asarray(
            [
                camera_center(
                    gt_images[name][1]
                )
                for name in common_names
            ],
            dtype=np.float64,
        )

        scale, rotation, translation = estimate_sim3(
            source_centers,
            target_centers,
        )

        xyz = reconstruction_xyz(
            reconstruction
        )
        aligned_xyz = transform_xyz(
            xyz,
            scale,
            rotation,
            translation,
        )

        cropped_xyz, fraction_inside = crop_points(
            aligned_xyz,
            lower,
            upper,
        )

        evaluated_xyz = voxel_downsample(
            cropped_xyz,
            args.voxel_size,
        )

        if len(evaluated_xyz) == 0:
            raise RuntimeError(
                f"{key}: no points inside evaluation crop"
            )

        write_binary_xyz_ply(
            output_dir / f"{key}_eth3d_aligned.ply",
            evaluated_xyz,
        )

        prediction_to_gt = gt_tree.query(
            evaluated_xyz,
            k=1,
            workers=-1,
        )[0]

        prediction_tree = cKDTree(
            evaluated_xyz
        )

        gt_to_prediction = prediction_tree.query(
            gt_eval,
            k=1,
            workers=-1,
        )[0]

        report["models"][key] = {
            "common_cameras": len(common_names),
            "sim3_scale_to_eth3d": scale,
            "source_points": int(len(xyz)),
            "fraction_inside_crop": fraction_inside,
            "points_inside_crop": int(len(cropped_xyz)),
            "voxelized_points": int(len(evaluated_xyz)),
            "accuracy_prediction_to_gt": (
                distance_statistics(prediction_to_gt)
            ),
            "completeness_gt_to_prediction": (
                distance_statistics(gt_to_prediction)
            ),
            "threshold_metrics": threshold_metrics(
                prediction_to_gt,
                gt_to_prediction,
                thresholds,
            ),
        }

    output_json = (
        output_dir
        / "sparse_geometry_evaluation.json"
    )

    output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSaved:", output_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
