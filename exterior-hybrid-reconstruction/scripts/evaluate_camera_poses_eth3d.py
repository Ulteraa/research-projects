from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


ROOT = Path(__file__).resolve().parents[1]

GT_MODEL = (
    ROOT
    / "data/eth3d_courtyard_full/courtyard"
    / "dslr_calibration_undistorted"
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

OUTPUT_PATH = (
    ROOT
    / "results/milestone1_12views"
    / "eth3d_camera_pose_evaluation.json"
)


def value_or_call(value: Any) -> Any:
    return value() if callable(value) else value


def canonical_name(name: str) -> str:
    """
    Convert both:
      000_DSC_0286.JPG
      dslr_images_undistorted/DSC_0286.JPG
    into:
      DSC_0286.JPG
    """
    basename = Path(name).name
    return re.sub(r"^\d+_", "", basename)


def image_map(
    reconstruction: pycolmap.Reconstruction,
) -> dict[str, Any]:
    return {
        canonical_name(image.name): image
        for image in reconstruction.images.values()
    }


def camera_center(image: Any) -> np.ndarray:
    center = value_or_call(image.projection_center)
    return np.asarray(center, dtype=np.float64).reshape(3)


def camera_rotation(image: Any) -> np.ndarray:
    pose = value_or_call(image.cam_from_world)
    rotation = value_or_call(pose.rotation)
    matrix = value_or_call(rotation.matrix)

    return np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(3, 3)


def estimate_sim3(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama least-squares similarity transform."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    variance = np.mean(
        np.sum(source_centered**2, axis=1)
    )

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
        ) / variance
    )

    translation = (
        target_mean
        - scale * rotation @ source_mean
    )

    return scale, rotation, translation


def rotation_error_degrees(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    relative = estimated @ ground_truth.T

    cosine = (
        np.trace(relative) - 1.0
    ) / 2.0

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(cosine))
    )


def maximum_pairwise_distance(
    points: np.ndarray,
) -> float:
    differences = (
        points[:, None, :]
        - points[None, :, :]
    )

    return float(
        np.max(
            np.linalg.norm(differences, axis=2)
        )
    )


def main() -> None:
    gt_reconstruction = pycolmap.Reconstruction(
        str(GT_MODEL)
    )
    gt_images = image_map(gt_reconstruction)

    report: dict[str, Any] = {
        "ground_truth_model": str(GT_MODEL),
        "models": {},
    }

    for key, model_path in MODEL_PATHS.items():
        reconstruction = pycolmap.Reconstruction(
            str(model_path)
        )
        estimated_images = image_map(reconstruction)

        common_names = sorted(
            set(gt_images) & set(estimated_images)
        )

        if len(common_names) < 3:
            raise RuntimeError(
                f"{key}: only {len(common_names)} matching images"
            )

        source_centers = np.asarray(
            [
                camera_center(estimated_images[name])
                for name in common_names
            ]
        )

        target_centers = np.asarray(
            [
                camera_center(gt_images[name])
                for name in common_names
            ]
        )

        scale, align_rotation, translation = estimate_sim3(
            source_centers,
            target_centers,
        )

        aligned_centers = (
            scale
            * (align_rotation @ source_centers.T).T
            + translation
        )

        translation_errors = np.linalg.norm(
            aligned_centers - target_centers,
            axis=1,
        )

        rotation_errors = []

        for name in common_names:
            estimated_rotation = camera_rotation(
                estimated_images[name]
            )
            gt_rotation = camera_rotation(
                gt_images[name]
            )

            # Convert the estimated world-to-camera rotation
            # into the ETH3D world coordinate system.
            aligned_rotation = (
                estimated_rotation @ align_rotation.T
            )

            rotation_errors.append(
                rotation_error_degrees(
                    aligned_rotation,
                    gt_rotation,
                )
            )

        rotation_errors = np.asarray(rotation_errors)

        trajectory_diameter = maximum_pairwise_distance(
            target_centers
        )

        rmse = float(
            np.sqrt(
                np.mean(translation_errors**2)
            )
        )

        report["models"][key] = {
            "common_images": len(common_names),
            "image_names": common_names,
            "sim3_scale_to_eth3d": scale,
            "eth3d_camera_path_diameter": trajectory_diameter,
            "translation_rmse": rmse,
            "translation_median": float(
                np.median(translation_errors)
            ),
            "translation_max": float(
                np.max(translation_errors)
            ),
            "translation_rmse_fraction_of_path": (
                rmse / trajectory_diameter
            ),
            "rotation_mean_degrees": float(
                np.mean(rotation_errors)
            ),
            "rotation_median_degrees": float(
                np.median(rotation_errors)
            ),
            "rotation_max_degrees": float(
                np.max(rotation_errors)
            ),
        }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
