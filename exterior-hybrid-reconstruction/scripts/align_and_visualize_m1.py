from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pycolmap


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "milestone1_12views"
ALIGNED_MODEL_DIR = RESULT_DIR / "aligned_models"

MODEL_PATHS = {
    "vggt_feedforward": (
        "VGGT feed-forward",
        ROOT / "outputs/vggt_feedforward_warm/sparse",
    ),
    "vggt_ba": (
        "VGGT + bundle adjustment",
        ROOT / "outputs/vggt_ba/sparse",
    ),
    "colmap": (
        "Classical COLMAP",
        ROOT / "outputs/colmap/sparse/0",
    ),
}

REFERENCE_KEY = "colmap"


def value_or_call(value: Any) -> Any:
    return value() if callable(value) else value


def camera_center(image: Any) -> np.ndarray:
    return np.asarray(
        value_or_call(image.projection_center),
        dtype=np.float64,
    ).reshape(3)


def image_map(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
    return {
        image.name: image
        for image in reconstruction.images.values()
    }


def points_and_colors(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[np.ndarray, np.ndarray]:
    points = list(reconstruction.points3D.values())

    xyz = np.asarray(
        [np.asarray(point.xyz, dtype=np.float64) for point in points],
        dtype=np.float64,
    )

    rgb = np.asarray(
        [np.asarray(point.color, dtype=np.uint8) for point in points],
        dtype=np.uint8,
    )

    return xyz, rgb


def sim3_matrix(sim3: pycolmap.Sim3d) -> np.ndarray:
    return np.asarray(
        value_or_call(sim3.matrix),
        dtype=np.float64,
    )


def sim3_scale(sim3: pycolmap.Sim3d) -> float:
    return float(
        np.asarray(sim3.scale, dtype=np.float64).reshape(-1)[0]
    )


def maximum_pairwise_distance(points: np.ndarray) -> float:
    pairwise = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(pairwise, axis=2)
    return float(np.max(distances))


def save_model(
    key: str,
    reconstruction: pycolmap.Reconstruction,
) -> None:
    model_dir = ALIGNED_MODEL_DIR / key

    if model_dir.exists():
        shutil.rmtree(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    reconstruction.write(str(model_dir))
    reconstruction.export_PLY(
        str(RESULT_DIR / f"{key}_aligned.ply")
    )


def set_equal_axes(
    ax: Any,
    center: np.ndarray,
    radius: float,
) -> None:
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_reconstruction(
    key: str,
    title: str,
    reconstruction: pycolmap.Reconstruction,
    plot_center: np.ndarray,
    plot_radius: float,
) -> None:
    xyz, rgb = points_and_colors(reconstruction)

    max_plot_points = 40_000
    if len(xyz) > max_plot_points:
        indices = np.linspace(
            0,
            len(xyz) - 1,
            max_plot_points,
            dtype=np.int64,
        )
        xyz = xyz[indices]
        rgb = rgb[indices]

    images = sorted(
        reconstruction.images.values(),
        key=lambda image: image.name,
    )
    centers = np.asarray(
        [camera_center(image) for image in images],
        dtype=np.float64,
    )

    figure = plt.figure(figsize=(9, 8))
    ax = figure.add_subplot(111, projection="3d")

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=rgb.astype(np.float64) / 255.0,
        s=0.35,
        depthshade=False,
    )

    ax.plot(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        linewidth=1.5,
        label="Camera path",
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        s=18,
    )

    set_equal_axes(ax, plot_center, plot_radius)

    ax.set_title(title)
    ax.set_xlabel("X — COLMAP frame")
    ax.set_ylabel("Y — COLMAP frame")
    ax.set_zlabel("Z — COLMAP frame")
    ax.view_init(elev=22, azim=-62)
    ax.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(
        RESULT_DIR / f"{key}_aligned.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)



def estimate_sim3_umeyama(
    source: np.ndarray,
    target: np.ndarray,
) -> pycolmap.Sim3d:
    """Least-squares Sim(3) mapping source points onto target points."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: source={source.shape}, target={target.shape}"
        )

    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"Expected Nx3 point arrays, received {source.shape}"
        )

    if source.shape[0] < 3:
        raise ValueError("At least three correspondences are required")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_variance = float(
        np.mean(np.sum(source_centered ** 2, axis=1))
    )

    if source_variance < 1e-15:
        raise ValueError("Degenerate source camera-center configuration")

    covariance = (
        target_centered.T @ source_centered
    ) / source.shape[0]

    u, singular_values, vt = np.linalg.svd(covariance)

    sign_correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign_correction[-1, -1] = -1.0

    rotation = u @ sign_correction @ vt

    scale = float(
        np.sum(
            singular_values * np.diag(sign_correction)
        ) / source_variance
    )

    translation = (
        target_mean
        - scale * rotation @ source_mean
    )

    matrix_3x4 = np.hstack(
        [
            scale * rotation,
            translation.reshape(3, 1),
        ]
    )

    return pycolmap.Sim3d(matrix_3x4)

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ALIGNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = MODEL_PATHS[REFERENCE_KEY][1]
    reference = pycolmap.Reconstruction(str(reference_path))
    reference_images = image_map(reference)

    reference_names = sorted(reference_images)
    reference_centers = np.asarray(
        [
            camera_center(reference_images[name])
            for name in reference_names
        ],
        dtype=np.float64,
    )

    reference_diameter = maximum_pairwise_distance(
        reference_centers
    )

    aligned: dict[str, pycolmap.Reconstruction] = {}
    report: dict[str, Any] = {
        "reference_model": REFERENCE_KEY,
        "reference_camera_path_diameter": reference_diameter,
        "models": {},
    }

    for key, (title, model_path) in MODEL_PATHS.items():
        reconstruction = pycolmap.Reconstruction(str(model_path))
        reconstruction_images = image_map(reconstruction)

        common_names = sorted(
            set(reference_images) & set(reconstruction_images)
        )

        if len(common_names) < 3:
            raise RuntimeError(
                f"{key}: only {len(common_names)} common images"
            )

        if key == REFERENCE_KEY:
            report["models"][key] = {
                "title": title,
                "common_cameras": len(common_names),
                "sim3_scale_to_colmap": 1.0,
                "camera_center_rmse": 0.0,
                "camera_center_rmse_fraction_of_reference_diameter": 0.0,
                "camera_center_max_error": 0.0,
                "sim3_matrix_3x4": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            }
        else:
            source_centers = np.asarray(
                [
                    camera_center(reconstruction_images[name])
                    for name in common_names
                ],
                dtype=np.float64,
            )

            target_centers = np.asarray(
                [
                    camera_center(reference_images[name])
                    for name in common_names
                ],
                dtype=np.float64,
            )

            sim3 = estimate_sim3_umeyama(
                source_centers,
                target_centers,
            )

            reconstruction.transform(sim3)

            transformed_images = image_map(reconstruction)
            transformed_centers = np.asarray(
                [
                    camera_center(transformed_images[name])
                    for name in common_names
                ],
                dtype=np.float64,
            )

            errors = np.linalg.norm(
                transformed_centers - target_centers,
                axis=1,
            )

            rmse = float(np.sqrt(np.mean(errors ** 2)))
            max_error = float(np.max(errors))

            report["models"][key] = {
                "title": title,
                "common_cameras": len(common_names),
                "sim3_scale_to_colmap": sim3_scale(sim3),
                "camera_center_rmse": rmse,
                "camera_center_rmse_fraction_of_reference_diameter": (
                    rmse / reference_diameter
                    if reference_diameter > 0
                    else None
                ),
                "camera_center_max_error": max_error,
                "sim3_matrix_3x4": sim3_matrix(sim3).tolist(),
            }

        aligned[key] = reconstruction
        save_model(key, reconstruction)

    all_xyz = np.concatenate(
        [
            points_and_colors(reconstruction)[0]
            for reconstruction in aligned.values()
        ],
        axis=0,
    )

    lower = np.percentile(all_xyz, 1.0, axis=0)
    upper = np.percentile(all_xyz, 99.0, axis=0)

    plot_center = (lower + upper) / 2.0
    plot_radius = float(np.max(upper - lower) / 2.0)
    plot_radius *= 1.08

    for key, reconstruction in aligned.items():
        title = MODEL_PATHS[key][0]
        plot_reconstruction(
            key,
            title,
            reconstruction,
            plot_center,
            plot_radius,
        )

    report_path = RESULT_DIR / "alignment_report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved aligned models: {ALIGNED_MODEL_DIR}")
    print("Saved three matched-view PNG visualizations")


if __name__ == "__main__":
    main()
