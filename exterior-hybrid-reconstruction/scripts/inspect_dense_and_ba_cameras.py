from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap


ROOT = Path(__file__).resolve().parents[1]

DENSE_DIR = (
    ROOT
    / "outputs/dense_hybrid/vggt_dense_predictions"
)

BA_MODEL = (
    ROOT
    / "outputs/vggt_ba/sparse"
)


def value_or_call(value: Any) -> Any:
    return value() if callable(value) else value


def canonical_name(name: str) -> str:
    return re.sub(
        r"^\d+_",
        "",
        Path(name).name,
    )


def camera_center_from_w2c(
    extrinsic: np.ndarray,
) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]

    return -rotation.T @ translation


def pycolmap_camera_center(image: Any) -> np.ndarray:
    return np.asarray(
        value_or_call(image.projection_center),
        dtype=np.float64,
    ).reshape(3)


def camera_matrix(camera: Any) -> np.ndarray:
    calibration = getattr(
        camera,
        "calibration_matrix",
        None,
    )

    if calibration is None:
        raise RuntimeError(
            "This PyCOLMAP camera does not expose "
            "calibration_matrix()"
        )

    return np.asarray(
        value_or_call(calibration),
        dtype=np.float64,
    ).reshape(3, 3)


def estimate_sim3(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate:
        target ~= scale * rotation @ source + translation
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_variance = float(
        np.mean(
            np.sum(source_centered**2, axis=1)
        )
    )

    covariance = (
        target_centered.T @ source_centered
    ) / len(source)

    u, singular_values, vt = np.linalg.svd(
        covariance
    )

    correction = np.eye(3)

    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt

    scale = float(
        np.sum(
            singular_values
            * np.diag(correction)
        )
        / source_variance
    )

    translation = (
        target_mean
        - scale * rotation @ source_mean
    )

    return scale, rotation, translation


def main() -> None:
    dense = np.load(
        DENSE_DIR / "cameras.npz"
    )

    names = [
        canonical_name(str(name))
        for name in dense["image_names"]
    ]

    dense_extrinsics = np.asarray(
        dense["extrinsics_camera_from_world"],
        dtype=np.float64,
    )

    dense_intrinsics = np.asarray(
        dense["intrinsics"],
        dtype=np.float64,
    )

    dense_by_name = {
        name: {
            "extrinsic": dense_extrinsics[index],
            "intrinsic": dense_intrinsics[index],
            "center": camera_center_from_w2c(
                dense_extrinsics[index]
            ),
        }
        for index, name in enumerate(names)
    }

    ba = pycolmap.Reconstruction(
        str(BA_MODEL)
    )

    ba_by_name = {}

    print("=== BA cameras ===")

    for image in sorted(
        ba.images.values(),
        key=lambda item: canonical_name(item.name),
    ):
        name = canonical_name(image.name)
        camera = ba.cameras[image.camera_id]

        model_name = getattr(
            camera,
            "model_name",
            None,
        )

        if model_name is None:
            model = getattr(camera, "model", "unknown")
            model_name = str(model)

        intrinsic = camera_matrix(camera)

        ba_by_name[name] = {
            "image": image,
            "camera": camera,
            "center": pycolmap_camera_center(image),
            "intrinsic": intrinsic,
        }

        print()
        print(name)
        print(
            "  size:",
            int(camera.width),
            "x",
            int(camera.height),
        )
        print("  model:", model_name)
        print(
            "  params:",
            np.asarray(camera.params),
        )
        print("  K:")
        print(intrinsic)

    common_names = sorted(
        set(dense_by_name)
        & set(ba_by_name)
    )

    print()
    print("=== Correspondence ===")
    print("Dense cameras:", len(dense_by_name))
    print("BA cameras:", len(ba_by_name))
    print("Common cameras:", len(common_names))
    print("Names:", common_names)

    if len(common_names) < 3:
        raise RuntimeError(
            "Not enough common cameras for Sim(3)"
        )

    dense_centers = np.asarray(
        [
            dense_by_name[name]["center"]
            for name in common_names
        ],
        dtype=np.float64,
    )

    ba_centers = np.asarray(
        [
            ba_by_name[name]["center"]
            for name in common_names
        ],
        dtype=np.float64,
    )

    scale, rotation, translation = estimate_sim3(
        dense_centers,
        ba_centers,
    )

    aligned_dense_centers = (
        scale
        * (rotation @ dense_centers.T).T
        + translation
    )

    errors = np.linalg.norm(
        aligned_dense_centers - ba_centers,
        axis=1,
    )

    print()
    print("=== Dense VGGT → BA Sim(3) ===")
    print("Scale:", scale)
    print("Rotation:")
    print(rotation)
    print("Translation:", translation)
    print(
        "Camera-center RMSE:",
        float(
            np.sqrt(np.mean(errors**2))
        ),
    )
    print(
        "Camera-center median error:",
        float(np.median(errors)),
    )
    print(
        "Camera-center maximum error:",
        float(np.max(errors)),
    )

    print()
    print("=== Intrinsic differences ===")

    intrinsic_errors = []

    for name in common_names:
        dense_k = dense_by_name[name]["intrinsic"]
        ba_k = ba_by_name[name]["intrinsic"]

        difference = np.linalg.norm(
            dense_k - ba_k
        )

        intrinsic_errors.append(difference)

        print(
            name,
            "||K_dense-K_BA||_F =",
            float(difference),
        )

    print()
    print(
        "Mean intrinsic matrix difference:",
        float(np.mean(intrinsic_errors)),
    )

    print()
    print("Important depth scale multiplier:")
    print(
        "Multiply VGGT depth by approximately",
        scale,
        "before using BA camera translations.",
    )


if __name__ == "__main__":
    main()
