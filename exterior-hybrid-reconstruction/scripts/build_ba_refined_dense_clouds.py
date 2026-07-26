from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DENSE_DIR = (
    ROOT
    / "outputs/dense_hybrid/vggt_dense_predictions"
)

DEFAULT_BA_MODEL = (
    ROOT
    / "outputs/vggt_ba/sparse"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/dense_hybrid/ba_refined"
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


def pycolmap_rotation(image: Any) -> np.ndarray:
    pose = value_or_call(image.cam_from_world)
    rotation = value_or_call(pose.rotation)
    matrix = value_or_call(rotation.matrix)

    return np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(3, 3)


def camera_matrix(camera: Any) -> np.ndarray:
    matrix = value_or_call(
        camera.calibration_matrix
    )

    return np.asarray(
        matrix,
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
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    source_variance = float(
        np.mean(
            np.sum(source_centered**2, axis=1)
        )
    )

    if source_variance < 1e-15:
        raise RuntimeError(
            "Degenerate source camera configuration"
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


def transform_points_sim3(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (
        scale * (points @ rotation.T)
        + translation.reshape(1, 3)
    )


def unproject_to_world(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera_rotation: np.ndarray,
    camera_center: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        world_points: [N,3]
        pixel_x: [N]
        pixel_y: [N]
    """
    height, width = depth.shape

    x_coordinates = np.arange(
        0,
        width,
        stride,
        dtype=np.int32,
    )
    y_coordinates = np.arange(
        0,
        height,
        stride,
        dtype=np.int32,
    )

    pixel_x, pixel_y = np.meshgrid(
        x_coordinates,
        y_coordinates,
    )

    pixel_x = pixel_x.reshape(-1)
    pixel_y = pixel_y.reshape(-1)

    z = depth[pixel_y, pixel_x]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    x = (
        pixel_x.astype(np.float64) - cx
    ) / fx * z

    y = (
        pixel_y.astype(np.float64) - cy
    ) / fy * z

    camera_points = np.stack(
        [x, y, z],
        axis=1,
    )

    # world-to-camera:
    #     x_camera = R * x_world + t
    #
    # Therefore:
    #     x_world = R^T * x_camera + C
    #
    # For row vectors:
    #     x_world = x_camera @ R + C
    world_points = (
        camera_points
        @ world_to_camera_rotation
        + camera_center.reshape(1, 3)
    )

    return world_points, pixel_x, pixel_y


def scaled_intrinsic(
    intrinsic: np.ndarray,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    scale_matrix = np.asarray(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return scale_matrix @ intrinsic


def rotation_error_degrees(
    rotation_a: np.ndarray,
    rotation_b: np.ndarray,
) -> float:
    relative = rotation_a @ rotation_b.T

    cosine = (
        np.trace(relative) - 1.0
    ) / 2.0

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def distribution_statistics(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    view_ids: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<f4"),
            ("view_id", "u1"),
        ]
    )

    vertices = np.empty(
        len(xyz),
        dtype=vertex_dtype,
    )

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    vertices["confidence"] = confidence
    vertices["view_id"] = view_ids

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property float confidence\n"
        "property uchar view_id\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dense_dir",
        type=Path,
        default=DEFAULT_DENSE_DIR,
    )
    parser.add_argument(
        "--ba_model",
        type=Path,
        default=DEFAULT_BA_MODEL,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help=(
            "Pixel sampling stride. Use 2 for validation; "
            "use 1 later for full-resolution fusion."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError(
            "--stride must be at least 1"
        )

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists; "
                "use --overwrite"
            )

        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dense_cameras = np.load(
        args.dense_dir / "cameras.npz"
    )

    image_names = [
        canonical_name(str(name))
        for name in dense_cameras["image_names"]
    ]

    dense_extrinsics = np.asarray(
        dense_cameras[
            "extrinsics_camera_from_world"
        ],
        dtype=np.float64,
    )

    dense_intrinsics = np.asarray(
        dense_cameras["intrinsics"],
        dtype=np.float64,
    )

    dense_indices = {
        name: index
        for index, name in enumerate(image_names)
    }

    dense_centers_by_name = {
        name: camera_center_from_w2c(
            dense_extrinsics[index]
        )
        for name, index in dense_indices.items()
    }

    ba_reconstruction = pycolmap.Reconstruction(
        str(args.ba_model)
    )

    ba_by_name: dict[str, dict[str, Any]] = {}

    for image in ba_reconstruction.images.values():
        name = canonical_name(image.name)
        camera = ba_reconstruction.cameras[
            image.camera_id
        ]

        rotation = pycolmap_rotation(image)
        center = pycolmap_camera_center(image)
        translation = -rotation @ center

        ba_by_name[name] = {
            "image": image,
            "camera": camera,
            "rotation": rotation,
            "center": center,
            "translation": translation,
            "intrinsic_original": camera_matrix(
                camera
            ),
            "width": int(camera.width),
            "height": int(camera.height),
        }

    common_names = sorted(
        set(dense_indices)
        & set(ba_by_name)
    )

    if len(common_names) < 3:
        raise RuntimeError(
            f"Only {len(common_names)} common cameras"
        )

    dense_centers = np.asarray(
        [
            dense_centers_by_name[name]
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

    scale, alignment_rotation, alignment_translation = (
        estimate_sim3(
            dense_centers,
            ba_centers,
        )
    )

    aligned_dense_centers = transform_points_sim3(
        dense_centers,
        scale,
        alignment_rotation,
        alignment_translation,
    )

    camera_center_errors = np.linalg.norm(
        aligned_dense_centers - ba_centers,
        axis=1,
    )

    print("Dense VGGT → BA alignment")
    print("  scale:", scale)
    print(
        "  camera-center RMSE:",
        float(
            np.sqrt(
                np.mean(camera_center_errors**2)
            )
        ),
    )

    feedforward_points_all = []
    ba_vggt_intrinsics_points_all = []
    ba_scaled_intrinsics_points_all = []

    rgb_all = []
    confidence_all = []
    view_ids_all = []

    pose_displacements = []
    intrinsic_displacements = []
    rotation_errors = []

    processed_ba_intrinsics = []
    ba_extrinsics = []
    ba_original_sizes = []

    per_view_reports = []

    for view_id, name in enumerate(common_names):
        dense_index = dense_indices[name]
        ba_record = ba_by_name[name]

        depth_path = (
            args.dense_dir
            / "depth"
            / f"{Path(name).stem}.npy"
        )
        confidence_path = (
            args.dense_dir
            / "depth_confidence"
            / f"{Path(name).stem}.npy"
        )
        rgb_path = (
            args.dense_dir
            / "processed_rgb"
            / f"{Path(name).stem}.png"
        )

        depth_dense_units = np.load(
            depth_path
        ).astype(np.float64)

        confidence = np.load(
            confidence_path
        ).astype(np.float32)

        rgb_image = np.asarray(
            Image.open(rgb_path).convert("RGB"),
            dtype=np.uint8,
        )

        if depth_dense_units.shape != confidence.shape:
            raise RuntimeError(
                f"{name}: depth/confidence shape mismatch"
            )

        if rgb_image.shape[:2] != depth_dense_units.shape:
            raise RuntimeError(
                f"{name}: RGB/depth shape mismatch"
            )

        processed_height, processed_width = (
            depth_dense_units.shape
        )

        vggt_intrinsic = dense_intrinsics[
            dense_index
        ]

        ba_intrinsic_processed = scaled_intrinsic(
            ba_record["intrinsic_original"],
            ba_record["width"],
            ba_record["height"],
            processed_width,
            processed_height,
        )

        processed_ba_intrinsics.append(
            ba_intrinsic_processed
        )

        ba_rotation = ba_record["rotation"]
        ba_center = ba_record["center"]
        ba_translation = ba_record["translation"]

        ba_extrinsics.append(
            np.hstack(
                [
                    ba_rotation,
                    ba_translation.reshape(3, 1),
                ]
            )
        )

        ba_original_sizes.append(
            [
                ba_record["width"],
                ba_record["height"],
            ]
        )

        # Depth and camera translations must use the same scale.
        depth_ba_units = (
            depth_dense_units * scale
        )

        dense_extrinsic = dense_extrinsics[
            dense_index
        ]
        dense_rotation = dense_extrinsic[:, :3]
        dense_center = camera_center_from_w2c(
            dense_extrinsic
        )

        # Baseline: original VGGT depth and camera, followed
        # by the single global Sim(3) into the BA frame.
        feedforward_world_dense, pixel_x, pixel_y = (
            unproject_to_world(
                depth_dense_units,
                vggt_intrinsic,
                dense_rotation,
                dense_center,
                args.stride,
            )
        )

        feedforward_world_ba = transform_points_sim3(
            feedforward_world_dense,
            scale,
            alignment_rotation,
            alignment_translation,
        )

        # Hybrid A: BA pose with the intrinsics associated
        # with VGGT's predicted depth.
        ba_vggt_intrinsics_world, _, _ = (
            unproject_to_world(
                depth_ba_units,
                vggt_intrinsic,
                ba_rotation,
                ba_center,
                args.stride,
            )
        )

        # Hybrid B: BA pose plus BA intrinsics rescaled to
        # the processed VGGT depth-map resolution.
        ba_scaled_intrinsics_world, _, _ = (
            unproject_to_world(
                depth_ba_units,
                ba_intrinsic_processed,
                ba_rotation,
                ba_center,
                args.stride,
            )
        )

        sampled_depth = depth_ba_units[
            pixel_y,
            pixel_x,
        ]

        sampled_confidence = confidence[
            pixel_y,
            pixel_x,
        ]

        sampled_rgb = rgb_image[
            pixel_y,
            pixel_x,
        ]

        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & np.isfinite(sampled_confidence)
            & np.isfinite(
                feedforward_world_ba
            ).all(axis=1)
            & np.isfinite(
                ba_vggt_intrinsics_world
            ).all(axis=1)
            & np.isfinite(
                ba_scaled_intrinsics_world
            ).all(axis=1)
        )

        feedforward_world_ba = (
            feedforward_world_ba[valid]
        )
        ba_vggt_intrinsics_world = (
            ba_vggt_intrinsics_world[valid]
        )
        ba_scaled_intrinsics_world = (
            ba_scaled_intrinsics_world[valid]
        )

        sampled_confidence = (
            sampled_confidence[valid]
        )
        sampled_rgb = sampled_rgb[valid]

        view_ids = np.full(
            len(sampled_confidence),
            view_id,
            dtype=np.uint8,
        )

        pose_displacement = np.linalg.norm(
            ba_vggt_intrinsics_world
            - feedforward_world_ba,
            axis=1,
        )

        intrinsic_displacement = np.linalg.norm(
            ba_scaled_intrinsics_world
            - ba_vggt_intrinsics_world,
            axis=1,
        )

        pose_displacements.append(
            pose_displacement
        )
        intrinsic_displacements.append(
            intrinsic_displacement
        )

        feedforward_points_all.append(
            feedforward_world_ba
        )
        ba_vggt_intrinsics_points_all.append(
            ba_vggt_intrinsics_world
        )
        ba_scaled_intrinsics_points_all.append(
            ba_scaled_intrinsics_world
        )

        rgb_all.append(sampled_rgb)
        confidence_all.append(sampled_confidence)
        view_ids_all.append(view_ids)

        # After changing the VGGT world frame by A,
        # its world-to-camera rotation becomes R_v * A^T.
        aligned_dense_rotation = (
            dense_rotation
            @ alignment_rotation.T
        )

        orientation_error = rotation_error_degrees(
            aligned_dense_rotation,
            ba_rotation,
        )

        rotation_errors.append(
            orientation_error
        )

        intrinsic_difference = float(
            np.linalg.norm(
                vggt_intrinsic
                - ba_intrinsic_processed
            )
        )

        per_view_reports.append(
            {
                "image_name": name,
                "valid_sampled_points": int(
                    len(sampled_confidence)
                ),
                "processed_size": [
                    processed_width,
                    processed_height,
                ],
                "ba_original_size": [
                    ba_record["width"],
                    ba_record["height"],
                ],
                "vggt_intrinsic": (
                    vggt_intrinsic.tolist()
                ),
                "ba_intrinsic_processed": (
                    ba_intrinsic_processed.tolist()
                ),
                "intrinsic_frobenius_difference": (
                    intrinsic_difference
                ),
                "aligned_vggt_to_ba_rotation_error_deg": (
                    orientation_error
                ),
            }
        )

        print(
            f"{name}: "
            f"{len(sampled_confidence):,} points, "
            f"K difference={intrinsic_difference:.3f}, "
            f"rotation change={orientation_error:.3f} deg"
        )

    feedforward_points = np.concatenate(
        feedforward_points_all,
        axis=0,
    )

    ba_vggt_intrinsics_points = np.concatenate(
        ba_vggt_intrinsics_points_all,
        axis=0,
    )

    ba_scaled_intrinsics_points = np.concatenate(
        ba_scaled_intrinsics_points_all,
        axis=0,
    )

    rgb = np.concatenate(
        rgb_all,
        axis=0,
    )

    confidence = np.concatenate(
        confidence_all,
        axis=0,
    )

    view_ids = np.concatenate(
        view_ids_all,
        axis=0,
    )

    pose_displacements_array = np.concatenate(
        pose_displacements,
        axis=0,
    )

    intrinsic_displacements_array = np.concatenate(
        intrinsic_displacements,
        axis=0,
    )

    output_files = {
        "feedforward_sim3": (
            args.output_dir
            / "dense_feedforward_sim3_to_ba.ply"
        ),
        "ba_vggt_intrinsics": (
            args.output_dir
            / "dense_ba_vggt_intrinsics.ply"
        ),
        "ba_scaled_intrinsics": (
            args.output_dir
            / "dense_ba_scaled_ba_intrinsics.ply"
        ),
    }

    write_binary_ply(
        output_files["feedforward_sim3"],
        feedforward_points,
        rgb,
        confidence,
        view_ids,
    )

    write_binary_ply(
        output_files["ba_vggt_intrinsics"],
        ba_vggt_intrinsics_points,
        rgb,
        confidence,
        view_ids,
    )

    write_binary_ply(
        output_files["ba_scaled_intrinsics"],
        ba_scaled_intrinsics_points,
        rgb,
        confidence,
        view_ids,
    )

    np.savez_compressed(
        args.output_dir / "cameras_ba_refined.npz",
        image_names=np.asarray(common_names),
        depth_scale_dense_to_ba=np.asarray(
            scale,
            dtype=np.float64,
        ),
        dense_to_ba_rotation=alignment_rotation,
        dense_to_ba_translation=alignment_translation,
        dense_extrinsics_camera_from_world=(
            dense_extrinsics[
                [dense_indices[name] for name in common_names]
            ]
        ),
        ba_extrinsics_camera_from_world=np.asarray(
            ba_extrinsics,
            dtype=np.float64,
        ),
        vggt_intrinsics=dense_intrinsics[
            [dense_indices[name] for name in common_names]
        ],
        ba_intrinsics_processed=np.asarray(
            processed_ba_intrinsics,
            dtype=np.float64,
        ),
        ba_original_sizes=np.asarray(
            ba_original_sizes,
            dtype=np.int32,
        ),
    )

    report = {
        "milestone": "Dense Hybrid Fusion 2B",
        "description": (
            "Controlled comparison of dense VGGT depth "
            "with feed-forward and BA-refined cameras"
        ),
        "sampling_stride": args.stride,
        "common_camera_count": len(common_names),
        "image_names": common_names,
        "depth_scale_dense_to_ba": scale,
        "dense_to_ba_rotation": (
            alignment_rotation.tolist()
        ),
        "dense_to_ba_translation": (
            alignment_translation.tolist()
        ),
        "camera_center_alignment": {
            "rmse": float(
                np.sqrt(
                    np.mean(camera_center_errors**2)
                )
            ),
            "median": float(
                np.median(camera_center_errors)
            ),
            "maximum": float(
                np.max(camera_center_errors)
            ),
        },
        "orientation_change_degrees": (
            distribution_statistics(
                np.asarray(rotation_errors)
            )
        ),
        "point_displacement_due_to_ba_pose": (
            distribution_statistics(
                pose_displacements_array
            )
        ),
        "point_displacement_due_to_intrinsic_change": (
            distribution_statistics(
                intrinsic_displacements_array
            )
        ),
        "point_count_per_cloud": int(
            len(feedforward_points)
        ),
        "confidence_statistics": (
            distribution_statistics(confidence)
        ),
        "outputs": {
            key: str(path)
            for key, path in output_files.items()
        },
        "views": per_view_reports,
    }

    report_path = (
        args.output_dir / "metadata.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Dense BA-refined export completed")
    print("Depth multiplier:", scale)
    print(
        "Points per cloud:",
        f"{len(feedforward_points):,}",
    )
    print(
        "Pose displacement:",
        report[
            "point_displacement_due_to_ba_pose"
        ],
    )
    print(
        "Intrinsic displacement:",
        report[
            "point_displacement_due_to_intrinsic_change"
        ],
    )
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
