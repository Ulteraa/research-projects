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

BA_MODEL = (
    ROOT
    / "outputs/vggt_ba/sparse"
)

DEFAULT_INPUT_DIR = (
    ROOT
    / "outputs/dense_hybrid/ba_refined"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "results/milestone2_dense_hybrid"
)

SPARSE_REPORT = (
    ROOT
    / "results/milestone1_12views"
    / "sparse_geometry_diagnostic"
    / "sparse_geometry_evaluation.json"
)

GT_CROP_PLY = (
    ROOT
    / "results/milestone1_12views"
    / "sparse_geometry_diagnostic"
    / "eth3d_gt_crop_voxelized.ply"
)

DENSE_VARIANTS = {
    "dense_feedforward": (
        "dense_feedforward_sim3_to_ba.ply"
    ),
    "dense_ba_vggt_intrinsics": (
        "dense_ba_vggt_intrinsics.ply"
    ),
    "dense_ba_scaled_ba_intrinsics": (
        "dense_ba_scaled_ba_intrinsics.ply"
    ),
}


PLY_TYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def value_or_call(value: Any) -> Any:
    return value() if callable(value) else value


def canonical_name(name: str) -> str:
    return re.sub(
        r"^\d+_",
        "",
        Path(name).name,
    )


def image_map(
    reconstruction: pycolmap.Reconstruction,
) -> dict[str, Any]:
    return {
        canonical_name(image.name): image
        for image in reconstruction.images.values()
    }


def camera_center(image: Any) -> np.ndarray:
    return np.asarray(
        value_or_call(image.projection_center),
        dtype=np.float64,
    ).reshape(3)


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
            "Degenerate camera-center configuration"
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


def transform_xyz(
    xyz: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return (
        scale * (xyz @ rotation.T)
        + translation.reshape(1, 3)
    )


def read_binary_ply(
    path: Path,
) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        format_name = None
        vertex_count = None
        current_element = None
        vertex_properties: list[
            tuple[str, str]
        ] = []

        while True:
            line = handle.readline()

            if not line:
                raise RuntimeError(
                    f"No end_header found in {path}"
                )

            text = line.decode(
                "ascii",
                errors="strict",
            ).strip()

            tokens = text.split()

            if tokens[:1] == ["format"]:
                format_name = tokens[1]

            elif tokens[:2] == ["element", "vertex"]:
                vertex_count = int(tokens[2])
                current_element = "vertex"

            elif tokens[:1] == ["element"]:
                current_element = tokens[1]

            elif (
                tokens[:1] == ["property"]
                and current_element == "vertex"
            ):
                if tokens[1] == "list":
                    raise RuntimeError(
                        "List-valued vertex properties "
                        "are not supported"
                    )

                property_type = tokens[1]
                property_name = tokens[2]

                if property_type not in PLY_TYPE_MAP:
                    raise RuntimeError(
                        f"Unsupported PLY type: "
                        f"{property_type}"
                    )

                vertex_properties.append(
                    (
                        property_name,
                        PLY_TYPE_MAP[property_type],
                    )
                )

            elif text == "end_header":
                data_offset = handle.tell()
                break

    if format_name != "binary_little_endian":
        raise RuntimeError(
            f"{path}: expected binary little-endian PLY, "
            f"got {format_name}"
        )

    if vertex_count is None:
        raise RuntimeError(
            f"{path}: missing vertex count"
        )

    vertex_dtype = np.dtype(vertex_properties)

    vertices = np.memmap(
        path,
        mode="r",
        dtype=vertex_dtype,
        offset=data_offset,
        shape=(vertex_count,),
    )

    output = {
        name: np.asarray(vertices[name])
        for name, _ in vertex_properties
    }

    for required in ["x", "y", "z"]:
        if required not in output:
            raise RuntimeError(
                f"{path}: missing property {required}"
            )

    return output


def xyz_from_ply(
    properties: dict[str, np.ndarray],
) -> np.ndarray:
    return np.stack(
        [
            properties["x"],
            properties["y"],
            properties["z"],
        ],
        axis=1,
    ).astype(np.float64)


def write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    view_ids: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dtype_fields: list[
        tuple[str, str]
    ] = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
    ]

    if rgb is not None:
        dtype_fields.extend(
            [
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )

    if confidence is not None:
        dtype_fields.append(
            ("confidence", "<f4")
        )

    if view_ids is not None:
        dtype_fields.append(
            ("view_id", "u1")
        )

    vertices = np.empty(
        len(xyz),
        dtype=np.dtype(dtype_fields),
    )

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    property_lines = [
        "property float x",
        "property float y",
        "property float z",
    ]

    if rgb is not None:
        vertices["red"] = rgb[:, 0]
        vertices["green"] = rgb[:, 1]
        vertices["blue"] = rgb[:, 2]

        property_lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )

    if confidence is not None:
        vertices["confidence"] = confidence
        property_lines.append(
            "property float confidence"
        )

    if view_ids is not None:
        vertices["view_id"] = view_ids
        property_lines.append(
            "property uchar view_id"
        )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        + "\n".join(property_lines)
        + "\nend_header\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def crop_mask(
    xyz: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.all(
        (xyz >= lower)
        & (xyz <= upper),
        axis=1,
    )


def voxel_downsample_indices(
    xyz: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    voxel_coordinates = np.floor(
        xyz / voxel_size
    ).astype(np.int64)

    _, indices = np.unique(
        voxel_coordinates,
        axis=0,
        return_index=True,
    )

    return np.sort(indices)


def distance_statistics(
    distances: np.ndarray,
) -> dict[str, float]:
    return {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(
            np.sqrt(
                np.mean(distances**2)
            )
        ),
        "p90_m": float(
            np.percentile(distances, 90)
        ),
        "p95_m": float(
            np.percentile(distances, 95)
        ),
        "p99_m": float(
            np.percentile(distances, 99)
        ),
        "maximum_m": float(
            np.max(distances)
        ),
    }


def threshold_metrics(
    prediction_to_gt: np.ndarray,
    gt_to_prediction: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}

    for threshold in thresholds:
        precision = float(
            np.mean(
                prediction_to_gt <= threshold
            )
        )

        recall = float(
            np.mean(
                gt_to_prediction <= threshold
            )
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
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--thresholds",
        default="0.02,0.05,0.10,0.20",
    )

    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help=(
            "Comma-separated name=filename pairs. "
            "Example: support1=a.ply,support2=b.ply"
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = [
        float(value)
        for value in args.thresholds.split(",")
    ]

    variants = dict(DENSE_VARIANTS)

    if args.variants:
        variants = {}

        for specification in args.variants.split(","):
            if "=" not in specification:
                raise ValueError(
                    "Each --variants entry must be "
                    "formatted as name=filename"
                )

            name, filename = specification.split("=", 1)

            name = name.strip()
            filename = filename.strip()

            if not name or not filename:
                raise ValueError(
                    f"Invalid variant specification: "
                    f"{specification!r}"
                )

            variants[name] = filename

    sparse_report = json.loads(
        SPARSE_REPORT.read_text(
            encoding="utf-8"
        )
    )

    lower = np.asarray(
        sparse_report["crop"]["lower"],
        dtype=np.float64,
    )

    upper = np.asarray(
        sparse_report["crop"]["upper"],
        dtype=np.float64,
    )

    selected_names = [
        canonical_name(name)
        for name in sparse_report[
            "selected_images"
        ]
    ]

    gt_properties = read_binary_ply(
        GT_CROP_PLY
    )
    gt_xyz = xyz_from_ply(gt_properties)

    print(
        "Ground-truth evaluation points:",
        f"{len(gt_xyz):,}",
    )
    print("Crop lower:", lower)
    print("Crop upper:", upper)

    gt_tree = cKDTree(gt_xyz)

    ba_reconstruction = pycolmap.Reconstruction(
        str(BA_MODEL)
    )
    gt_reconstruction = pycolmap.Reconstruction(
        str(GT_MODEL)
    )

    ba_images = image_map(
        ba_reconstruction
    )
    gt_images = image_map(
        gt_reconstruction
    )

    common_names = sorted(
        set(selected_names)
        & set(ba_images)
        & set(gt_images)
    )

    if len(common_names) < 3:
        raise RuntimeError(
            f"Only {len(common_names)} common cameras"
        )

    ba_centers = np.asarray(
        [
            camera_center(
                ba_images[name]
            )
            for name in common_names
        ],
        dtype=np.float64,
    )

    gt_centers = np.asarray(
        [
            camera_center(
                gt_images[name]
            )
            for name in common_names
        ],
        dtype=np.float64,
    )

    scale, rotation, translation = estimate_sim3(
        ba_centers,
        gt_centers,
    )

    aligned_centers = transform_xyz(
        ba_centers,
        scale,
        rotation,
        translation,
    )

    camera_errors = np.linalg.norm(
        aligned_centers - gt_centers,
        axis=1,
    )

    print()
    print("BA → ETH3D Sim(3)")
    print("  scale:", scale)
    print(
        "  camera-center RMSE:",
        float(
            np.sqrt(
                np.mean(camera_errors**2)
            )
        ),
    )

    report: dict[str, Any] = {
        "milestone": (
            "Dense Hybrid Fusion 2B evaluation"
        ),
        "description": (
            "Controlled dense-cloud comparison "
            "against ETH3D laser-scan geometry"
        ),
        "evaluation_note": (
            "Diagnostic evaluation using a sampled "
            "laser scan and a common visible-region crop; "
            "not an official ETH3D leaderboard score."
        ),
        "voxel_size_m": args.voxel_size,
        "thresholds_m": thresholds,
        "crop": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "gt_evaluation_points": int(
            len(gt_xyz)
        ),
        "ba_to_eth3d_alignment": {
            "common_cameras": len(common_names),
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "camera_center_rmse_m": float(
                np.sqrt(
                    np.mean(camera_errors**2)
                )
            ),
            "camera_center_median_error_m": float(
                np.median(camera_errors)
            ),
            "camera_center_max_error_m": float(
                np.max(camera_errors)
            ),
        },
        "variants": {},
    }

    for key, filename in variants.items():
        input_path = (
            args.input_dir / filename
        )

        print()
        print("Evaluating:", key)

        properties = read_binary_ply(
            input_path
        )

        xyz_ba = xyz_from_ply(
            properties
        )

        xyz_gt = transform_xyz(
            xyz_ba,
            scale,
            rotation,
            translation,
        )

        inside = crop_mask(
            xyz_gt,
            lower,
            upper,
        )

        xyz_crop = xyz_gt[inside]

        if len(xyz_crop) == 0:
            raise RuntimeError(
                f"{key}: no points inside crop"
            )

        retained_properties = {
            name: values[inside]
            for name, values in properties.items()
            if name not in {"x", "y", "z"}
        }

        voxel_indices = voxel_downsample_indices(
            xyz_crop,
            args.voxel_size,
        )

        xyz_eval = xyz_crop[
            voxel_indices
        ]

        rgb = None
        if all(
            channel in retained_properties
            for channel in [
                "red",
                "green",
                "blue",
            ]
        ):
            rgb = np.stack(
                [
                    retained_properties["red"][
                        voxel_indices
                    ],
                    retained_properties["green"][
                        voxel_indices
                    ],
                    retained_properties["blue"][
                        voxel_indices
                    ],
                ],
                axis=1,
            ).astype(np.uint8)

        confidence = (
            retained_properties["confidence"][
                voxel_indices
            ].astype(np.float32)
            if "confidence" in retained_properties
            else None
        )

        view_ids = (
            retained_properties["view_id"][
                voxel_indices
            ].astype(np.uint8)
            if "view_id" in retained_properties
            else None
        )

        output_ply = (
            args.output_dir
            / f"{key}_eth3d_voxelized.ply"
        )

        write_binary_ply(
            output_ply,
            xyz_eval,
            rgb=rgb,
            confidence=confidence,
            view_ids=view_ids,
        )

        prediction_to_gt = gt_tree.query(
            xyz_eval,
            k=1,
            workers=-1,
        )[0]

        prediction_tree = cKDTree(
            xyz_eval
        )

        gt_to_prediction = prediction_tree.query(
            gt_xyz,
            k=1,
            workers=-1,
        )[0]

        report["variants"][key] = {
            "input_file": str(input_path),
            "output_file": str(output_ply),
            "source_points": int(
                len(xyz_ba)
            ),
            "fraction_inside_crop": float(
                np.mean(inside)
            ),
            "points_inside_crop": int(
                len(xyz_crop)
            ),
            "voxelized_points": int(
                len(xyz_eval)
            ),
            "accuracy_prediction_to_gt": (
                distance_statistics(
                    prediction_to_gt
                )
            ),
            "completeness_gt_to_prediction": (
                distance_statistics(
                    gt_to_prediction
                )
            ),
            "threshold_metrics": threshold_metrics(
                prediction_to_gt,
                gt_to_prediction,
                thresholds,
            ),
        }

        metrics_5cm = report["variants"][key][
            "threshold_metrics"
        ]["0.050_m"]

        metrics_20cm = report["variants"][key][
            "threshold_metrics"
        ]["0.200_m"]

        print(
            "  source points:",
            f"{len(xyz_ba):,}",
        )
        print(
            "  voxelized points:",
            f"{len(xyz_eval):,}",
        )
        print(
            "  median accuracy:",
            report["variants"][key][
                "accuracy_prediction_to_gt"
            ]["median_m"],
        )
        print(
            "  precision @ 5 cm:",
            metrics_5cm["precision"],
        )
        print(
            "  recall @ 20 cm:",
            metrics_20cm["recall"],
        )
        print(
            "  F1 @ 20 cm:",
            metrics_20cm["f1"],
        )

    report_path = (
        args.output_dir
        / "dense_variant_evaluation.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
