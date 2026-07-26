from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DENSE_DIR = (
    ROOT
    / "outputs/dense_hybrid/vggt_dense_predictions"
)

DEFAULT_CAMERA_FILE = (
    ROOT
    / "outputs/dense_hybrid/ba_refined"
    / "cameras_ba_refined.npz"
)

DEFAULT_ALIGNMENT_REPORT = (
    ROOT
    / "results/milestone2_dense_hybrid"
    / "dense_variant_evaluation.json"
)

DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs/dense_hybrid"
    / "multiview_consistency"
)


def canonical_name(name: str) -> str:
    return re.sub(
        r"^\d+_",
        "",
        Path(name).name,
    )


def camera_center(
    extrinsic: np.ndarray,
) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    return -rotation.T @ translation


def unproject(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    stride: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    height, width = depth.shape

    x_values = np.arange(
        0,
        width,
        stride,
        dtype=np.int32,
    )
    y_values = np.arange(
        0,
        height,
        stride,
        dtype=np.int32,
    )

    pixel_x, pixel_y = np.meshgrid(
        x_values,
        y_values,
    )

    pixel_x = pixel_x.reshape(-1)
    pixel_y = pixel_y.reshape(-1)

    z = depth[pixel_y, pixel_x]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    camera_xyz = np.stack(
        [
            (pixel_x - cx) / fx * z,
            (pixel_y - cy) / fy * z,
            z,
        ],
        axis=1,
    )

    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]

    center = camera_center(extrinsic)

    world_xyz = (
        camera_xyz @ rotation
        + center.reshape(1, 3)
    )

    return world_xyz, pixel_x, pixel_y


def project(
    world_xyz: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]

    camera_xyz = (
        world_xyz @ rotation.T
        + translation.reshape(1, 3)
    )

    z = camera_xyz[:, 2]

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])

    safe_z = np.where(
        np.abs(z) > 1e-12,
        z,
        1.0,
    )

    pixel_x = (
        fx * camera_xyz[:, 0] / safe_z
        + cx
    )

    pixel_y = (
        fy * camera_xyz[:, 1] / safe_z
        + cy
    )

    return pixel_x, pixel_y, z


def bilinear_sample(
    array: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = array.shape

    x0 = np.floor(pixel_x).astype(np.int64)
    y0 = np.floor(pixel_y).astype(np.int64)

    x1 = x0 + 1
    y1 = y0 + 1

    valid = (
        np.isfinite(pixel_x)
        & np.isfinite(pixel_y)
        & (x0 >= 0)
        & (y0 >= 0)
        & (x1 < width)
        & (y1 < height)
    )

    output = np.full(
        len(pixel_x),
        np.nan,
        dtype=np.float64,
    )

    if not np.any(valid):
        return output, valid

    xv = pixel_x[valid]
    yv = pixel_y[valid]

    x0v = x0[valid]
    y0v = y0[valid]
    x1v = x1[valid]
    y1v = y1[valid]

    dx = xv - x0v
    dy = yv - y0v

    value_00 = array[y0v, x0v]
    value_10 = array[y0v, x1v]
    value_01 = array[y1v, x0v]
    value_11 = array[y1v, x1v]

    output[valid] = (
        value_00 * (1.0 - dx) * (1.0 - dy)
        + value_10 * dx * (1.0 - dy)
        + value_01 * (1.0 - dx) * dy
        + value_11 * dx * dy
    )

    valid &= np.isfinite(output)

    return output, valid


def write_supported_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    visible_views: np.ndarray,
    source_view: np.ndarray,
) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<f4"),
            ("support_count", "u1"),
            ("visible_view_count", "u1"),
            ("source_view", "u1"),
        ]
    )

    vertices = np.empty(
        len(xyz),
        dtype=dtype,
    )

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    vertices["confidence"] = confidence
    vertices["support_count"] = support
    vertices["visible_view_count"] = visible_views
    vertices["source_view"] = source_view

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
        "property uchar support_count\n"
        "property uchar visible_view_count\n"
        "property uchar source_view\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def write_fused_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    support_count: np.ndarray,
    view_count: np.ndarray,
    observation_count: np.ndarray,
) -> None:
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<f4"),
            ("support_count", "u1"),
            ("view_count", "u1"),
            ("observation_count", "<u2"),
        ]
    )

    vertices = np.empty(
        len(xyz),
        dtype=dtype,
    )

    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]

    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    vertices["confidence"] = confidence
    vertices["support_count"] = support_count
    vertices["view_count"] = view_count
    vertices["observation_count"] = observation_count

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
        "property uchar support_count\n"
        "property uchar view_count\n"
        "property ushort observation_count\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def fuse_voxels(
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    view_ids: np.ndarray,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    voxel_keys = np.floor(
        xyz / voxel_size
    ).astype(np.int64)

    order = np.lexsort(
        (
            voxel_keys[:, 2],
            voxel_keys[:, 1],
            voxel_keys[:, 0],
        )
    )

    sorted_keys = voxel_keys[order]
    sorted_xyz = xyz[order]
    sorted_rgb = rgb[order].astype(np.float64)
    sorted_confidence = confidence[order]
    sorted_support = support[order]
    sorted_view_ids = view_ids[order]

    boundaries = np.any(
        np.diff(
            sorted_keys,
            axis=0,
        ) != 0,
        axis=1,
    )

    starts = np.concatenate(
        [
            np.asarray([0]),
            np.flatnonzero(boundaries) + 1,
        ]
    )

    ends = np.concatenate(
        [
            starts[1:],
            np.asarray([len(sorted_xyz)]),
        ]
    )

    observation_count = (
        ends - starts
    ).astype(np.uint16)

    log_confidence = np.log1p(
        np.maximum(
            sorted_confidence,
            0.0,
        )
    )

    positive = log_confidence[
        log_confidence > 0
    ]

    reference = (
        float(np.median(positive))
        if len(positive) > 0
        else 1.0
    )

    confidence_weight = np.clip(
        log_confidence / max(reference, 1e-12),
        0.25,
        4.0,
    )

    weights = (
        confidence_weight
        * (1.0 + sorted_support)
    )

    weight_sum = np.add.reduceat(
        weights,
        starts,
    )

    fused_xyz = (
        np.add.reduceat(
            sorted_xyz * weights[:, None],
            starts,
        )
        / weight_sum[:, None]
    )

    fused_rgb = (
        np.add.reduceat(
            sorted_rgb * weights[:, None],
            starts,
        )
        / weight_sum[:, None]
    )

    fused_rgb = np.clip(
        np.rint(fused_rgb),
        0,
        255,
    ).astype(np.uint8)

    fused_confidence = (
        np.add.reduceat(
            sorted_confidence * weights,
            starts,
        )
        / weight_sum
    ).astype(np.float32)

    fused_support = np.maximum.reduceat(
        sorted_support,
        starts,
    ).astype(np.uint8)

    view_bits = (
        np.left_shift(
            np.uint16(1),
            sorted_view_ids.astype(np.uint16),
        )
    )

    fused_view_bits = np.bitwise_or.reduceat(
        view_bits,
        starts,
    )

    fused_view_count = np.asarray(
        [
            int(value).bit_count()
            for value in fused_view_bits
        ],
        dtype=np.uint8,
    )

    return {
        "xyz": fused_xyz,
        "rgb": fused_rgb,
        "confidence": fused_confidence,
        "support_count": fused_support,
        "view_count": fused_view_count,
        "observation_count": observation_count,
    }


def histogram(
    values: np.ndarray,
    maximum: int,
) -> dict[str, int]:
    counts = np.bincount(
        values.astype(np.int64),
        minlength=maximum + 1,
    )

    return {
        str(index): int(counts[index])
        for index in range(maximum + 1)
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dense_dir",
        type=Path,
        default=DEFAULT_DENSE_DIR,
    )
    parser.add_argument(
        "--camera_file",
        type=Path,
        default=DEFAULT_CAMERA_FILE,
    )
    parser.add_argument(
        "--alignment_report",
        type=Path,
        default=DEFAULT_ALIGNMENT_REPORT,
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
    )
    parser.add_argument(
        "--absolute_threshold_m",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--relative_threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--voxel_size_m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--minimum_supports",
        default="1,2,3",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    minimum_supports = [
        int(value)
        for value in args.minimum_supports.split(",")
    ]

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

    camera_data = np.load(
        args.camera_file
    )

    names = [
        canonical_name(str(name))
        for name in camera_data["image_names"]
    ]

    extrinsics = np.asarray(
        camera_data[
            "ba_extrinsics_camera_from_world"
        ],
        dtype=np.float64,
    )

    intrinsics = np.asarray(
        camera_data["vggt_intrinsics"],
        dtype=np.float64,
    )

    depth_scale = float(
        np.asarray(
            camera_data[
                "depth_scale_dense_to_ba"
            ]
        ).reshape(-1)[0]
    )

    alignment_report = json.loads(
        args.alignment_report.read_text(
            encoding="utf-8"
        )
    )

    ba_to_metric_scale = float(
        alignment_report[
            "ba_to_eth3d_alignment"
        ]["scale"]
    )

    voxel_size_ba = (
        args.voxel_size_m
        / ba_to_metric_scale
    )

    print("Views:", len(names))
    print("Depth scale VGGT → BA:", depth_scale)
    print("BA → metric scale:", ba_to_metric_scale)
    print("Metric voxel size:", args.voxel_size_m)
    print("BA voxel size:", voxel_size_ba)

    depths = []
    confidences = []
    rgb_images = []

    for name in names:
        stem = Path(name).stem

        depth = np.load(
            args.dense_dir
            / "depth"
            / f"{stem}.npy"
        ).astype(np.float64)

        confidence = np.load(
            args.dense_dir
            / "depth_confidence"
            / f"{stem}.npy"
        ).astype(np.float64)

        rgb = np.asarray(
            Image.open(
                args.dense_dir
                / "processed_rgb"
                / f"{stem}.png"
            ).convert("RGB"),
            dtype=np.uint8,
        )

        depths.append(
            depth * depth_scale
        )
        confidences.append(confidence)
        rgb_images.append(rgb)

    all_xyz = []
    all_rgb = []
    all_confidence = []
    all_support = []
    all_visible_views = []
    all_source_views = []

    per_view_report = []

    number_of_views = len(names)

    for source_index, source_name in enumerate(names):
        source_depth = depths[source_index]

        world_xyz, pixel_x, pixel_y = unproject(
            source_depth,
            intrinsics[source_index],
            extrinsics[source_index],
            args.stride,
        )

        source_confidence = confidences[
            source_index
        ][pixel_y, pixel_x]

        source_rgb = rgb_images[
            source_index
        ][pixel_y, pixel_x]

        source_z = source_depth[
            pixel_y,
            pixel_x,
        ]

        valid_source = (
            np.isfinite(source_z)
            & (source_z > 0)
            & np.isfinite(source_confidence)
            & np.isfinite(world_xyz).all(axis=1)
        )

        world_xyz = world_xyz[valid_source]
        source_confidence = source_confidence[
            valid_source
        ]
        source_rgb = source_rgb[valid_source]

        support_count = np.zeros(
            len(world_xyz),
            dtype=np.uint8,
        )

        visible_view_count = np.zeros(
            len(world_xyz),
            dtype=np.uint8,
        )

        for target_index in range(number_of_views):
            if target_index == source_index:
                continue

            projected_x, projected_y, projected_z = (
                project(
                    world_xyz,
                    intrinsics[target_index],
                    extrinsics[target_index],
                )
            )

            sampled_depth, valid_depth = (
                bilinear_sample(
                    depths[target_index],
                    projected_x,
                    projected_y,
                )
            )

            sampled_confidence, valid_confidence = (
                bilinear_sample(
                    confidences[target_index],
                    projected_x,
                    projected_y,
                )
            )

            visible = (
                valid_depth
                & valid_confidence
                & np.isfinite(projected_z)
                & (projected_z > 0)
                & np.isfinite(sampled_depth)
                & (sampled_depth > 0)
                & np.isfinite(sampled_confidence)
            )

            visible_view_count += visible.astype(
                np.uint8
            )

            depth_difference_m = (
                np.abs(
                    projected_z
                    - sampled_depth
                )
                * ba_to_metric_scale
            )

            target_depth_m = (
                sampled_depth
                * ba_to_metric_scale
            )

            allowed_difference_m = (
                args.absolute_threshold_m
                + args.relative_threshold
                * target_depth_m
            )

            consistent = (
                visible
                & (
                    depth_difference_m
                    <= allowed_difference_m
                )
            )

            support_count += consistent.astype(
                np.uint8
            )

        all_xyz.append(world_xyz)
        all_rgb.append(source_rgb)
        all_confidence.append(source_confidence)
        all_support.append(support_count)
        all_visible_views.append(visible_view_count)

        source_views = np.full(
            len(world_xyz),
            source_index,
            dtype=np.uint8,
        )

        all_source_views.append(source_views)

        per_view_report.append(
            {
                "image_name": source_name,
                "candidate_points": int(
                    len(world_xyz)
                ),
                "support_histogram": histogram(
                    support_count,
                    number_of_views - 1,
                ),
                "visible_view_histogram": histogram(
                    visible_view_count,
                    number_of_views - 1,
                ),
                "mean_support": float(
                    np.mean(support_count)
                ),
                "median_support": float(
                    np.median(support_count)
                ),
            }
        )

        print(
            f"{source_name}: "
            f"{len(world_xyz):,} candidates, "
            f"mean support "
            f"{np.mean(support_count):.2f}, "
            f"support >= 2: "
            f"{np.mean(support_count >= 2):.3f}"
        )

    xyz = np.concatenate(
        all_xyz,
        axis=0,
    )

    rgb = np.concatenate(
        all_rgb,
        axis=0,
    )

    confidence = np.concatenate(
        all_confidence,
        axis=0,
    )

    support = np.concatenate(
        all_support,
        axis=0,
    )

    visible_views = np.concatenate(
        all_visible_views,
        axis=0,
    )

    source_views = np.concatenate(
        all_source_views,
        axis=0,
    )

    write_supported_ply(
        args.output_dir
        / "ba_vggt_k_with_support.ply",
        xyz,
        rgb,
        confidence,
        support,
        visible_views,
        source_views,
    )

    variants = {}

    for minimum_support in minimum_supports:
        keep = support >= minimum_support

        selected_xyz = xyz[keep]
        selected_rgb = rgb[keep]
        selected_confidence = confidence[keep]
        selected_support = support[keep]
        selected_views = source_views[keep]

        fused = fuse_voxels(
            selected_xyz,
            selected_rgb,
            selected_confidence,
            selected_support,
            selected_views,
            voxel_size_ba,
        )

        output_path = (
            args.output_dir
            / (
                f"fused_support_ge_"
                f"{minimum_support}.ply"
            )
        )

        write_fused_ply(
            output_path,
            fused["xyz"],
            fused["rgb"],
            fused["confidence"],
            fused["support_count"],
            fused["view_count"],
            fused["observation_count"],
        )

        variants[str(minimum_support)] = {
            "minimum_support": minimum_support,
            "selected_observations": int(
                len(selected_xyz)
            ),
            "selected_fraction": float(
                np.mean(keep)
            ),
            "fused_voxels": int(
                len(fused["xyz"])
            ),
            "mean_fused_view_count": float(
                np.mean(
                    fused["view_count"]
                )
            ),
            "median_fused_view_count": float(
                np.median(
                    fused["view_count"]
                )
            ),
            "output_file": str(output_path),
        }

        print()
        print(
            f"Support >= {minimum_support}:"
        )
        print(
            "  selected observations:",
            f"{len(selected_xyz):,}",
        )
        print(
            "  fused voxels:",
            f"{len(fused['xyz']):,}",
        )

    report = {
        "milestone": (
            "Dense Hybrid Fusion 2C "
            "multiview consistency"
        ),
        "input_variant": (
            "BA poses with VGGT intrinsics"
        ),
        "view_count": number_of_views,
        "sampling_stride": args.stride,
        "depth_scale_vggt_to_ba": depth_scale,
        "ba_to_metric_scale": ba_to_metric_scale,
        "absolute_threshold_m": (
            args.absolute_threshold_m
        ),
        "relative_threshold": (
            args.relative_threshold
        ),
        "voxel_size_m": args.voxel_size_m,
        "voxel_size_ba": voxel_size_ba,
        "candidate_points": int(len(xyz)),
        "overall_support_histogram": histogram(
            support,
            number_of_views - 1,
        ),
        "overall_visible_view_histogram": histogram(
            visible_views,
            number_of_views - 1,
        ),
        "mean_support": float(
            np.mean(support)
        ),
        "median_support": float(
            np.median(support)
        ),
        "variants": variants,
        "views": per_view_report,
    }

    report_path = (
        args.output_dir / "metadata.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Overall support histogram:")
    print(report["overall_support_histogram"])
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
