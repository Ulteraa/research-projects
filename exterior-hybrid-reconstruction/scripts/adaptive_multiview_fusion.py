from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.multiview_consistency_fusion import (  # noqa: E402
    bilinear_sample,
    fuse_voxels,
    project,
    unproject,
    write_fused_ply,
)


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
    / "outputs/dense_hybrid/adaptive_multiview"
)


def canonical_name(name: str) -> str:
    return re.sub(
        r"^\d+_",
        "",
        Path(name).name,
    )


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


def ratio_name(value: float) -> str:
    return f"{value:.2f}".replace(".", "")


def save_fused_variant(
    output_path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    support: np.ndarray,
    source_views: np.ndarray,
    keep: np.ndarray,
    voxel_size_ba: float,
) -> dict[str, float | int | str]:
    selected_xyz = xyz[keep]
    selected_rgb = rgb[keep]
    selected_confidence = confidence[keep]
    selected_support = support[keep]
    selected_views = source_views[keep]

    if len(selected_xyz) == 0:
        raise RuntimeError(
            f"No points selected for {output_path.name}"
        )

    fused = fuse_voxels(
        selected_xyz,
        selected_rgb,
        selected_confidence,
        selected_support,
        selected_views,
        voxel_size_ba,
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

    return {
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
            np.mean(fused["view_count"])
        ),
        "median_fused_view_count": float(
            np.median(fused["view_count"])
        ),
        "output_file": str(output_path),
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
        "--ratio_thresholds",
        default="0.33,0.50,0.67",
    )
    parser.add_argument(
        "--target_confidence_quantile",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--unobserved_confidence_quantile",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    ratio_thresholds = [
        float(value)
        for value in args.ratio_thresholds.split(",")
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

    depths: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    rgb_images: list[np.ndarray] = []

    target_confidence_thresholds = []
    unobserved_confidence_thresholds = []

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

        valid_confidence = confidence[
            np.isfinite(confidence)
        ]

        target_threshold = float(
            np.quantile(
                valid_confidence,
                args.target_confidence_quantile,
            )
        )

        unobserved_threshold = float(
            np.quantile(
                valid_confidence,
                args.unobserved_confidence_quantile,
            )
        )

        depths.append(
            depth * depth_scale
        )
        confidences.append(confidence)
        rgb_images.append(rgb)

        target_confidence_thresholds.append(
            target_threshold
        )
        unobserved_confidence_thresholds.append(
            unobserved_threshold
        )

    all_xyz = []
    all_rgb = []
    all_confidence = []
    all_support = []
    all_conflict = []
    all_occluded = []
    all_visible = []
    all_source_views = []
    all_unobserved_confident = []

    per_view_report = []

    number_of_views = len(names)

    for source_index, source_name in enumerate(names):
        world_xyz, pixel_x, pixel_y = unproject(
            depths[source_index],
            intrinsics[source_index],
            extrinsics[source_index],
            args.stride,
        )

        source_depth = depths[source_index][
            pixel_y,
            pixel_x,
        ]

        source_confidence = confidences[
            source_index
        ][pixel_y, pixel_x]

        source_rgb = rgb_images[
            source_index
        ][pixel_y, pixel_x]

        valid_source = (
            np.isfinite(source_depth)
            & (source_depth > 0)
            & np.isfinite(source_confidence)
            & np.isfinite(world_xyz).all(axis=1)
        )

        world_xyz = world_xyz[valid_source]
        source_confidence = source_confidence[
            valid_source
        ]
        source_rgb = source_rgb[valid_source]

        point_count = len(world_xyz)

        support_count = np.zeros(
            point_count,
            dtype=np.uint8,
        )

        conflict_count = np.zeros(
            point_count,
            dtype=np.uint8,
        )

        occluded_count = np.zeros(
            point_count,
            dtype=np.uint8,
        )

        visible_count = np.zeros(
            point_count,
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
                & (
                    sampled_confidence
                    >= target_confidence_thresholds[
                        target_index
                    ]
                )
            )

            visible_count += visible.astype(
                np.uint8
            )

            signed_difference_m = (
                projected_z - sampled_depth
            ) * ba_to_metric_scale

            target_depth_m = (
                sampled_depth
                * ba_to_metric_scale
            )

            tolerance_m = (
                args.absolute_threshold_m
                + args.relative_threshold
                * target_depth_m
            )

            consistent = (
                visible
                & (
                    np.abs(signed_difference_m)
                    <= tolerance_m
                )
            )

            # Target camera sees a closer surface.
            # The source point may simply be hidden.
            occluded = (
                visible
                & (
                    signed_difference_m
                    > tolerance_m
                )
            )

            # Source point lies in front of the target
            # depth and should normally be visible.
            conflict = (
                visible
                & (
                    signed_difference_m
                    < -tolerance_m
                )
            )

            support_count += consistent.astype(
                np.uint8
            )

            occluded_count += occluded.astype(
                np.uint8
            )

            conflict_count += conflict.astype(
                np.uint8
            )

        comparable_count = (
            support_count.astype(np.int16)
            + conflict_count.astype(np.int16)
        )

        support_ratio = np.zeros(
            point_count,
            dtype=np.float64,
        )

        tested = comparable_count > 0

        support_ratio[tested] = (
            support_count[tested]
            / comparable_count[tested]
        )

        unobserved_confident = (
            (~tested)
            & (
                source_confidence
                >= unobserved_confidence_thresholds[
                    source_index
                ]
            )
        )

        all_xyz.append(world_xyz)
        all_rgb.append(source_rgb)
        all_confidence.append(source_confidence)
        all_support.append(support_count)
        all_conflict.append(conflict_count)
        all_occluded.append(occluded_count)
        all_visible.append(visible_count)
        all_unobserved_confident.append(
            unobserved_confident
        )

        all_source_views.append(
            np.full(
                point_count,
                source_index,
                dtype=np.uint8,
            )
        )

        per_view_report.append(
            {
                "image_name": source_name,
                "candidate_points": point_count,
                "mean_support": float(
                    np.mean(support_count)
                ),
                "mean_conflict": float(
                    np.mean(conflict_count)
                ),
                "mean_occluded": float(
                    np.mean(occluded_count)
                ),
                "tested_fraction": float(
                    np.mean(tested)
                ),
                "unobserved_confident_fraction": float(
                    np.mean(unobserved_confident)
                ),
            }
        )

        print(
            f"{source_name}: "
            f"support={np.mean(support_count):.2f}, "
            f"conflict={np.mean(conflict_count):.2f}, "
            f"occluded={np.mean(occluded_count):.2f}, "
            f"tested={np.mean(tested):.3f}"
        )

    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    confidence = np.concatenate(all_confidence)
    support = np.concatenate(all_support)
    conflict = np.concatenate(all_conflict)
    occluded = np.concatenate(all_occluded)
    visible = np.concatenate(all_visible)
    source_views = np.concatenate(all_source_views)

    unobserved_confident = np.concatenate(
        all_unobserved_confident
    )

    comparable = (
        support.astype(np.int16)
        + conflict.astype(np.int16)
    )

    tested = comparable > 0

    support_ratio = np.zeros(
        len(xyz),
        dtype=np.float64,
    )

    support_ratio[tested] = (
        support[tested]
        / comparable[tested]
    )

    variants = {}

    # Essential experimental control:
    # confidence-weighted fusion without filtering.
    all_keep = np.ones(
        len(xyz),
        dtype=bool,
    )

    fused_all_path = (
        args.output_dir / "fused_all.ply"
    )

    variants["fused_all"] = save_fused_variant(
        fused_all_path,
        xyz,
        rgb,
        confidence,
        support,
        source_views,
        all_keep,
        voxel_size_ba,
    )

    for ratio_threshold in ratio_thresholds:
        keep_tested = (
            tested
            & (support >= 1)
            & (
                support_ratio
                >= ratio_threshold
            )
        )

        # Preserve high-confidence geometry that cannot
        # be judged because it is outside overlapping views
        # or occluded everywhere else.
        keep = (
            keep_tested
            | unobserved_confident
        )

        key = (
            "adaptive_ratio_"
            + ratio_name(ratio_threshold)
        )

        output_path = (
            args.output_dir / f"{key}.ply"
        )

        variants[key] = save_fused_variant(
            output_path,
            xyz,
            rgb,
            confidence,
            support,
            source_views,
            keep,
            voxel_size_ba,
        )

        variants[key][
            "ratio_threshold"
        ] = ratio_threshold

        variants[key][
            "kept_tested_points"
        ] = int(
            np.sum(keep_tested)
        )

        variants[key][
            "kept_unobserved_confident_points"
        ] = int(
            np.sum(unobserved_confident)
        )

        print()
        print(key)
        print(
            "  selected:",
            f"{np.sum(keep):,}",
        )
        print(
            "  fused:",
            f"{variants[key]['fused_voxels']:,}",
        )

    report = {
        "milestone": (
            "Dense Hybrid Fusion 2D "
            "occlusion-aware adaptive fusion"
        ),
        "candidate_points": int(len(xyz)),
        "depth_scale_vggt_to_ba": depth_scale,
        "ba_to_metric_scale": ba_to_metric_scale,
        "voxel_size_m": args.voxel_size_m,
        "voxel_size_ba": voxel_size_ba,
        "absolute_threshold_m": (
            args.absolute_threshold_m
        ),
        "relative_threshold": (
            args.relative_threshold
        ),
        "target_confidence_quantile": (
            args.target_confidence_quantile
        ),
        "unobserved_confidence_quantile": (
            args.unobserved_confidence_quantile
        ),
        "tested_fraction": float(
            np.mean(tested)
        ),
        "unobserved_confident_fraction": float(
            np.mean(unobserved_confident)
        ),
        "mean_support": float(
            np.mean(support)
        ),
        "mean_conflict": float(
            np.mean(conflict)
        ),
        "mean_occluded": float(
            np.mean(occluded)
        ),
        "support_histogram": histogram(
            support,
            number_of_views - 1,
        ),
        "conflict_histogram": histogram(
            conflict,
            number_of_views - 1,
        ),
        "occlusion_histogram": histogram(
            occluded,
            number_of_views - 1,
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
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
