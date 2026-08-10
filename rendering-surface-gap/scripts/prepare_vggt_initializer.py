#!/usr/bin/env python3
"""Align, certify, and export a frozen VGGT surface for RootSplat.

This stage is deliberately independent of RootSplat optimization.  A surface
is accepted only when its predicted cameras agree with the calibrated scene,
its samples have independent-view geometric support, and calibrated masks
support the aligned geometry.  The JSON report is cryptographically bound to
the exact PLY consumed by training.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from rootsplat import DTUScene
from rootsplat.initialization import (orient_normals_to_cameras,
                                      surface_mask_support)
from rootsplat.vggt_initializer import (
    camera_centers_from_extrinsics, filter_cross_view_support,
    point_map_normals, robust_camera_alignment, select_confident_points,
    write_oriented_ply)


def _apply_selection(pool, selection):
    selection = np.asarray(selection)
    return {key: value[selection] for key, value in pool.items()}


def _check(value, relation, threshold):
    value, threshold = float(value), float(threshold)
    if relation == ">=":
        passed = value >= threshold
    elif relation == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported gate relation: {relation}")
    return dict(value=value, relation=relation, threshold=threshold,
                passed=bool(passed))


def _preview(path, points, colors, camera_centers, max_points=60_000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points, colors = np.asarray(points), np.asarray(colors)
    if len(points) > int(max_points):
        ids = np.random.default_rng(int(seed)).choice(
            len(points), int(max_points), replace=False)
        points, colors = points[ids], colors[ids]
    if not np.issubdtype(colors.dtype, np.floating):
        colors = colors.astype(np.float32) / 255.0
    pairs = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, (first, second, xlabel, ylabel) in zip(axes, pairs):
        axis.scatter(points[:, first], points[:, second], s=.12, c=colors,
                     alpha=.65, rasterized=True)
        axis.scatter(camera_centers[:, first], camera_centers[:, second],
                     s=14, marker="^", c="#ff2b2b", edgecolors="black",
                     linewidths=.25, label="known cameras")
        axis.set(xlabel=xlabel, ylabel=ylabel, aspect="equal")
        axis.grid(alpha=.15)
    axes[0].legend(loc="best", fontsize=7)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Raw export_vggt_dtu NPZ")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--preview")
    parser.add_argument("--mask-downscale", type=float, default=.25)
    parser.add_argument("--confidence-quantile", type=float, default=.5)
    parser.add_argument("--max-per-view", type=int, default=100_000)
    parser.add_argument("--alignment-threshold", type=float, default=.08)
    parser.add_argument("--alignment-trials", type=int, default=512)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--support-distance", type=float, default=.02)
    parser.add_argument("--min-support-views", type=int, default=1)
    parser.add_argument("--max-normal-angle", type=float, default=60.0)
    parser.add_argument("--mask-fraction", type=float, default=.8)
    parser.add_argument("--mask-min-views", type=int, default=4)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--max-final-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        raise FileNotFoundError(f"VGGT raw export does not exist: {raw_path}")
    required = {"view_ids", "world_points", "world_points_conf", "images",
                "extrinsic", "intrinsic", "depth"}
    with np.load(raw_path) as raw:
        missing = required.difference(raw.files)
        if missing:
            raise KeyError("VGGT raw export misses: " + ", ".join(sorted(missing)))
        view_ids = raw["view_ids"].astype(np.int64)
        point_maps = raw["world_points"].astype(np.float64)
        confidence = raw["world_points_conf"].astype(np.float32)
        colors = raw["images"]
        extrinsic = raw["extrinsic"].astype(np.float64)

    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.mask_downscale),
        test_every=0, require_masks=True, require_scale_matrices=True)
    if view_ids.ndim != 1 or len(view_ids) < 4 or len(np.unique(view_ids)) != len(view_ids):
        raise ValueError("VGGT view_ids must contain at least four unique views")
    if np.any(view_ids < 0) or np.any(view_ids >= len(scene.views)):
        raise ValueError("VGGT view_ids lie outside the calibrated scene")
    if len(point_maps) != len(view_ids) or len(extrinsic) != len(view_ids):
        raise ValueError("VGGT arrays and view_ids are misaligned")

    raw_view_ids = view_ids.copy()
    known_centers = np.stack([
        scene.view_cameras[int(index)].center.detach().cpu().numpy()
        for index in view_ids], axis=0)
    predicted_centers = camera_centers_from_extrinsics(extrinsic)
    similarity, alignment_audit = robust_camera_alignment(
        predicted_centers, known_centers,
        threshold_fraction=float(args.alignment_threshold),
        trials=int(args.alignment_trials), seed=int(args.seed))
    aligned_view_keep = np.asarray(alignment_audit["inlier_mask"], dtype=bool)
    view_ids = view_ids[aligned_view_keep]
    known_centers = known_centers[aligned_view_keep]
    point_maps = point_maps[aligned_view_keep]
    confidence = confidence[aligned_view_keep]
    colors = colors[aligned_view_keep]
    aligned_maps = similarity.apply_points(point_maps)
    normal_maps, valid_normals = point_map_normals(aligned_maps)
    pool, confidence_audit = select_confident_points(
        aligned_maps, confidence, colors, normal_maps, valid_normals,
        confidence_quantile=float(args.confidence_quantile),
        max_per_view=int(args.max_per_view), seed=int(args.seed))

    cross_keep, support, cross_view_audit = filter_cross_view_support(
        pool["points"], pool["normals"], pool["view_slot"], known_centers,
        neighbor_count=int(args.neighbor_count),
        distance=float(args.support_distance),
        min_support_views=int(args.min_support_views),
        max_normal_angle_degrees=float(args.max_normal_angle))
    pool["support"] = support
    pool = _apply_selection(pool, cross_keep)

    bound_keep = np.max(np.abs(pool["points"]), axis=-1) <= float(args.bound)
    bound_audit = dict(
        input_points=int(len(pool["points"])),
        retained_points=int(bound_keep.sum()),
        retained_fraction=float(bound_keep.mean()) if len(bound_keep) else 0.0,
        bound=float(args.bound))
    pool = _apply_selection(pool, bound_keep)

    mask_keep, mask_audit = surface_mask_support(
        pool["points"], scene.view_cameras,
        [view.mask for view in scene.views],
        min_fraction=float(args.mask_fraction),
        min_views=int(args.mask_min_views))
    pool = _apply_selection(pool, mask_keep)
    pool["normals"] = orient_normals_to_cameras(
        pool["points"], pool["normals"], known_centers)

    if len(pool["points"]) > int(args.max_final_points):
        chosen = np.random.default_rng(int(args.seed)).choice(
            len(pool["points"]), int(args.max_final_points), replace=False)
        pool = _apply_selection(pool, chosen)
    extent = np.ptp(pool["points"], axis=0) if len(pool["points"]) else np.zeros(3)
    output_audit = dict(
        points=int(len(pool["points"])), extent=extent.tolist(),
        bounds=([pool["points"].min(0).tolist(), pool["points"].max(0).tolist()]
                if len(pool["points"]) else None),
        views_contributing=int(len(np.unique(pool["view_slot"]))),
        max_final_points=int(args.max_final_points))

    surface = Path(args.surface)
    surface_sha256 = write_oriented_ply(
        surface, pool["points"], pool["normals"], pool["colors"],
        pool["confidence"], view_ids[pool["view_slot"]], pool["support"])
    preview = Path(args.preview) if args.preview else surface.with_suffix(".png")
    _preview(preview, pool["points"], pool["colors"], known_centers,
             seed=int(args.seed))

    checks = dict(
        alignment_inlier_fraction=_check(
            alignment_audit["inlier_fraction"], ">=", .75),
        alignment_residual_median=_check(
            alignment_audit["residual_normalized_median"], "<=", .05),
        alignment_inlier_residual_p95=_check(
            alignment_audit["residual_normalized_inlier_p95"], "<=", .08),
        rotation_determinant_error=_check(
            abs(alignment_audit["rotation_determinant"] - 1.0), "<=", 1e-5),
        cross_view_points=_check(cross_view_audit["retained_points"], ">=", 10_000),
        cross_view_fraction=_check(
            cross_view_audit["retained_fraction"], ">=", .02),
        mask_retained_fraction=_check(mask_audit["retained_fraction"], ">=", .10),
        output_points=_check(output_audit["points"], ">=", 10_000),
        contributing_views=_check(output_audit["views_contributing"], ">=", 4),
        minimum_extent=_check(float(np.min(extent)), ">=", .10))
    failures = [name for name, check in checks.items() if not check["passed"]]
    report = dict(
        schema="rootsplat.vggt_initializer_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        raw=str(raw_path.resolve()), scene=str(Path(args.scene).resolve()),
        surface=str(surface.resolve()), surface_sha256=surface_sha256,
        preview=str(preview.resolve()), raw_view_ids=raw_view_ids.tolist(),
        accepted_view_ids=view_ids.tolist(),
        similarity=dict(
            scale=float(similarity.scale),
            rotation=similarity.rotation.tolist(),
            translation=similarity.translation.tolist()),
        alignment=alignment_audit, confidence=confidence_audit,
        cross_view=cross_view_audit, bounds=bound_audit,
        masks=mask_audit, output=output_audit, checks=checks,
        options=vars(args))
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
