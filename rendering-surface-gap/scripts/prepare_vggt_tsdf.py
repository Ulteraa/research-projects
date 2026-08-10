#!/usr/bin/env python3
"""Fuse calibrated VGGT range images into a gated RootSplat grid SDF.

This command is deliberately independent of RootSplat training.  It must pass
and its mesh must be inspected before creating an optimizer or paying for a
long appearance run.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.data import DTUScene
from rootsplat.tsdf import (
    check, compute_visual_hull, cross_view_filter, depth_map_points,
    extract_sdf_mesh, fuse_weighted_tsdf, load_vggt_alignment,
    mesh_evidence_gate, reinitialize_signed_distance, reproject_point_map,
    save_sdf_grid, sdf_metric_diagnostics, sha256_file)
from rootsplat.vggt_initializer import (
    camera_centers_from_extrinsics, robust_camera_alignment)


def preview(path, mesh, evidence, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    rng = np.random.default_rng(int(seed))
    mesh_points, _ = trimesh.sample.sample_surface(
        mesh, min(100_000, max(20_000, len(mesh.faces) * 2)), seed=rng)
    if len(evidence) > 100_000:
        evidence = evidence[rng.choice(len(evidence), 100_000, replace=False)]
    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (a, b, aname, bname) in zip(axes, planes):
        axis.scatter(evidence[:, a], evidence[:, b], s=.12, c="#e66101",
                     alpha=.30, rasterized=True, label="accepted VGGT depth")
        axis.scatter(mesh_points[:, a], mesh_points[:, b], s=.12, c="#0571b0",
                     alpha=.35, rasterized=True, label="TSDF/SDF surface")
        axis.set_xlabel(aname); axis.set_ylabel(bname)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1, 1); axis.set_ylim(-1, 1)
        axis.grid(alpha=.15)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle("VGGT depth evidence versus reconstructed SDF surface")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--raw", required=True,
                        help="raw_predictions.npz from export_vggt_dtu.py")
    parser.add_argument("--initializer-gate", required=True)
    parser.add_argument("--output", required=True,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--confidence-quantile", type=float, default=.5)
    parser.add_argument("--cross-view-neighbors", type=int, default=4)
    parser.add_argument("--cross-view-distance", type=float, default=.03)
    parser.add_argument("--cross-view-min-support", type=int, default=1)
    parser.add_argument("--truncation-voxels", type=float, default=4.0)
    parser.add_argument("--visual-hull-fraction", type=float, default=.8)
    parser.add_argument("--visual-hull-min-views", type=int, default=4)
    parser.add_argument("--min-fusion-views", type=int, default=2)
    parser.add_argument("--min-behind-views", type=int, default=1)
    parser.add_argument("--free-view-threshold", type=int, default=2)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--point-to-surface-p95-max", type=float, default=.04)
    parser.add_argument("--surface-to-point-p95-max", type=float, default=.08)
    parser.add_argument("--unsupported-distance", type=float, default=.05)
    parser.add_argument("--unsupported-fraction-max", type=float, default=.10)
    parser.add_argument("--free-space-violation-max", type=float, default=.02)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--min-largest-component-fraction", type=float, default=.90)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing TSDF experiment: {output}")
    output.mkdir(parents=True)
    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    raw, gate_view_ids, gate_slots, _gate_similarity, initializer_report = \
        load_vggt_alignment(args.raw, args.initializer_gate)
    train_ids = set(int(value) for value in scene.train_ids)
    train_selection = np.asarray(
        [int(view_id) in train_ids for view_id in gate_view_ids], dtype=bool)
    view_ids = gate_view_ids[train_selection]
    slots = gate_slots[train_selection]
    heldout_view_ids = gate_view_ids[~train_selection]
    if len(view_ids) < 4:
        raw.close()
        raise RuntimeError(
            "Fewer than four accepted VGGT training views remain after "
            "held-out views are removed")
    # Re-estimate Sim(3) without using a held-out image or its VGGT pose.  The
    # original initializer report is an integrity prerequisite, not the final
    # alignment consumed by this evaluation-safe reconstruction.
    predicted_centers = camera_centers_from_extrinsics(raw["extrinsic"][slots])
    known_centers = np.stack([
        scene.view_cameras[int(view_id)].center.detach().cpu().numpy()
        for view_id in view_ids], axis=0)
    similarity, train_alignment_audit = robust_camera_alignment(
        predicted_centers, known_centers,
        threshold_fraction=float(initializer_report.get(
            "options", {}).get("alignment_threshold", .08)),
        trials=int(initializer_report.get(
            "options", {}).get("alignment_trials", 512)),
        seed=int(args.seed))
    depth_blocks, weight_blocks, raster_audits, cameras = [], [], [], []
    try:
        raw_points = raw["world_points"]
        raw_confidence = raw["world_points_conf"]
        for view_id, slot in zip(view_ids, slots):
            points = similarity.apply_points(raw_points[int(slot)]).astype(np.float32)
            confidence = raw_confidence[int(slot)]
            view = scene.views[int(view_id)]
            depth, weight, audit = reproject_point_map(
                points, confidence, view.camera, mask=view.mask,
                confidence_quantile=float(args.confidence_quantile),
                bound=float(args.bound))
            audit["view_id"] = int(view_id)
            audit["view_name"] = view.name
            depth_blocks.append(depth); weight_blocks.append(weight)
            raster_audits.append(audit); cameras.append(view.camera)
    finally:
        raw.close()
    depth_maps, weight_maps, support_maps, cross_audit = cross_view_filter(
        np.stack(depth_blocks), np.stack(weight_blocks), cameras,
        neighbors=int(args.cross_view_neighbors),
        tolerance=float(args.cross_view_distance),
        min_support_views=int(args.cross_view_min_support))
    calibrated_depths = output / "calibrated_depths.npz"
    np.savez_compressed(
        calibrated_depths, view_ids=view_ids.astype(np.int32),
        depth=depth_maps.astype(np.float32), weight=weight_maps.astype(np.float32),
        support=support_maps.astype(np.uint8))
    fusion = fuse_weighted_tsdf(
        depth_maps, weight_maps, cameras,
        resolution=int(args.resolution), bound=float(args.bound),
        truncation_voxels=float(args.truncation_voxels),
        device=args.device)
    visual_hull, hull_audit = compute_visual_hull(
        scene.train_cameras, [scene.views[index].mask for index in scene.train_ids],
        resolution=int(args.resolution), bound=float(args.bound),
        min_fraction=float(args.visual_hull_fraction),
        min_views=int(args.visual_hull_min_views), device=args.device)
    sdf, inside, sign_audit = reinitialize_signed_distance(
        fusion, visual_hull,
        min_fusion_views=int(args.min_fusion_views),
        min_behind_views=int(args.min_behind_views),
        free_view_threshold=int(args.free_view_threshold),
        closing_iterations=int(args.closing_iterations))
    metric_audit = sdf_metric_diagnostics(sdf, fusion["voxel_size"])
    mesh = extract_sdf_mesh(sdf, bound=float(args.bound))
    mesh_path = output / "surface_normalized.ply"
    mesh.export(mesh_path)
    evidence = depth_map_points(depth_maps, cameras, seed=int(args.seed))
    evidence_gate = mesh_evidence_gate(
        mesh, evidence, cameras, depth_maps,
        truncation=float(fusion["truncation"]), seed=int(args.seed),
        point_to_surface_p95_max=float(args.point_to_surface_p95_max),
        surface_to_point_p95_max=float(args.surface_to_point_p95_max),
        unsupported_distance=float(args.unsupported_distance),
        unsupported_fraction_max=float(args.unsupported_fraction_max),
        free_space_violation_max=float(args.free_space_violation_max),
        max_components=int(args.max_components),
        min_largest_component_fraction=float(
            args.min_largest_component_fraction))
    preview_path = output / "surface_preview.png"
    preview(preview_path, mesh, evidence, seed=int(args.seed))
    report_path = output / "tsdf_gate.json"
    sdf_path = output / "base_sdf.npz"
    sdf_sha256 = save_sdf_grid(
        sdf_path, sdf, bound=float(args.bound), report_path=report_path)
    structural_checks = dict(
        accepted_depth_pixels=check(
            int(np.count_nonzero(depth_maps)), ">=", 10_000),
        cross_view_retained_fraction=check(
            cross_audit["retained_fraction"], ">=", .10),
        interior_fraction_min=check(
            sign_audit["final_inside_fraction"], ">=", .001),
        interior_fraction_max=check(
            sign_audit["final_inside_fraction"], "<=", .50),
        positive_volume_boundary=check(
            sign_audit["boundary_positive"], "==", True),
        heldout_geometry_views=check(
            int(len(set(view_ids.tolist()) - train_ids)), "==", 0),
        eikonal_abs_mean=check(metric_audit["mean"], "<=", .25),
        eikonal_abs_p95=check(metric_audit["p95"], "<=", .75),
        mesh_faces=check(int(len(mesh.faces)), ">=", 1_000))
    failures = [name for name, value in structural_checks.items()
                if not value["passed"]]
    failures.extend([f"evidence.{name}" for name in evidence_gate["failures"]])
    report = dict(
        schema="rootsplat.vggt_tsdf_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        scene=str(Path(args.scene).resolve()), raw=str(Path(args.raw).resolve()),
        initializer_gate=str(Path(args.initializer_gate).resolve()),
        initializer_surface_sha256=initializer_report.get("surface_sha256"),
        protocol="train_only", accepted_view_ids=view_ids.tolist(),
        excluded_heldout_view_ids=heldout_view_ids.tolist(),
        train_alignment=train_alignment_audit,
        train_similarity=dict(
            scale=float(similarity.scale),
            rotation=similarity.rotation.tolist(),
            translation=similarity.translation.tolist()),
        rasterization=raster_audits,
        cross_view=cross_audit, visual_hull=hull_audit,
        fusion=dict(
            resolution=int(fusion["resolution"]), bound=float(fusion["bound"]),
            voxel_size=float(fusion["voxel_size"]),
            truncation=float(fusion["truncation"]),
            observed_voxels=int(np.count_nonzero(fusion["weight"])),
            observed_fraction=float(np.mean(fusion["weight"] > 0)),
            max_view_count=int(fusion["view_count"].max())),
        sign_reinitialization=sign_audit, metric_sdf=metric_audit,
        evidence_gate=evidence_gate,
        checks=structural_checks,
        sdf=str(sdf_path.resolve()), sdf_sha256=sdf_sha256,
        mesh=str(mesh_path.resolve()), mesh_sha256=sha256_file(mesh_path),
        preview=str(preview_path.resolve()),
        calibrated_depths=str(calibrated_depths.resolve()),
        calibrated_depths_sha256=sha256_file(calibrated_depths),
        mesh_topology=dict(
            vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
            watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            euler_number=int(mesh.euler_number)),
        options=vars(args))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    print(f"Gate report: {report_path}")
    print(f"Visual review: {preview_path}")
    print("DO NOT TRAIN unless status is pass and the PLY visibly matches the scene.")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
