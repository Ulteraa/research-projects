#!/usr/bin/env python3
"""Build a gated metric SDF from train-only calibrated VGGT evidence.

Unlike v0.7.5, this command never interprets occluded/behind-depth voxels as
occupied.  It reconstructs a closed surface from oriented observations with
screened Poisson and evaluates the signed distance to that exact mesh.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.data import DTUScene
from rootsplat.poisson_sdf import (
    depth_maps_to_oriented_points, mesh_to_sdf_grid,
    screened_poisson_mesh, voxel_downsample_oriented,
    write_oriented_evidence)
from rootsplat.tsdf import (
    check, cross_view_filter, depth_map_points, load_vggt_alignment,
    mesh_evidence_gate, reproject_point_map, save_sdf_grid,
    sdf_metric_diagnostics, sha256_file)
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
                     alpha=.35, rasterized=True, label="Poisson/SDF surface")
        axis.set_xlabel(aname)
        axis.set_ylabel(bname)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)
        axis.grid(alpha=.15)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle(
        "Train-only VGGT depth evidence versus screened-Poisson SDF surface")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--initializer-gate", required=True)
    parser.add_argument("--output", required=True,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--confidence-quantile", type=float, default=.5)
    parser.add_argument("--cross-view-neighbors", type=int, default=4)
    parser.add_argument("--cross-view-distance", type=float, default=.03)
    parser.add_argument("--cross-view-min-support", type=int, default=1)
    parser.add_argument("--normal-edge-length", type=float, default=.04)
    parser.add_argument("--oriented-voxel-size", type=float, default=.005)
    parser.add_argument("--max-oriented-points", type=int, default=250_000)
    parser.add_argument("--poisson-depth", type=int, default=9)
    parser.add_argument("--poisson-scale", type=float, default=1.02)
    parser.add_argument("--poisson-linear-fit", action="store_true")
    parser.add_argument("--poisson-threads", type=int, default=-1)
    parser.add_argument("--sign-samples", type=int, default=3)
    parser.add_argument("--point-to-surface-p95-max", type=float, default=.04)
    parser.add_argument("--surface-to-point-p95-max", type=float, default=.08)
    parser.add_argument("--unsupported-distance", type=float, default=.05)
    parser.add_argument("--unsupported-fraction-max", type=float, default=.10)
    parser.add_argument("--free-space-violation-max", type=float, default=.02)
    parser.add_argument("--max-components", type=int, default=8)
    parser.add_argument("--min-largest-component-fraction", type=float,
                        default=.90)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Poisson-SDF experiment: {output}")
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
        raise RuntimeError("Fewer than four accepted VGGT training views remain")

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
            depth_blocks.append(depth)
            weight_blocks.append(weight)
            raster_audits.append(audit)
            cameras.append(view.camera)
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

    oriented_points, oriented_normals, oriented_weights, _supports, _views, \
        normal_audit = depth_maps_to_oriented_points(
            depth_maps, weight_maps, cameras, support_maps=support_maps,
            max_edge_length=float(args.normal_edge_length))
    poisson_points, poisson_normals, _poisson_weights, downsample_audit = \
        voxel_downsample_oriented(
            oriented_points, oriented_normals, oriented_weights,
            voxel_size=float(args.oriented_voxel_size), bound=float(args.bound),
            max_points=int(args.max_oriented_points), seed=int(args.seed))
    oriented_path = output / "oriented_evidence.ply"
    write_oriented_evidence(oriented_path, poisson_points, poisson_normals)
    mesh, poisson_audit = screened_poisson_mesh(
        poisson_points, poisson_normals, depth=int(args.poisson_depth),
        scale=float(args.poisson_scale), linear_fit=bool(args.poisson_linear_fit),
        threads=int(args.poisson_threads))
    mesh_path = output / "surface_normalized.ply"
    mesh.export(mesh_path)
    sdf, grid_audit = mesh_to_sdf_grid(
        mesh, resolution=int(args.resolution), bound=float(args.bound),
        sign_samples=int(args.sign_samples))
    metric_audit = sdf_metric_diagnostics(
        sdf, 2.0 * float(args.bound) / int(args.resolution))
    evidence = depth_map_points(depth_maps, cameras, seed=int(args.seed))
    evidence_gate = mesh_evidence_gate(
        mesh, evidence, cameras, depth_maps,
        truncation=4.0 * grid_audit["voxel_size"], seed=int(args.seed),
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

    report_path = output / "poisson_sdf_gate.json"
    sdf_path = output / "base_sdf.npz"
    sdf_sha256 = save_sdf_grid(
        sdf_path, sdf, bound=float(args.bound), report_path=report_path)
    structural_checks = dict(
        accepted_depth_pixels=check(
            int(np.count_nonzero(depth_maps)), ">=", 10_000),
        cross_view_retained_fraction=check(
            cross_audit["retained_fraction"], ">=", .10),
        oriented_surface_points=check(
            int(len(poisson_points)), ">=", 10_000),
        heldout_geometry_views=check(
            int(len(set(view_ids.tolist()) - train_ids)), "==", 0),
        watertight_poisson_surface=check(
            bool(poisson_audit["watertight"]), "==", True),
        positive_volume_boundary=check(
            bool(grid_audit["boundary_positive"]), "==", True),
        interior_fraction_min=check(
            grid_audit["inside_fraction"], ">=", .001),
        interior_fraction_max=check(
            grid_audit["inside_fraction"], "<=", .25),
        eikonal_abs_mean=check(metric_audit["mean"], "<=", .25),
        eikonal_abs_p95=check(metric_audit["p95"], "<=", .75),
        mesh_faces=check(int(len(mesh.faces)), ">=", 1_000))
    failures = [name for name, value in structural_checks.items()
                if not value["passed"]]
    failures.extend([f"evidence.{name}" for name in evidence_gate["failures"]])
    report = dict(
        schema="rootsplat.vggt_poisson_sdf_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        scene=str(Path(args.scene).resolve()), raw=str(Path(args.raw).resolve()),
        initializer_gate=str(Path(args.initializer_gate).resolve()),
        initializer_surface_sha256=initializer_report.get("surface_sha256"),
        protocol="train_only", accepted_view_ids=view_ids.tolist(),
        excluded_heldout_view_ids=heldout_view_ids.tolist(),
        train_alignment=train_alignment_audit,
        train_similarity=dict(
            scale=float(similarity.scale), rotation=similarity.rotation.tolist(),
            translation=similarity.translation.tolist()),
        rasterization=raster_audits, cross_view=cross_audit,
        normal_estimation=normal_audit, downsampling=downsample_audit,
        poisson=poisson_audit, grid_sdf=grid_audit,
        metric_sdf=metric_audit, evidence_gate=evidence_gate,
        checks=structural_checks,
        sdf=str(sdf_path.resolve()), sdf_sha256=sdf_sha256,
        mesh=str(mesh_path.resolve()), mesh_sha256=sha256_file(mesh_path),
        oriented_evidence=str(oriented_path.resolve()),
        oriented_evidence_sha256=sha256_file(oriented_path),
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
