#!/usr/bin/env python3
"""Build a closed metric SDF from strict silhouettes and VGGT depth carving."""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.carved_sdf import (
    compute_silhouette_votes, evidence_tethered_carving,
    ray_solid_thickness_audit)
from rootsplat.data import DTUScene
from rootsplat.tsdf import (
    check, depth_map_points, extract_sdf_mesh, fuse_weighted_tsdf,
    mesh_evidence_gate, save_sdf_grid, sdf_metric_diagnostics, sha256_file)


def preview(path, mesh, evidence, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    rng = np.random.default_rng(int(seed))
    surface, _ = trimesh.sample.sample_surface(
        mesh, min(120_000, max(30_000, len(mesh.faces) * 2)), seed=rng)
    if len(evidence) > 120_000:
        evidence = evidence[rng.choice(len(evidence), 120_000, replace=False)]
    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (a, b, aname, bname) in zip(axes, planes):
        axis.scatter(evidence[:, a], evidence[:, b], s=.12, c="#e66101",
                     alpha=.30, rasterized=True, label="train-only VGGT depth")
        axis.scatter(surface[:, a], surface[:, b], s=.12, c="#0571b0",
                     alpha=.35, rasterized=True, label="carved SDF surface")
        axis.set(xlabel=aname, ylabel=bname, aspect="equal",
                 xlim=(-1, 1), ylim=(-1, 1))
        axis.grid(alpha=.15)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle("VGGT depth evidence versus evidence-tethered carved surface")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--depth-gate", required=True)
    parser.add_argument("--output", required=True,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--truncation-voxels", type=float, default=4.0)
    parser.add_argument("--mask-dilation-pixels", type=int, default=2)
    parser.add_argument("--min-visible-views", type=int, default=16)
    parser.add_argument("--max-background-views", type=int, default=2)
    parser.add_argument("--min-fusion-views", type=int, default=2)
    parser.add_argument("--min-behind-views", type=int, default=3)
    parser.add_argument("--min-total-depth-votes", type=int, default=4)
    parser.add_argument("--min-behind-fraction", type=float, default=.75)
    parser.add_argument("--max-free-views", type=int, default=1)
    parser.add_argument("--closing-iterations", type=int, default=1)
    parser.add_argument("--point-to-surface-p95-max", type=float, default=.04)
    parser.add_argument("--surface-to-point-p95-max", type=float, default=.08)
    parser.add_argument("--unsupported-distance", type=float, default=.05)
    parser.add_argument("--unsupported-fraction-max", type=float, default=.10)
    parser.add_argument("--free-space-violation-max", type=float, default=.02)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--max-genus", type=int, default=32)
    parser.add_argument("--min-ray-thickness", type=float, default=.08)
    parser.add_argument("--max-thin-ray-fraction", type=float, default=.50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite carved-SDF output: {output}")
    output.mkdir(parents=True)
    depths_path, depth_gate_path = Path(args.depths), Path(args.depth_gate)
    depth_report = json.loads(depth_gate_path.read_text(encoding="utf8"))
    if depth_report.get("schema") != "rootsplat.vggt_fulltrain_depth_gate.v1" or \
            depth_report.get("status") != "pass" or depth_report.get("failures"):
        raise RuntimeError("A passing full-training-view VGGT depth gate is required")
    if depth_report.get("output_sha256") != sha256_file(depths_path):
        raise RuntimeError("Full-training-view depth archive digest mismatch")

    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    with np.load(depths_path) as archive:
        required = {"view_ids", "depth", "weight", "support"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError("Calibrated depth archive misses: " +
                           ", ".join(sorted(missing)))
        view_ids = archive["view_ids"].astype(np.int64)
        depth_maps = archive["depth"].astype(np.float32)
        weight_maps = archive["weight"].astype(np.float32)
    expected_train = np.asarray(scene.train_ids, dtype=np.int64)
    heldout = set(int(value) for value in scene.test_ids)
    if not np.array_equal(view_ids, expected_train):
        raise RuntimeError("Carving requires all training views in canonical order")
    if set(view_ids.tolist()) & heldout:
        raise RuntimeError("Held-out views reached the geometry constructor")
    if depth_maps.shape != weight_maps.shape or len(depth_maps) != len(view_ids):
        raise ValueError("Depth, weight and view arrays are misaligned")
    cameras = [scene.view_cameras[int(index)] for index in view_ids]
    masks = [scene.views[int(index)].mask for index in view_ids]

    fusion = fuse_weighted_tsdf(
        depth_maps, weight_maps, cameras,
        resolution=int(args.resolution), bound=float(args.bound),
        truncation_voxels=float(args.truncation_voxels), device=args.device)
    silhouette = compute_silhouette_votes(
        cameras, masks, resolution=int(args.resolution),
        bound=float(args.bound),
        dilation_pixels=int(args.mask_dilation_pixels), device=args.device)
    sdf, inside, carve_audit = evidence_tethered_carving(
        fusion, silhouette,
        min_visible_views=int(args.min_visible_views),
        max_background_views=int(args.max_background_views),
        min_fusion_views=int(args.min_fusion_views),
        min_behind_views=int(args.min_behind_views),
        min_total_depth_votes=int(args.min_total_depth_votes),
        min_behind_fraction=float(args.min_behind_fraction),
        max_free_views=int(args.max_free_views),
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
        min_largest_component_fraction=.95)
    thickness = ray_solid_thickness_audit(
        inside, depth_maps, cameras, bound=float(args.bound),
        seed=int(args.seed))

    components = int(evidence_gate["components"])
    euler = int(mesh.euler_number)
    genus = max(0, int(round((2 * components - euler) / 2))) \
        if mesh.is_watertight and mesh.is_winding_consistent else -1
    area = float(mesh.area)
    volume = float(abs(mesh.volume))
    equivalent_thickness = float(2.0 * volume / max(area, 1e-12))
    checks = dict(
        exact_training_view_coverage=check(
            int(len(view_ids)), "==", int(len(expected_train))),
        heldout_geometry_views=check(
            int(len(set(view_ids.tolist()) & heldout)), "==", 0),
        accepted_depth_pixels=check(
            int(np.count_nonzero(depth_maps)), ">=", 100_000),
        strict_hull_voxels=check(
            int(carve_audit["strict_hull_voxels"]), ">=", 10_000),
        admitted_core_voxels=check(
            int(carve_audit["admitted_core_voxels"]), ">=", 10_000),
        interior_fraction_min=check(float(inside.mean()), ">=", .001),
        interior_fraction_max=check(float(inside.mean()), "<=", .20),
        positive_volume_boundary=check(
            bool(carve_audit["boundary_positive"]), "==", True),
        ray_surface_entry_fraction=check(
            float(thickness["surface_entry_fraction"]), ">=", .90),
        ray_thickness_median=check(
            float(thickness["thickness_median"]), ">=",
            float(args.min_ray_thickness)),
        thin_ray_fraction=check(
            float(thickness["thin_fraction_005"]), "<=",
            float(args.max_thin_ray_fraction)),
        eikonal_abs_mean=check(metric_audit["mean"], "<=", .25),
        eikonal_abs_p95=check(metric_audit["p95"], "<=", .75),
        watertight=check(bool(mesh.is_watertight), "==", True),
        winding_consistent=check(bool(mesh.is_winding_consistent), "==", True),
        mesh_faces=check(int(len(mesh.faces)), ">=", 1_000),
        genus=check(int(genus), "<=", int(args.max_genus)))
    failures = [name for name, value in checks.items() if not value["passed"]]
    failures.extend([f"evidence.{name}" for name in evidence_gate["failures"]])

    preview_path = output / "surface_preview.png"
    preview(preview_path, mesh, evidence, seed=int(args.seed))
    report_path = output / "carved_sdf_gate.json"
    sdf_path = output / "base_sdf.npz"
    sdf_sha256 = save_sdf_grid(
        sdf_path, sdf, bound=float(args.bound), report_path=report_path)
    report = dict(
        schema="rootsplat.vggt_carved_sdf_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        protocol="train_only", scene=str(Path(args.scene).resolve()),
        depths=str(depths_path.resolve()), depth_gate=str(depth_gate_path.resolve()),
        accepted_view_ids=view_ids.tolist(), excluded_heldout_view_ids=[],
        fusion=dict(
            resolution=int(fusion["resolution"]), bound=float(fusion["bound"]),
            voxel_size=float(fusion["voxel_size"]),
            truncation=float(fusion["truncation"]),
            observed_voxels=int(np.count_nonzero(fusion["weight"])),
            observed_fraction=float(np.mean(fusion["weight"] > 0)),
            max_view_count=int(fusion["view_count"].max())),
        silhouette=dict(
            views=int(silhouette["views"]),
            dilation_pixels=int(silhouette["dilation_pixels"]),
            visible_views_max=int(silhouette["visible"].max()),
            foreground_votes_max=int(silhouette["foreground"].max())),
        carving=carve_audit, ray_solid=thickness,
        metric_sdf=metric_audit, evidence_gate=evidence_gate, checks=checks,
        mesh_topology=dict(
            vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
            components=components, watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            euler_number=euler, genus=genus, area=area, volume=volume,
            equivalent_thickness=equivalent_thickness),
        sdf=str(sdf_path.resolve()), sdf_sha256=sdf_sha256,
        mesh=str(mesh_path.resolve()), mesh_sha256=sha256_file(mesh_path),
        preview=str(preview_path.resolve()), options=vars(args))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    print(f"Gate report: {report_path}")
    print(f"Visual review: {preview_path}")
    print("DO NOT TRAIN unless every gate passes and the mesh is visually correct.")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
