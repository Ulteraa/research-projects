#!/usr/bin/env python3
"""Gate a COLMAP visibility graph-cut mesh and convert it to a metric SDF."""
from pathlib import Path
import argparse
import json

import numpy as np
import trimesh

from rootsplat.data import DTUScene
from rootsplat.poisson_sdf import mesh_to_sdf_grid
from rootsplat.tsdf import (
    check, depth_map_points, mesh_evidence_gate, save_sdf_grid,
    sdf_metric_diagnostics, sha256_file)


def _preview(path, mesh, evidence, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(int(seed))
    surface, _ = trimesh.sample.sample_surface(
        mesh, min(120_000, max(30_000, len(mesh.faces) * 2)), seed=rng)
    if len(evidence) > 120_000:
        evidence = evidence[rng.choice(len(evidence), 120_000, replace=False)]
    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (a, b, aname, bname) in zip(axes, planes):
        axis.scatter(evidence[:, a], evidence[:, b], s=.12, c="#e66101",
                     alpha=.25, rasterized=True, label="42-view VGGT depth")
        axis.scatter(surface[:, a], surface[:, b], s=.12, c="#0571b0",
                     alpha=.35, rasterized=True, label="Delaunay graph-cut surface")
        axis.set(xlabel=aname, ylabel=bname, aspect="equal", xlim=(-1, 1),
                 ylim=(-1, 1))
        axis.grid(alpha=.15)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle("Train-only depth evidence versus visibility graph-cut SDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _load_dominant_mesh(path, minimum_area_fraction=.995):
    raw = trimesh.load(path, force="mesh", process=True)
    if not isinstance(raw, trimesh.Trimesh) or len(raw.faces) < 4:
        raise RuntimeError("COLMAP Delaunay output is not a triangle mesh")
    components = raw.split(only_watertight=False)
    areas = np.asarray([part.area for part in components], dtype=np.float64)
    largest = int(np.argmax(areas))
    fraction = float(areas[largest] / max(areas.sum(), 1e-12))
    if fraction < float(minimum_area_fraction):
        raise RuntimeError(
            "Delaunay output has material disconnected components; refusing "
            "to hide them before signed-distance conversion")
    mesh = components[largest].copy()
    mesh.remove_unreferenced_vertices()
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh, multibody=False)
    if mesh.is_watertight and float(mesh.volume) < 0:
        mesh.invert()
    audit = dict(
        raw_vertices=int(len(raw.vertices)), raw_faces=int(len(raw.faces)),
        raw_components=int(len(components)), raw_component_faces=[
            int(len(part.faces)) for part in components],
        largest_component_area_fraction=fraction,
        kept_vertices=int(len(mesh.vertices)), kept_faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        signed_volume=float(mesh.volume), euler_number=int(mesh.euler_number))
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(
            "Dominant Delaunay component is not a closed oriented surface; "
            "refusing to assign a signed distance")
    return mesh, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--depth-gate", required=True)
    parser.add_argument("--workspace-gate", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--sign-samples", type=int, default=3)
    parser.add_argument("--minimum-component-area-fraction", type=float,
                        default=.995)
    parser.add_argument("--point-to-surface-p95-max", type=float, default=.04)
    parser.add_argument("--surface-to-point-p95-max", type=float, default=.08)
    parser.add_argument("--unsupported-distance", type=float, default=.05)
    parser.add_argument("--unsupported-fraction-max", type=float, default=.10)
    parser.add_argument("--free-space-violation-max", type=float, default=.02)
    parser.add_argument("--max-genus", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Delaunay-SDF output: {output}")
    depths_path, depth_gate_path = Path(args.depths), Path(args.depth_gate)
    workspace_gate_path, raw_mesh_path = Path(args.workspace_gate), Path(args.mesh)
    depth_report = json.loads(depth_gate_path.read_text(encoding="utf8"))
    workspace_report = json.loads(workspace_gate_path.read_text(encoding="utf8"))
    if depth_report.get("schema") != "rootsplat.vggt_fulltrain_depth_gate.v1" or \
            depth_report.get("status") != "pass" or depth_report.get("failures"):
        raise RuntimeError("A passing full-training-view depth gate is required")
    if depth_report.get("output_sha256") != sha256_file(depths_path):
        raise RuntimeError("Full-training-view depth archive digest mismatch")
    if workspace_report.get("schema") != \
            "rootsplat.vggt_delaunay_workspace_gate.v1" or \
            workspace_report.get("status") != "pass" or \
            workspace_report.get("failures"):
        raise RuntimeError("A passing Delaunay workspace gate is required")
    if workspace_report.get("depth_sha256") != sha256_file(depths_path):
        raise RuntimeError("Delaunay workspace belongs to a different depth cache")

    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    with np.load(depths_path) as archive:
        view_ids = archive["view_ids"].astype(np.int64)
        depth_maps = archive["depth"].astype(np.float32)
    expected = np.asarray(scene.train_ids, dtype=np.int64)
    if not np.array_equal(view_ids, expected) or \
            workspace_report.get("view_ids") != expected.tolist():
        raise RuntimeError("Delaunay-SDF inputs do not cover the canonical train split")
    if set(view_ids.tolist()) & set(scene.test_ids):
        raise RuntimeError("Held-out views reached the Delaunay-SDF constructor")
    cameras = [scene.view_cameras[int(index)] for index in view_ids]
    mesh, mesh_audit = _load_dominant_mesh(
        raw_mesh_path,
        minimum_area_fraction=float(args.minimum_component_area_fraction))
    output.mkdir(parents=True)
    mesh_path = output / "surface_normalized.ply"
    mesh.export(mesh_path)
    sdf, grid_audit = mesh_to_sdf_grid(
        mesh, resolution=int(args.resolution), bound=float(args.bound),
        sign_samples=int(args.sign_samples))
    metric_audit = sdf_metric_diagnostics(sdf, grid_audit["voxel_size"])
    evidence = depth_map_points(depth_maps, cameras, seed=int(args.seed))
    evidence_gate = mesh_evidence_gate(
        mesh, evidence, cameras, depth_maps,
        truncation=4.0 * grid_audit["voxel_size"], seed=int(args.seed),
        point_to_surface_p95_max=float(args.point_to_surface_p95_max),
        surface_to_point_p95_max=float(args.surface_to_point_p95_max),
        unsupported_distance=float(args.unsupported_distance),
        unsupported_fraction_max=float(args.unsupported_fraction_max),
        free_space_violation_max=float(args.free_space_violation_max),
        max_components=1, min_largest_component_fraction=1.0)
    euler = int(mesh.euler_number)
    genus = max(0, int(round((2 - euler) / 2)))
    checks = dict(
        exact_training_view_coverage=check(len(view_ids), "==", len(expected)),
        heldout_geometry_views=check(
            len(set(view_ids.tolist()) & set(scene.test_ids)), "==", 0),
        raw_mesh_available=check(raw_mesh_path.is_file(), "==", True),
        dominant_component_fraction=check(
            mesh_audit["largest_component_area_fraction"], ">=",
            float(args.minimum_component_area_fraction)),
        watertight=check(bool(mesh.is_watertight), "==", True),
        winding_consistent=check(bool(mesh.is_winding_consistent), "==", True),
        positive_volume_boundary=check(grid_audit["boundary_positive"], "==", True),
        interior_fraction_min=check(grid_audit["inside_fraction"], ">=", .001),
        interior_fraction_max=check(grid_audit["inside_fraction"], "<=", .30),
        eikonal_abs_mean=check(metric_audit["mean"], "<=", .25),
        eikonal_abs_p95=check(metric_audit["p95"], "<=", .75),
        mesh_faces=check(len(mesh.faces), ">=", 1_000),
        genus=check(genus, "<=", int(args.max_genus)))
    failures = [name for name, value in checks.items() if not value["passed"]]
    failures.extend([f"evidence.{name}" for name in evidence_gate["failures"]])
    preview_path = output / "surface_preview.png"
    _preview(preview_path, mesh, evidence, seed=int(args.seed))
    report_path = output / "delaunay_sdf_gate.json"
    sdf_path = output / "base_sdf.npz"
    sdf_sha256 = save_sdf_grid(
        sdf_path, sdf, bound=float(args.bound), report_path=report_path)
    report = dict(
        schema="rootsplat.vggt_delaunay_sdf_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        protocol="train_only", scene=str(Path(args.scene).resolve()),
        depths=str(depths_path.resolve()), depth_gate=str(depth_gate_path.resolve()),
        workspace_gate=str(workspace_gate_path.resolve()),
        workspace_gate_sha256=sha256_file(workspace_gate_path),
        raw_mesh=str(raw_mesh_path.resolve()),
        raw_mesh_sha256=sha256_file(raw_mesh_path),
        accepted_view_ids=view_ids.tolist(), excluded_heldout_view_ids=[],
        mesh_processing=mesh_audit, grid_sdf=grid_audit,
        metric_sdf=metric_audit, evidence_gate=evidence_gate, checks=checks,
        mesh_topology=dict(
            vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
            components=1, watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            euler_number=euler, genus=genus),
        sdf=str(sdf_path.resolve()), sdf_sha256=sdf_sha256,
        mesh=str(mesh_path.resolve()), mesh_sha256=sha256_file(mesh_path),
        preview=str(preview_path.resolve()), options=vars(args))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    print(f"Gate report: {report_path}")
    print(f"Visual review: {preview_path}")
    print("DO NOT TRAIN APPEARANCE unless this gate passes and the mesh is correct.")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
