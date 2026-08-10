#!/usr/bin/env python3
"""Fit RootSplat's SDF directly to calibrated train-only VGGT evidence.

This is deliberately a geometry-only gate.  It uses no RGB loss and starts no
Gaussian appearance optimization.  The accepted output is a digest-bound grid
SDF that can later be loaded as RootSplat's frozen base geometry.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from rootsplat.data import DTUScene
from rootsplat.neural_pull import (
    pull_projection, signed_ray_margin_loss,
    ray_solid_audit, sample_sdf_grid)
from rootsplat.sdf import NeuralSDF
from rootsplat.tsdf import (
    check, extract_sdf_mesh, mesh_evidence_gate, save_sdf_grid,
    sdf_metric_diagnostics, sha256_file)
from rootsplat.tsdf import camera_rays_numpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--query-sigma-start", type=float, default=.04)
    parser.add_argument("--query-sigma-end", type=float, default=.006)
    parser.add_argument("--ray-sign-offset", type=float, default=.03)
    parser.add_argument("--ray-sign-margin", type=float, default=.005)
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--balance-voxel", type=float, default=.005)
    parser.add_argument("--grid-resolution", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def calibrated_points(depths, weights, view_ids, cameras):
    points, directions, ranges, confidences = [], [], [], []
    for slot, view_id in enumerate(view_ids):
        depth = depths[slot]
        y, x = np.nonzero((depth > 0) & (weights[slot] > 0))
        pixels = np.stack([x + .5, y + .5], axis=-1)
        origin, direction = camera_rays_numpy(cameras[slot], pixels)
        points.append(origin + depth[y, x, None] * direction)
        directions.append(direction)
        ranges.append(depth[y, x])
        confidences.append(weights[slot, y, x])
    if not points:
        raise RuntimeError("No calibrated VGGT samples are available")
    return tuple(np.concatenate(value).astype(np.float32) for value in
                 (points, directions, ranges, confidences))


def spatially_balance(points, directions, ranges, weights, voxel, maximum,
                      bound, seed):
    valid = np.isfinite(points).all(-1) & np.isfinite(directions).all(-1) & \
        np.isfinite(ranges) & np.isfinite(weights) & (weights > 0) & \
        (np.max(np.abs(points), axis=-1) <= float(bound))
    points, directions = points[valid], directions[valid]
    ranges, weights = ranges[valid], weights[valid]
    cell = np.floor((points + float(bound)) / float(voxel)).astype(np.int64)
    order = np.lexsort((np.arange(len(points)), -weights,
                        cell[:, 2], cell[:, 1], cell[:, 0]))
    sorted_cell = cell[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = np.any(sorted_cell[1:] != sorted_cell[:-1], axis=1)
    selected = order[keep]
    if len(selected) > int(maximum):
        rng = np.random.default_rng(int(seed))
        probability = weights[selected].astype(np.float64)
        probability /= probability.sum()
        selected = rng.choice(selected, int(maximum), replace=False,
                              p=probability)
    return points[selected], directions[selected], ranges[selected], \
        weights[selected]


def boundary_points(count, bound, rng):
    point = rng.uniform(-bound, bound, size=(count, 3)).astype(np.float32)
    face = rng.integers(0, 6, size=count)
    axis = face // 2
    point[np.arange(count), axis] = np.where(face % 2, bound, -bound)
    return point


def evaluate_field(field, points, directions, rng, device, samples=20_000):
    ids = rng.choice(len(points), min(int(samples), len(points)), replace=False)
    point = torch.as_tensor(points[ids], dtype=torch.float32, device=device)
    direction = torch.as_tensor(
        directions[ids], dtype=torch.float32, device=device)
    with torch.no_grad():
        value = field(point).abs().cpu().numpy()
        free = field(point - .03 * direction).cpu().numpy()
        inside = field(point + .03 * direction).cpu().numpy()
    return dict(
        surface_abs_mean=float(value.mean()),
        surface_abs_p95=float(np.quantile(value, .95)),
        free_positive_fraction=float(np.mean(free > 0)),
        inside_negative_fraction=float(np.mean(inside < 0)))


def preview(path, mesh, evidence, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    rng = np.random.default_rng(int(seed))
    surface, _ = trimesh.sample.sample_surface(
        mesh, min(100_000, max(20_000, len(mesh.faces) * 2)), seed=rng)
    if len(evidence) > 100_000:
        evidence = evidence[rng.choice(len(evidence), 100_000, replace=False)]
    planes = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, (a, b, aname, bname) in zip(axes, planes):
        axis.scatter(evidence[:, a], evidence[:, b], s=.12, c="#e66101",
                     alpha=.30, rasterized=True, label="VGGT evidence")
        axis.scatter(surface[:, a], surface[:, b], s=.12, c="#0571b0",
                     alpha=.35, rasterized=True, label="Neural SDF")
        axis.set_xlabel(aname); axis.set_ylabel(bname)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1, 1); axis.set_ylim(-1, 1); axis.grid(alpha=.15)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle("Train-only VGGT evidence versus direct neural SDF")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    for name in ("steps", "batch_size", "max_points", "grid_resolution"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()
    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    archive = np.load(args.depths)
    view_ids = archive["view_ids"].astype(int)
    depths = archive["depth"].astype(np.float32)
    weights = archive["weight"].astype(np.float32)
    if depths.shape != weights.shape or len(depths) != len(view_ids):
        raise ValueError("Calibrated depth archive is misaligned")
    heldout = sorted(set(int(v) for v in view_ids) - set(scene.train_ids))
    if heldout:
        raise RuntimeError("Geometry initializer contains held-out views: " +
                           ", ".join(map(str, heldout)))
    cameras = [scene.view_cameras[int(view_id)] for view_id in view_ids]
    points, directions, ranges, confidence = calibrated_points(
        depths, weights, view_ids, cameras)
    source_points = len(points)
    points, directions, ranges, confidence = spatially_balance(
        points, directions, ranges, confidence,
        voxel=float(args.balance_voxel), maximum=int(args.max_points),
        bound=float(args.bound), seed=seed)
    if len(points) < 10_000:
        raise RuntimeError("Fewer than 10,000 balanced VGGT points remain")
    tree = cKDTree(points)

    device = torch.device(args.device)
    field = NeuralSDF(
        bound=float(args.bound), width=64, depth=2, geo_init_radius=.75).to(device)
    optimizer = torch.optim.Adam(field.parameters(), lr=float(args.learning_rate))
    trace_path = output / "training.jsonl"
    trace = []
    start_time = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        progress = (step - 1) / max(int(args.steps) - 1, 1)
        sigma = float(args.query_sigma_start) * \
            (float(args.query_sigma_end) / float(args.query_sigma_start)) ** progress
        ids = rng.integers(0, len(points), size=int(args.batch_size))
        point_np = points[ids]
        direction_np = directions[ids]
        query_np = point_np + rng.normal(
            0.0, sigma, size=point_np.shape).astype(np.float32)
        nearest = tree.query(query_np, workers=-1)[1]
        target_np = points[nearest]
        point = torch.as_tensor(point_np, device=device)
        direction = torch.as_tensor(direction_np, device=device)
        query = torch.as_tensor(query_np, device=device)
        target = torch.as_tensor(target_np, device=device)

        pulled, _value, gradient = pull_projection(field, query)
        pull = torch.sqrt((pulled - target).square().sum(-1) + 1e-8).mean()
        surface = field(point).abs().mean()
        sign, _free_local, _inside_local = signed_ray_margin_loss(
            field, point, direction, offset=float(args.ray_sign_offset),
            margin=float(args.ray_sign_margin))
        maximum = np.minimum(.5 * ranges[ids], .5)
        free_offset = .05 + rng.random(len(ids)).astype(np.float32) * \
            np.maximum(maximum - .05, .001)
        free_point = point - torch.as_tensor(
            free_offset, device=device)[:, None] * direction
        free = F.relu(float(args.ray_sign_margin) - field(free_point)).mean()
        eikonal_near = (gradient.norm(dim=-1) - 1.0).square().mean()

        uniform = torch.empty(512, 3, device=device).uniform_(
            -float(args.bound), float(args.bound))
        _uniform_value, uniform_gradient = field.s_and_grad(
            uniform, create_graph=True)
        eikonal_uniform = (uniform_gradient.norm(dim=-1) - 1.0).square().mean()
        boundary = torch.as_tensor(
            boundary_points(384, float(args.bound), rng), device=device)
        boundary_loss = F.relu(.02 - field(boundary)).mean()
        loss = pull + .25 * surface + .20 * sign + .10 * eikonal_near + \
            .03 * eikonal_uniform + .10 * free + .05 * boundary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            field.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == int(args.steps):
            row = dict(
                step=step, loss=float(loss.detach()), pull=float(pull.detach()),
                surface=float(surface.detach()), sign=float(sign.detach()),
                free=float(free.detach()), eikonal_near=float(eikonal_near.detach()),
                eikonal_uniform=float(eikonal_uniform.detach()),
                boundary=float(boundary_loss.detach()), sigma=sigma,
                gradient_norm=float(norm.detach()))
            trace.append(row)
            with trace_path.open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    checkpoint = output / "checkpoints/final.pt"
    torch.save(dict(
        schema="rootsplat.neural_pull_sdf_checkpoint.v1",
        state_dict=field.state_dict(), options=vars(args), trace=trace), checkpoint)
    field_audit = evaluate_field(
        field, points, directions, rng, device=device)
    solid_audit = ray_solid_audit(
        field, points, directions, samples=2_000, seed=seed)
    sdf = sample_sdf_grid(
        field, resolution=int(args.grid_resolution), bound=float(args.bound))
    voxel = 2.0 * float(args.bound) / int(args.grid_resolution)
    mesh = extract_sdf_mesh(sdf, bound=float(args.bound))
    mesh_path = output / "surface_normalized.ply"
    mesh.export(mesh_path)
    metric = sdf_metric_diagnostics(sdf, voxel)
    evidence_gate = mesh_evidence_gate(
        mesh, points, cameras, depths, truncation=4.0 * voxel,
        samples=150_000, seed=seed)
    preview_path = output / "surface_preview.png"
    preview(preview_path, mesh, points, seed=seed)

    report_path = output / "neural_pull_sdf_gate.json"
    sdf_path = output / "base_sdf.npz"
    sdf_sha256 = save_sdf_grid(
        sdf_path, sdf, bound=float(args.bound), report_path=report_path)
    boundary_positive = bool(
        np.all(sdf[[0, -1]] > 0) and np.all(sdf[:, [0, -1]] > 0) and
        np.all(sdf[:, :, [0, -1]] > 0))
    inside_fraction = float(np.mean(sdf < 0))
    checks = dict(
        balanced_points=check(int(len(points)), ">=", 10_000),
        heldout_geometry_views=check(len(heldout), "==", 0),
        positive_volume_boundary=check(boundary_positive, "==", True),
        interior_fraction_min=check(inside_fraction, ">=", .01),
        interior_fraction_max=check(inside_fraction, "<=", .30),
        surface_abs_p95=check(field_audit["surface_abs_p95"], "<=", .04),
        local_free_positive=check(
            field_audit["free_positive_fraction"], ">=", .90),
        local_inside_negative=check(
            field_audit["inside_negative_fraction"], ">=", .90),
        ray_entered_inside=check(
            solid_audit["entered_inside_fraction"], ">=", .90),
        ray_thickness_median=check(
            solid_audit["thickness_median"], ">=", .08),
        ray_thin_fraction=check(
            solid_audit["thin_fraction_005"], "<=", .35),
        eikonal_abs_mean=check(metric["mean"], "<=", .25),
        eikonal_abs_p95=check(metric["p95"], "<=", .75),
        watertight=check(bool(mesh.is_watertight), "==", True),
        winding_consistent=check(bool(mesh.is_winding_consistent), "==", True),
        mesh_faces=check(int(len(mesh.faces)), ">=", 1_000))
    failures = [name for name, value in checks.items() if not value["passed"]]
    failures.extend([f"evidence.{name}" for name in evidence_gate["failures"]])
    report = dict(
        schema="rootsplat.vggt_neural_pull_sdf_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        protocol="train_only", scene=str(Path(args.scene).resolve()),
        calibrated_depths=str(Path(args.depths).resolve()),
        accepted_view_ids=view_ids.tolist(), excluded_heldout_view_ids=heldout,
        points=dict(source=int(source_points), balanced=int(len(points))),
        training_seconds=float(time.perf_counter() - start_time),
        field=field_audit, ray_solid=solid_audit, metric_sdf=metric,
        evidence_gate=evidence_gate, checks=checks,
        grid_sdf=dict(resolution=int(args.grid_resolution), voxel_size=voxel,
                      inside_fraction=inside_fraction,
                      boundary_positive=boundary_positive),
        mesh_topology=dict(
            vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
            watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            euler_number=int(mesh.euler_number)),
        sdf=str(sdf_path.resolve()), sdf_sha256=sdf_sha256,
        mesh=str(mesh_path.resolve()), mesh_sha256=sha256_file(mesh_path),
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256_file(checkpoint),
        preview=str(preview_path.resolve()), options=vars(args))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    print("Gate report:", report_path)
    print("Visual review:", preview_path)
    print("DO NOT TRAIN APPEARANCE unless status is pass and the preview is correct.")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
