#!/usr/bin/env python3
"""Fit a neural SDF from all accepted train-only VGGT range images.

This geometry-only capacity gate supervises signed distance along calibrated
camera rays.  It does not optimize RGB, Gaussian appearance, MV-RoMa tracks,
or any held-out DTU view.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from rootsplat.data import DTUScene
from rootsplat.neural_pull import ray_solid_audit, sample_sdf_grid
from rootsplat.ray_sdf import (
    oriented_ray_evidence, projective_sdf_target,
    spatially_balance_rays, split_ray_evidence)
from rootsplat.sdf import NeuralSDF
from rootsplat.tsdf import (
    check, extract_sdf_mesh, mesh_evidence_gate, save_sdf_grid,
    sdf_metric_diagnostics, sha256_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--depth-gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--band-start", type=float, default=.06)
    parser.add_argument("--band-end", type=float, default=.012)
    parser.add_argument("--sign-offset", type=float, default=.025)
    parser.add_argument("--sign-margin", type=float, default=.003)
    parser.add_argument("--normal-edge-length", type=float, default=.04)
    parser.add_argument("--minimum-incidence", type=float, default=.15)
    parser.add_argument("--balance-voxel", type=float, default=.004)
    parser.add_argument("--max-rays", type=int, default=400_000)
    parser.add_argument("--validation-fraction", type=float, default=.10)
    parser.add_argument("--grid-resolution", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def boundary_points(count, bound, rng):
    points = rng.uniform(-bound, bound, size=(count, 3)).astype(np.float32)
    face = rng.integers(0, 6, size=count)
    axis = face // 2
    points[np.arange(count), axis] = np.where(face % 2, bound, -bound)
    return points


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def evaluate_field(field, arrays, ids, rng, device, bound, samples=20_000):
    points, directions, normals, ranges, weights, _supports, _views = arrays
    ids = rng.choice(ids, min(int(samples), len(ids)), replace=False)
    point_np = points[ids]
    direction_np = directions[ids]
    normal_np = normals[ids]
    point = torch.as_tensor(point_np, device=device)
    direction = torch.as_tensor(direction_np, device=device)
    normal = torch.as_tensor(normal_np, device=device)
    with torch.enable_grad():
        value, gradient = field.s_and_grad(point, create_graph=False)
    unit = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cosine = (unit * normal).sum(-1).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine)).detach().cpu().numpy()
    with torch.no_grad():
        free_local = field(point - .025 * direction).cpu().numpy()
        inside_local = field(point + .025 * direction).cpu().numpy()
        offset_np = rng.uniform(-.03, .03, size=len(ids)).astype(np.float32)
        target_np, _ = projective_sdf_target(
            offset_np, normal_np, direction_np)
        band_query = point + torch.as_tensor(
            offset_np, device=device)[:, None] * direction
        band_value = field(band_query).cpu().numpy()
        maximum = np.minimum(.60, .70 * ranges[ids])
        far_offset = .06 + rng.random(len(ids)).astype(np.float32) * \
            np.maximum(maximum - .06, .001)
        far_np = point_np - far_offset[:, None] * direction_np
        valid_far = np.max(np.abs(far_np), axis=-1) <= float(bound)
        far_value = field(torch.as_tensor(
            far_np[valid_far], device=device)).cpu().numpy() \
            if valid_far.any() else np.empty(0, np.float32)
    absolute = np.abs(value.detach().cpu().numpy())
    band_error = np.abs(band_value - target_np)
    return dict(
        samples=int(len(ids)), surface_abs_mean=float(absolute.mean()),
        surface_abs_p95=float(np.quantile(absolute, .95)),
        band_abs_mean=float(band_error.mean()),
        band_abs_p95=float(np.quantile(band_error, .95)),
        local_free_positive_fraction=float(np.mean(free_local > 0)),
        local_inside_negative_fraction=float(np.mean(inside_local < 0)),
        far_free_samples=int(len(far_value)),
        far_free_positive_fraction=float(np.mean(far_value > 0))
            if len(far_value) else 0.0,
        normal_mean_degrees=float(angle.mean()),
        normal_p95_degrees=float(np.quantile(angle, .95)))


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
                     alpha=.30, rasterized=True, label="42-view VGGT depth")
        axis.scatter(surface[:, a], surface[:, b], s=.12, c="#0571b0",
                     alpha=.35, rasterized=True, label="ray-supervised SDF")
        axis.set(xlabel=aname, ylabel=bname, aspect="equal",
                 xlim=(-1, 1), ylim=(-1, 1))
        axis.grid(alpha=.15)
    axes[0].legend(markerscale=12, loc="upper right")
    figure.suptitle("Train-only VGGT evidence versus full-view ray SDF")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if int(args.steps) <= 0 or int(args.batch_size) <= 0 or \
            int(args.grid_resolution) < 32:
        raise ValueError("Invalid training or grid options")
    if not 0 < float(args.band_end) <= float(args.band_start):
        raise ValueError("Ray-band schedule is invalid")
    output.mkdir(parents=True)
    (output / "checkpoints").mkdir()

    depth_path = Path(args.depths)
    depth_gate_path = Path(args.depth_gate)
    depth_gate = json.loads(depth_gate_path.read_text(encoding="utf8"))
    if depth_gate.get("schema") != "rootsplat.vggt_fulltrain_depth_gate.v1" or \
            depth_gate.get("status") != "pass" or depth_gate.get("failures"):
        raise RuntimeError("A passing 42-training-view depth gate is required")
    if depth_gate.get("output_sha256") != sha256_file(depth_path):
        raise RuntimeError("Full-view depth archive digest mismatch")

    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    with np.load(depth_path) as archive:
        required = {"view_ids", "depth", "weight", "support"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError("Full-view depth archive misses: " +
                           ", ".join(sorted(missing)))
        view_ids = archive["view_ids"].astype(np.int64)
        depths = archive["depth"].astype(np.float32)
        weights = archive["weight"].astype(np.float32)
        supports = archive["support"].astype(np.uint8)
    expected_train = np.asarray(scene.train_ids, dtype=np.int64)
    heldout = set(int(value) for value in scene.test_ids)
    if not np.array_equal(view_ids, expected_train):
        raise RuntimeError("Ray SDF requires every training view in canonical order")
    if set(view_ids.tolist()) & heldout:
        raise RuntimeError("Held-out views reached ray-SDF fitting")
    cameras = [scene.view_cameras[int(view_id)] for view_id in view_ids]

    arrays, ray_audit = oriented_ray_evidence(
        depths, weights, supports, cameras,
        max_edge_length=float(args.normal_edge_length),
        minimum_incidence=float(args.minimum_incidence),
        bound=float(args.bound))
    arrays, balance_audit = spatially_balance_rays(
        *arrays, voxel=float(args.balance_voxel), maximum=int(args.max_rays),
        bound=float(args.bound), seed=seed)
    points, directions, normals, ranges, confidence, _, ray_view_slots = arrays
    train_ids, validation_ids = split_ray_evidence(
        len(points), validation_fraction=float(args.validation_fraction),
        seed=seed)

    device = torch.device(args.device)
    field = NeuralSDF(
        bound=float(args.bound), width=64, depth=2,
        geo_init_radius=.75).to(device)
    optimizer = torch.optim.Adam(
        field.parameters(), lr=float(args.learning_rate))
    initial_validation_field = evaluate_field(
        field, arrays, validation_ids, np.random.default_rng(seed + 101),
        device, float(args.bound))
    trace_path = output / "training.jsonl"
    trace = []
    started = time.perf_counter()
    for step in range(1, int(args.steps) + 1):
        progress = (step - 1) / max(int(args.steps) - 1, 1)
        band_width = float(args.band_start) * \
            (float(args.band_end) / float(args.band_start)) ** progress
        ids = rng.choice(train_ids, int(args.batch_size), replace=True)
        point_np = points[ids]
        direction_np = directions[ids]
        normal_np = normals[ids]
        ranges_np = ranges[ids]
        point = torch.as_tensor(point_np, device=device)
        direction = torch.as_tensor(direction_np, device=device)
        normal = torch.as_tensor(normal_np, device=device)
        sample_weight = torch.sqrt(torch.as_tensor(
            np.maximum(confidence[ids], 1e-8), device=device))
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-8)

        surface_value, surface_gradient = field.s_and_grad(
            point, create_graph=True)
        surface_loss = weighted_mean(surface_value.abs(), sample_weight)
        surface_unit = surface_gradient / surface_gradient.norm(
            dim=-1, keepdim=True).clamp_min(1e-9)
        normal_loss = weighted_mean(
            1.0 - (surface_unit * normal).sum(-1).clamp(-1, 1),
            sample_weight)

        offset_np = rng.uniform(
            -band_width, band_width, size=len(ids)).astype(np.float32)
        target_np, _incidence = projective_sdf_target(
            offset_np, normal_np, direction_np,
            minimum_incidence=float(args.minimum_incidence))
        offset = torch.as_tensor(offset_np, device=device)
        target = torch.as_tensor(target_np, device=device)
        band_query = point + offset[:, None] * direction
        band_value, band_gradient = field.s_and_grad(
            band_query, create_graph=True)
        band_error = F.smooth_l1_loss(
            band_value, target, reduction="none", beta=.005)
        band_loss = weighted_mean(band_error, sample_weight)

        sign_offset = float(args.sign_offset)
        margin = float(args.sign_margin)
        free_local = field(point - sign_offset * direction)
        inside_local = field(point + sign_offset * direction)
        sign_loss = weighted_mean(
            F.relu(margin - free_local) + F.relu(margin + inside_local),
            sample_weight)

        maximum = np.minimum(.60, .70 * ranges_np)
        free_offset_np = .06 + rng.random(len(ids)).astype(np.float32) * \
            np.maximum(maximum - .06, .001)
        far_np = point_np - free_offset_np[:, None] * direction_np
        valid_far = np.max(np.abs(far_np), axis=-1) <= float(args.bound)
        if valid_far.any():
            far_value = field(torch.as_tensor(far_np[valid_far], device=device))
            far_weight = sample_weight[torch.as_tensor(valid_far, device=device)]
            free_loss = weighted_mean(F.relu(margin - far_value), far_weight)
        else:
            free_loss = surface_value.new_zeros(())

        near_gradient = torch.cat([surface_gradient, band_gradient], dim=0)
        eikonal_near = (near_gradient.norm(dim=-1) - 1.0).square().mean()
        uniform = torch.empty(512, 3, device=device).uniform_(
            -float(args.bound), float(args.bound))
        _uniform_value, uniform_gradient = field.s_and_grad(
            uniform, create_graph=True)
        eikonal_uniform = (uniform_gradient.norm(dim=-1) - 1.0).square().mean()
        boundary = torch.as_tensor(
            boundary_points(384, float(args.bound), rng), device=device)
        boundary_loss = F.relu(.02 - field(boundary)).mean()

        loss = band_loss + .25 * surface_loss + .15 * sign_loss + \
            .15 * free_loss + .05 * normal_loss + .05 * eikonal_near + \
            .01 * eikonal_uniform + .05 * boundary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            field.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()

        if step == 1 or step % 25 == 0 or step == int(args.steps):
            row = dict(
                step=int(step), loss=float(loss.detach()),
                band=float(band_loss.detach()),
                surface=float(surface_loss.detach()),
                sign=float(sign_loss.detach()), free=float(free_loss.detach()),
                normal=float(normal_loss.detach()),
                eikonal_near=float(eikonal_near.detach()),
                eikonal_uniform=float(eikonal_uniform.detach()),
                boundary=float(boundary_loss.detach()),
                band_width=float(band_width),
                gradient_norm=float(gradient_norm.detach()))
            trace.append(row)
            with trace_path.open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    checkpoint = output / "checkpoints/final.pt"
    torch.save(dict(
        schema="rootsplat.fullview_ray_sdf_checkpoint.v1",
        state_dict=field.state_dict(), options=vars(args), trace=trace), checkpoint)
    train_field = evaluate_field(
        field, arrays, train_ids, np.random.default_rng(seed + 100),
        device, float(args.bound))
    validation_field = evaluate_field(
        field, arrays, validation_ids, np.random.default_rng(seed + 101),
        device, float(args.bound))
    solid = ray_solid_audit(
        field, points[validation_ids], directions[validation_ids],
        samples=4_000, maximum_depth=.5, steps=101, seed=seed)
    sdf = sample_sdf_grid(
        field, resolution=int(args.grid_resolution), bound=float(args.bound))
    voxel_size = 2.0 * float(args.bound) / int(args.grid_resolution)
    metric = sdf_metric_diagnostics(sdf, voxel_size)
    mesh = extract_sdf_mesh(sdf, bound=float(args.bound))
    mesh_path = output / "surface_normalized.ply"
    mesh.export(mesh_path)
    evidence_gate = mesh_evidence_gate(
        mesh, points, cameras, depths, truncation=4.0 * voxel_size,
        samples=200_000, seed=seed,
        point_to_surface_p95_max=.04, surface_to_point_p95_max=.08,
        unsupported_distance=.05, unsupported_fraction_max=.10,
        free_space_violation_max=.02, max_components=4,
        min_largest_component_fraction=.95)
    preview_path = output / "surface_preview.png"
    preview(preview_path, mesh, points, seed=seed)

    report_path = output / "fullview_ray_sdf_gate.json"
    sdf_path = output / "base_sdf.npz"
    sdf_sha256 = save_sdf_grid(
        sdf_path, sdf, bound=float(args.bound), report_path=report_path)
    boundary_positive = bool(
        np.all(sdf[[0, -1], :, :] > 0) and
        np.all(sdf[:, [0, -1], :] > 0) and
        np.all(sdf[:, :, [0, -1]] > 0))
    inside_fraction = float(np.mean(sdf < 0))
    components = int(evidence_gate["components"])
    euler = int(mesh.euler_number)
    genus = max(0, int(round((2 * components - euler) / 2))) \
        if mesh.is_watertight and mesh.is_winding_consistent else -1
    checks = dict(
        exact_training_view_coverage=check(
            int(len(view_ids)), "==", int(len(expected_train))),
        heldout_geometry_views=check(
            int(len(set(view_ids.tolist()) & heldout)), "==", 0),
        balanced_rays=check(int(len(points)), ">=", 50_000),
        contributing_views=check(
            int(len(np.unique(ray_view_slots))), "==", int(len(view_ids))),
        validation_rays=check(int(len(validation_ids)), ">=", 5_000),
        positive_volume_boundary=check(boundary_positive, "==", True),
        interior_fraction_min=check(inside_fraction, ">=", .005),
        interior_fraction_max=check(inside_fraction, "<=", .20),
        validation_surface_abs_p95=check(
            validation_field["surface_abs_p95"], "<=", .03),
        validation_surface_improvement=check(
            validation_field["surface_abs_mean"] /
            max(initial_validation_field["surface_abs_mean"], 1e-8),
            "<=", .80),
        validation_band_abs_p95=check(
            validation_field["band_abs_p95"], "<=", .04),
        validation_band_improvement=check(
            validation_field["band_abs_mean"] /
            max(initial_validation_field["band_abs_mean"], 1e-8),
            "<=", .80),
        validation_local_free_positive=check(
            validation_field["local_free_positive_fraction"], ">=", .90),
        validation_local_inside_negative=check(
            validation_field["local_inside_negative_fraction"], ">=", .90),
        validation_far_free_positive=check(
            validation_field["far_free_positive_fraction"], ">=", .95),
        validation_normal_mean_degrees=check(
            validation_field["normal_mean_degrees"], "<=", 25.0),
        ray_entered_inside=check(
            solid["entered_inside_fraction"], ">=", .90),
        ray_thickness_median=check(solid["thickness_median"], ">=", .08),
        ray_thin_fraction=check(solid["thin_fraction_005"], "<=", .35),
        ray_censored_fraction=check(solid["censored_fraction"], "<=", .60),
        eikonal_abs_mean=check(metric["mean"], "<=", .25),
        eikonal_abs_p95=check(metric["p95"], "<=", .75),
        watertight=check(bool(mesh.is_watertight), "==", True),
        winding_consistent=check(bool(mesh.is_winding_consistent), "==", True),
        mesh_faces=check(int(len(mesh.faces)), ">=", 1_000),
        genus=check(int(genus), "<=", 32))
    failures = [name for name, value in checks.items() if not value["passed"]]
    failures.extend([f"evidence.{name}" for name in evidence_gate["failures"]])
    report = dict(
        schema="rootsplat.vggt_fullview_ray_sdf_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        protocol="train_only", scene=str(Path(args.scene).resolve()),
        depths=str(depth_path.resolve()), depth_gate=str(depth_gate_path.resolve()),
        accepted_view_ids=view_ids.tolist(), excluded_heldout_view_ids=[],
        ray_evidence=ray_audit, balancing=balance_audit,
        split=dict(train=int(len(train_ids)), validation=int(len(validation_ids)),
                   overlap=int(len(np.intersect1d(train_ids, validation_ids)))),
        training_seconds=float(time.perf_counter() - started),
        initial_validation_field=initial_validation_field,
        train_field=train_field, validation_field=validation_field,
        ray_solid=solid, metric_sdf=metric, evidence_gate=evidence_gate,
        checks=checks,
        grid_sdf=dict(
            resolution=int(args.grid_resolution), voxel_size=float(voxel_size),
            inside_fraction=inside_fraction, boundary_positive=boundary_positive),
        mesh_topology=dict(
            vertices=int(len(mesh.vertices)), faces=int(len(mesh.faces)),
            components=components, watertight=bool(mesh.is_watertight),
            winding_consistent=bool(mesh.is_winding_consistent),
            euler_number=euler, genus=genus, area=float(mesh.area),
            volume=float(abs(mesh.volume))),
        sdf=str(sdf_path.resolve()), sdf_sha256=sdf_sha256,
        mesh=str(mesh_path.resolve()), mesh_sha256=sha256_file(mesh_path),
        checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256_file(checkpoint),
        preview=str(preview_path.resolve()), options=vars(args))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    print("Gate report:", report_path)
    print("Visual review:", preview_path)
    print("DO NOT TRAIN APPEARANCE unless every gate passes and the mesh is correct.")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
