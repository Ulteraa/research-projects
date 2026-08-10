#!/usr/bin/env python3
"""Evaluate the mask-only SDF bootstrap before any joint/appearance training."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.artifacts import save_json, write_ply
from rootsplat.bootstrap_gate import (
    add_iou_preservation_checks, eikonal_diagnostics, evaluate_scene_masks,
    geometry_state_audit, make_gate_checks,
    mesh_normal_diagnostics, save_gate_montage, sphere_shape_diagnostics)
from rootsplat.data import transform_points
from rootsplat.evaluation import topology_statistics
from rootsplat.experiment import load_checkpoint, load_config
from rootsplat.marching_tets import face_area
from rootsplat.sdf import Sphere


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--ray-samples", type=int, default=64)
    p.add_argument("--ray-chunk", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--eikonal-samples", type=int, default=32768)
    p.add_argument("--export-resolution", type=int, default=0)
    p.add_argument("--stage", choices=("bootstrap", "joint", "appearance"),
                   default="bootstrap")
    p.add_argument("--reference-report",
                   help="Passing bootstrap JSON required after bootstrap")
    p.add_argument("--reference-checkpoint",
                   help="Bootstrap checkpoint required for --stage appearance")
    p.add_argument("--max-iou-drop", type=float, default=0.03)
    a = p.parse_args()
    if str(a.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    run = Path(a.run)
    cfg = load_config(run / "config.resolved.yaml")
    dcfg, mcfg = cfg["dataset"], cfg["model"]
    bootstrap_iters = int(mcfg.get("bootstrap_iters", -1))
    joint_iters = int(mcfg.get("joint_iters", -1))
    if a.stage == "bootstrap" and joint_iters != 0:
        raise RuntimeError(
            "Bootstrap gate requires a checkpoint trained with joint_iters=0")
    if a.stage in ("joint", "appearance") and joint_iters <= 0:
        raise RuntimeError("Post-bootstrap gate requires joint_iters>0")
    if a.stage in ("joint", "appearance") and not a.reference_report:
        raise RuntimeError("Post-bootstrap gate requires --reference-report")
    if a.stage == "appearance":
        if not a.reference_checkpoint:
            raise RuntimeError(
                "--stage appearance requires --reference-checkpoint")
        if int(mcfg.get("appearance_only_iters", 0)) != joint_iters:
            raise RuntimeError(
                "Appearance gate requires appearance_only_iters == joint_iters")
    prefix = {"bootstrap": "bootstrap_gate",
              "joint": "joint_geometry_gate",
              "appearance": "appearance_gate"}[a.stage]
    scene = DTUScene(
        dcfg["scene"], device=a.device,
        downscale=dcfg.get("downscale", 0.25),
        test_every=dcfg.get("test_every", 8),
        priors_dir=dcfg.get("priors_dir"),
        depth_type=dcfg.get("depth_type", "ray"),
        normal_space=dcfg.get("normal_space", "world"),
        depth_scale=dcfg.get("depth_scale", 1.0),
        require_masks=dcfg.get("require_masks", False),
        require_scale_matrices=dcfg.get("require_scale_matrices", False))
    model = RootSplat(Config(**mcfg), device=a.device)
    checkpoint = Path(a.checkpoint) if a.checkpoint else run / "checkpoints" / "final.pt"
    state = load_checkpoint(checkpoint, model, map_location=a.device, restore_rng=False)
    expected_step = bootstrap_iters + \
        (joint_iters if a.stage in ("joint", "appearance") else 0)
    if int(state.get("step", -1)) != expected_step:
        raise RuntimeError(
            "Checkpoint step does not equal the declared stage duration")
    model.eval()

    # Four fixed views span both optimization and held-out sets for visual audit.
    retain = []
    for group in (scene.train_ids, scene.test_ids):
        if group:
            retain.extend([group[0], group[len(group) // 2]])
    retain = list(dict.fromkeys(retain))
    learned_train, learned_train_rows, learned_train_images = evaluate_scene_masks(
        model.sdf, scene, scene.train_ids, model.cfg.bound,
        stride=a.stride, n_samples=a.ray_samples,
        temperature=a.temperature, ray_chunk=a.ray_chunk,
        retain_view_ids=retain)
    learned_test, learned_test_rows, learned_test_images = evaluate_scene_masks(
        model.sdf, scene, scene.test_ids, model.cfg.bound,
        stride=a.stride, n_samples=a.ray_samples,
        temperature=a.temperature, ray_chunk=a.ray_chunk,
        retain_view_ids=retain)

    initial = Sphere(0.5).to(a.device)
    initial_train, initial_train_rows, initial_train_images = evaluate_scene_masks(
        initial, scene, scene.train_ids, model.cfg.bound,
        stride=a.stride, n_samples=a.ray_samples,
        temperature=a.temperature, ray_chunk=a.ray_chunk,
        retain_view_ids=retain)
    initial_test, initial_test_rows, initial_test_images = evaluate_scene_masks(
        initial, scene, scene.test_ids, model.cfg.bound,
        stride=a.stride, n_samples=a.ray_samples,
        temperature=a.temperature, ray_chunk=a.ray_chunk,
        retain_view_ids=retain)

    resolution = a.export_resolution or cfg.get("training", {}).get(
        "export_resolution", 64)
    vertices, faces = model.export_mesh(res=resolution, cams=scene.train_cameras)
    normalized = vertices.detach().cpu().numpy()
    face_np = faces.detach().cpu().numpy()
    world = transform_points(normalized, scene.normalized_to_world)
    world_faces = face_np[:, [0, 2, 1]] if \
        np.linalg.det(scene.normalized_to_world[:3, :3]) < 0 else face_np
    mesh_dir = run / "mesh"
    normalized_mesh = mesh_dir / f"{prefix}_normalized.ply"
    world_mesh = mesh_dir / f"{prefix}_world.ply"
    write_ply(normalized_mesh, normalized, face_np)
    write_ply(world_mesh, world, world_faces)
    topology = topology_statistics(face_np, vertex_count=len(normalized))
    area = face_area(vertices, faces).detach().cpu().numpy()
    area_mean = float(area.mean()) if len(area) else float("nan")
    sliver_threshold = 1e-8 * area_mean if len(area) else float("nan")
    sliver = area <= sliver_threshold if len(area) else np.zeros(0, dtype=bool)
    topology.update(face_area=dict(
        minimum=float(area.min()) if len(area) else float("nan"),
        median=float(np.median(area)) if len(area) else float("nan"),
        mean=area_mean,
        relative_sliver_threshold=sliver_threshold,
        relative_sliver_faces=int(sliver.sum()),
        relative_sliver_area_fraction=float(area[sliver].sum() / area.sum())
        if len(area) and float(area.sum()) > 0 else 0.0))
    topology.update(normalized_bounds=[normalized.min(0).tolist(),
                                      normalized.max(0).tolist()] if len(normalized)
                    else None)
    eikonal = eikonal_diagnostics(
        model.sdf, model.cfg.bound, samples=a.eikonal_samples)
    normal = mesh_normal_diagnostics(model.sdf, vertices, faces)
    shape = sphere_shape_diagnostics(vertices, model.cfg.bound)
    decision = make_gate_checks(
        learned_train, learned_test, initial_train, initial_test,
        eikonal, normal, shape, topology)

    reference_path = None
    reference_sha256 = None
    geometry_audit = None
    reference_checkpoint_path = None
    reference_checkpoint_sha256 = None
    if a.stage in ("joint", "appearance"):
        reference_path = Path(a.reference_report)
        reference = json.loads(reference_path.read_text())
        decision = add_iou_preservation_checks(
            decision, learned_train, learned_test, reference,
            max_drop=a.max_iou_drop)
        reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if a.stage == "appearance":
        reference_checkpoint_path = Path(a.reference_checkpoint)
        reference_model = RootSplat(Config(**mcfg), device=a.device)
        reference_state = load_checkpoint(
            reference_checkpoint_path, reference_model,
            map_location=a.device, restore_rng=False)
        if int(reference_state.get("step", -1)) != bootstrap_iters:
            raise RuntimeError(
                "Reference checkpoint step does not equal bootstrap_iters")
        # The bootstrap never renders and therefore legitimately has no cached
        # topology. Compare representation parameters against that checkpoint,
        # then compare the final runtime cache against the first post-bootstrap
        # checkpoint, after topology has been materialized. Reload the final
        # model because export_mesh() above intentionally replaces its grid.
        parameter_audit = geometry_state_audit(
            model, reference_model, include_runtime=False)
        anchor_paths = sorted((run / "checkpoints").glob("step_*.pt"))
        anchor_paths = [path for path in anchor_paths
                        if bootstrap_iters < int(path.stem.split("_")[-1])
                        < expected_step]
        if not anchor_paths:
            raise RuntimeError(
                "Appearance gate requires an intermediate step checkpoint")
        anchor_path = anchor_paths[0]
        anchor_model = RootSplat(Config(**mcfg), device=a.device)
        anchor_state = load_checkpoint(
            anchor_path, anchor_model, map_location=a.device,
            restore_rng=False)
        final_model = RootSplat(Config(**mcfg), device=a.device)
        load_checkpoint(checkpoint, final_model, map_location=a.device,
                        restore_rng=False)
        stability_audit = geometry_state_audit(final_model, anchor_model)
        mismatches = [f"bootstrap_parameters:{value}"
                      for value in parameter_audit["mismatches"]]
        mismatches += [f"runtime_stability:{value}"
                       for value in stability_audit["mismatches"]]
        geometry_audit = dict(
            exact=bool(parameter_audit["exact"] and
                       stability_audit["exact"]),
            mismatch_count=len(mismatches), mismatches=mismatches[:64],
            bootstrap_parameter_audit=parameter_audit,
            runtime_anchor=str(anchor_path),
            runtime_anchor_step=int(anchor_state["step"]),
            runtime_stability_audit=stability_audit)
        decision["thresholds"]["geometry_checkpoint_exact"] = True
        decision["checks"]["geometry_checkpoint_exact"] = dict(
            value=bool(geometry_audit["exact"]), relation="is",
            threshold=True, passed=bool(geometry_audit["exact"]))
        decision["failures"] = [
            name for name, check in decision["checks"].items()
            if not check["passed"]]
        decision["status"] = "pass" if not decision["failures"] else "fail"
        reference_checkpoint_sha256 = hashlib.sha256(
            reference_checkpoint_path.read_bytes()).hexdigest()

    learned_images = {**learned_train_images, **learned_test_images}
    initial_images = {**initial_train_images, **initial_test_images}
    save_gate_montage(
        run / "renders" / f"{prefix}_montage.png",
        learned_images, initial_images,
        {i: scene.views[i].name for i in retain})
    payload = dict(
        schema="rootsplat.geometry_gate.v2", stage=a.stage,
        checkpoint=str(checkpoint),
        checkpoint_step=int(state["step"]), stride=int(a.stride),
        ray_samples=int(a.ray_samples), temperature=float(a.temperature),
        reference_report=str(reference_path) if reference_path else None,
        reference_report_sha256=reference_sha256,
        reference_checkpoint=str(reference_checkpoint_path)
            if reference_checkpoint_path else None,
        reference_checkpoint_sha256=reference_checkpoint_sha256,
        geometry_state_audit=geometry_audit,
        learned=dict(train=learned_train, heldout=learned_test,
                     train_per_view=learned_train_rows,
                     heldout_per_view=learned_test_rows),
        initial_sphere=dict(train=initial_train, heldout=initial_test,
                            train_per_view=initial_train_rows,
                            heldout_per_view=initial_test_rows),
        eikonal=eikonal, mesh_normal_degrees=normal,
        sphere_shape=shape, topology=topology, decision=decision,
        artifacts=dict(
            montage=str(run / "renders" / f"{prefix}_montage.png"),
            normalized_mesh=str(normalized_mesh),
            world_mesh=str(world_mesh)))
    output = run / "metrics" / f"{prefix}.json"
    save_json(output, payload)
    print(json.dumps(dict(
        status=decision["status"], failures=decision["failures"],
        learned_train_iou=learned_train["iou"],
        initial_train_iou=initial_train["iou"],
        learned_heldout_iou=learned_test.get("iou"),
        initial_heldout_iou=initial_test.get("iou"),
        eikonal_abs_mean=eikonal["mean"],
        mesh_normal_mean_degrees=normal["mean"],
        sphere_relative_rmse=shape.get("relative_rmse"),
        extent_aspect_ratio=shape.get("extent_aspect_ratio")), indent=2))
    print(f"Gate report: {output}")


if __name__ == "__main__":
    main()
