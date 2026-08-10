#!/usr/bin/env python3
"""Train RootSplat on one prepared DTU scene."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import math
import shutil

import numpy as np
import torch
from tqdm import trange

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.artifacts import save_json
from rootsplat.evaluation import export_current_mesh
from rootsplat.experiment import (MetricLogger, load_checkpoint, load_config,
                                  save_checkpoint, save_yaml, seed_everything)
from rootsplat.initialization import (filter_oriented_surface_by_masks,
                                      fit_surface_base_sdf,
                                      load_oriented_surface,
                                      orient_normals_to_cameras,
                                      verify_initializer_gate_report)
from rootsplat.tsdf import verify_tsdf_gate_report


def make_optimizer(model, train_cfg):
    """Create named parameter groups when stage-specific rates are declared."""
    keys = ("geometry_lr", "appearance_lr", "quadrature_lr", "background_lr")
    if not any(key in train_cfg for key in keys):
        return torch.optim.Adam(
            [{"params": list(model.parameters()), "name": "all",
              "initial_lr": float(model.cfg.lr)}], lr=model.cfg.lr)

    specifications = (
        ("sdf", model.sdf, "geometry_lr"),
        ("appearance", model.app, "appearance_lr"),
        ("quadrature", model.quad, "quadrature_lr"),
        ("background", model.background, "background_lr"),
    )
    groups = []
    for name, module, key in specifications:
        parameters = list(module.parameters())
        if not parameters:
            continue
        rate = float(train_cfg.get(key, model.cfg.lr))
        groups.append(dict(params=parameters, lr=rate, initial_lr=rate,
                           name=name))
    if not groups:
        raise RuntimeError("No trainable parameters were assigned to the optimizer")
    assigned = {id(p) for group in groups for p in group["params"]}
    expected = {id(p) for p in model.parameters()}
    if assigned != expected:
        raise RuntimeError("Optimizer parameter groups do not partition the model")
    return torch.optim.Adam(groups)


def update_learning_rates(optimizer, step, train_cfg):
    """Apply a reproducible absolute-step cosine decay, if requested."""
    schedule = str(train_cfg.get("lr_decay", "none")).lower()
    if schedule in ("", "none"):
        return
    if schedule != "cosine":
        raise ValueError(f"Unsupported training.lr_decay={schedule!r}")
    start = int(train_cfg.get("lr_decay_start", 0))
    end = int(train_cfg.get("lr_decay_end", start + 1))
    minimum = float(train_cfg.get("lr_min_ratio", 0.1))
    if end <= start or not 0.0 <= minimum <= 1.0:
        raise ValueError("Invalid cosine learning-rate schedule")
    progress = min(1.0, max(0.0, (float(step) - start) / (end - start)))
    factor = minimum + (1.0 - minimum) * 0.5 * \
        (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        base = float(group.get("initial_lr", group["lr"]))
        group["lr"] = base * factor


def learning_rate_terms(optimizer):
    result = {}
    aliases = dict(sdf="lr_sdf", appearance="lr_appearance",
                   quadrature="lr_quadrature", background="lr_background")
    for group in optimizer.param_groups:
        key = aliases.get(group.get("name"))
        if key:
            result[key] = float(group["lr"])
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--scene", help="Override dataset.scene in YAML")
    p.add_argument("--output", help="Override experiment.output in YAML")
    p.add_argument("--resume")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = dict(cfg.get("dataset", {}))
    train_cfg = dict(cfg.get("training", {}))
    model_cfg = dict(cfg.get("model", {}))
    exp_cfg = dict(cfg.get("experiment", {}))
    init_cfg = dict(cfg.get("initialization", {}))
    if args.scene:
        dataset_cfg["scene"] = args.scene
    if args.output:
        exp_cfg["output"] = args.output
    device = args.device or exp_cfg.get("device", "cuda")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    output = Path(exp_cfg.get("output", "runs/rootsplat"))
    output.mkdir(parents=True, exist_ok=True)
    seed = int(exp_cfg.get("seed", 0)); seed_everything(seed)

    scene = DTUScene(dataset_cfg["scene"], device=device,
                     downscale=dataset_cfg.get("downscale", 0.25),
                     test_every=dataset_cfg.get("test_every", 8),
                     priors_dir=dataset_cfg.get("priors_dir"),
                     depth_type=dataset_cfg.get("depth_type", "ray"),
                     normal_space=dataset_cfg.get("normal_space", "world"),
                     depth_scale=dataset_cfg.get("depth_scale", 1.0),
                     require_masks=dataset_cfg.get("require_masks", False),
                     require_scale_matrices=dataset_cfg.get(
                         "require_scale_matrices", False),
                     tracks_path=dataset_cfg.get("tracks"),
                     track_min_confidence=dataset_cfg.get(
                         "track_min_confidence", 0.0))
    variant = str(exp_cfg.get("variant", "rgb")).lower()
    has_depth = any(scene.views[i].depth is not None for i in scene.train_ids)
    has_mask = any(scene.views[i].mask is not None for i in scene.train_ids)
    if variant == "prior" and not has_depth:
        raise RuntimeError("variant=prior requires priors/depth files; use variant=rgb otherwise")

    model = RootSplat(Config(**model_cfg), device=device)
    if (model.cfg.weights.get("track_consensus", 0.0) > 0 or
            model.cfg.weights.get("track_reprojection", 0.0) > 0 or
            model.cfg.weights.get("track_surface", 0.0) > 0) and \
            scene.tracks is None:
        raise RuntimeError(
            "Track losses are enabled but dataset.tracks was not provided")
    if model.cfg.weights.get("track_surface", 0.0) > 0 and \
            scene.tracks.point3d is None:
        raise RuntimeError(
            "track_surface requires a calibrated track cache containing point3d")
    if model.cfg.weights.get("track_surface", 0.0) > 0 and \
            not bool(scene.tracks.metadata.get("train_only", False)):
        raise RuntimeError(
            "track_surface requires train-only triangulation to prevent test leakage")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = 0
    resume_provenance = None
    initializer_audit = None
    resume_optimizer = bool(exp_cfg.get("resume_optimizer", True))
    if args.resume:
        optimizer = make_optimizer(model, train_cfg)
        resume_path = Path(args.resume).resolve()
        checkpoint = load_checkpoint(resume_path, model,
                                     optimizer if resume_optimizer else None,
                                     map_location=device, generator=generator,
                                     restore_rng=True)
        start = int(checkpoint["step"])
        resume_provenance = dict(
            path=str(resume_path), step=start,
            sha256=hashlib.sha256(resume_path.read_bytes()).hexdigest(),
            optimizer_restored=resume_optimizer)
    else:
        initializer_path = init_cfg.get("surface")
        grid_path = init_cfg.get("grid_sdf") or model.cfg.base_sdf_grid
        if initializer_path and grid_path:
            raise RuntimeError(
                "Choose exactly one initializer: surface point fitting or grid_sdf")
        if model.cfg.bounded_residual and not initializer_path and not grid_path:
            raise RuntimeError(
                "model.bounded_residual requires initialization.surface, "
                "initialization.grid_sdf, or an explicit --resume checkpoint")
        if grid_path:
            if not model.cfg.base_sdf_grid:
                raise RuntimeError(
                    "initialization.grid_sdf requires model.base_sdf_grid")
            if Path(grid_path).resolve() != Path(model.cfg.base_sdf_grid).resolve():
                raise RuntimeError(
                    "initialization.grid_sdf and model.base_sdf_grid disagree")
            gate_path = init_cfg.get("grid_gate_report")
            if not gate_path:
                raise RuntimeError(
                    "A grid SDF is accepted only with initialization.grid_gate_report")
            grid_audit = verify_tsdf_gate_report(gate_path, grid_path)
            initializer_audit = dict(
                status="pass", failures=[], kind="gated_grid_sdf",
                grid=grid_audit,
                note=("The calibrated grid is loaded directly; no point-only "
                      "sphere fit is executed."))
        elif initializer_path:
            if not model.cfg.bounded_residual:
                raise RuntimeError(
                    "Surface initialization requires model.bounded_residual=true")
            gate_audit = None
            if init_cfg.get("gate_report"):
                gate_audit = verify_initializer_gate_report(
                    init_cfg["gate_report"], initializer_path,
                    expected_schema=init_cfg.get("gate_schema"))
            elif bool(init_cfg.get("require_gate_report", False)):
                raise RuntimeError(
                    "initialization.require_gate_report=true but no gate_report "
                    "was provided")
            space = str(init_cfg.get("surface_space", "world")).lower()
            if space not in ("world", "normalized"):
                raise ValueError(
                    "initialization.surface_space must be world or normalized")
            transform = scene.world_to_normalized if space == "world" else None
            points, normals, load_audit = load_oriented_surface(
                initializer_path, transform=transform,
                samples=int(init_cfg.get("samples", 250_000)), seed=seed,
                bound=model.cfg.bound)
            mask_audit = None
            if bool(init_cfg.get("filter_by_masks", True)):
                points, normals, mask_audit = filter_oriented_surface_by_masks(
                    points, normals, scene.view_cameras,
                    [view.mask for view in scene.views],
                    min_fraction=float(init_cfg.get(
                        "mask_filter_fraction", 0.8)),
                    min_views=int(init_cfg.get("mask_filter_min_views", 4)))
            centers = np.stack([
                camera.center.detach().cpu().numpy()
                for camera in scene.train_cameras], axis=0)
            normals = orient_normals_to_cameras(points, normals, centers)
            fit_audit = fit_surface_base_sdf(
                model.sdf, points, normals,
                iterations=int(init_cfg.get("iterations", 1000)),
                batch_size=int(init_cfg.get("batch_size", 8192)),
                lr=float(init_cfg.get("lr", 1e-3)),
                offset=float(init_cfg.get("offset", 0.01)),
                normal_weight=float(init_cfg.get("normal_weight", 0.1)),
                eikonal_weight=float(init_cfg.get("eikonal_weight", 0.1)),
                seed=seed)
            initializer_audit = dict(gate=gate_audit, loader=load_audit,
                                     mask_filter=mask_audit, fit=fit_audit,
                                     surface_space=space)
            initializer_checks = dict(
                fitted_surface_abs_p95=dict(
                    value=fit_audit["fitted_surface_abs_p95"], relation="<=",
                    threshold=float(init_cfg.get("max_surface_abs_p95", .03))),
                fitted_eikonal_abs_mean=dict(
                    value=fit_audit["fitted_eikonal_abs_mean"], relation="<=",
                    threshold=float(init_cfg.get("max_eikonal_abs_mean", .25))),
                fitted_normal_mean_degrees=dict(
                    value=fit_audit["fitted_normal_mean_degrees"], relation="<=",
                    threshold=float(init_cfg.get("max_normal_mean_degrees", 30.0))))
            for check in initializer_checks.values():
                check["passed"] = check["value"] <= check["threshold"]
            failures = [name for name, check in initializer_checks.items()
                        if not check["passed"]]
            initializer_audit.update(
                checks=initializer_checks,
                status="pass" if not failures else "fail",
                failures=failures)
            if failures:
                save_json(output / "metrics" / "initializer_failed.json",
                          initializer_audit)
                raise RuntimeError(
                    "Surface-to-SDF initialization gate failed: " +
                    ", ".join(failures))
        optimizer = make_optimizer(model, train_cfg)
    if bool(exp_cfg.get("resume_required", False)) and not args.resume:
        raise RuntimeError("This experiment requires an explicit --resume checkpoint")
    required_step = exp_cfg.get("resume_step_required")
    if required_step is not None and start != int(required_step):
        raise RuntimeError(
            f"Resume checkpoint step {start} != required step {int(required_step)}")
    resolved = dict(experiment={**exp_cfg, "variant": variant, "device": device,
                                "seed": seed, "output": str(output),
                                "resume_provenance": resume_provenance},
                    dataset=dataset_cfg, initialization=init_cfg,
                    model=asdict(model.cfg), training=train_cfg)
    save_yaml(output / "config.resolved.yaml", resolved)
    scene.write_manifest(output / "manifest.json")
    if initializer_audit is not None:
        save_json(output / "metrics" / "initializer.json", initializer_audit)
        save_checkpoint(output / "checkpoints" / "initializer.pt", model,
                        optimizer, 0, resolved, scene.summary(), generator)
    shutil.copy2(args.config, output / "config.input.yaml")
    logger = MetricLogger(output, append=start > 0)

    # RGB reconstructions also need a geometry-only stage: calibrated masks
    # bootstrap the visual hull before appearance is allowed to explain images.
    # A configured bootstrap is skipped only when neither masks nor enabled
    # depth priors exist.
    bootstrap_enabled = model.cfg.bootstrap_iters > 0 and \
        (has_mask or (variant == "prior" and has_depth))
    total = model.cfg.joint_iters + \
        (model.cfg.bootstrap_iters if bootstrap_enabled else 0)
    if total <= 0:
        raise RuntimeError(
            "Training schedule has no executable updates: enable bootstrap "
            "supervision or set joint_iters > 0")
    if start < 0 or start > total:
        raise RuntimeError(
            f"Checkpoint step {start} lies outside configured schedule [0,{total}]")
    patch = int(train_cfg.get("patch_size", 32))
    fg_min = float(train_cfg.get("foreground_min", 0.1))
    boundary_fraction = float(train_cfg.get("boundary_fraction", 0.0))
    log_every = int(train_cfg.get("log_every", 10))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 500))
    fixed_view_id = train_cfg.get("fixed_view_id")
    if fixed_view_id is not None:
        fixed_view_id = int(fixed_view_id)
        if fixed_view_id not in scene.train_ids:
            raise ValueError(
                f"training.fixed_view_id={fixed_view_id} is not a training view")
    bar = trange(start, total, initial=start, total=total, dynamic_ncols=True)
    for step in bar:
        mode = model.optimization_mode(step, bootstrap_enabled)
        prefix = "appearance" if mode == "appearance" else "geometry"
        stage_patch = int(train_cfg.get(f"{prefix}_patch_size", patch))
        stage_fg_min = float(train_cfg.get(
            f"{prefix}_foreground_min", fg_min))
        stage_boundary_fraction = float(train_cfg.get(
            f"{prefix}_boundary_fraction", boundary_fraction))
        update_learning_rates(optimizer, step, train_cfg)
        batch = scene.sample_patch(view_id=fixed_view_id,
                                   patch_size=stage_patch,
                                   foreground_min=stage_fg_min,
                                   boundary_fraction=stage_boundary_fraction,
                                   generator=generator,
                                   track_max_tracks=train_cfg.get(
                                       "track_max_tracks", 64),
                                   track_max_views=train_cfg.get(
                                       "track_max_views", 4))
        if variant == "rgb":
            batch["depth"] = None
            batch["normal"] = None
        loss, terms = model.training_step(batch, step, optimizer)
        value = float(loss)
        if not math.isfinite(value):
            save_checkpoint(output / "checkpoints" / "nonfinite.pt", model,
                            optimizer, step, resolved, scene.summary(), generator)
            raise FloatingPointError(f"Non-finite loss at step {step}")
        if step % log_every == 0:
            row = dict(step=step, loss=value, view_id=batch["view_id"],
                       **terms, **learning_rate_terms(optimizer))
            logger.write(row)
            bar.set_postfix(loss=f"{value:.4g}", faces=0 if model.F is None else len(model.F))
        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            save_checkpoint(output / "checkpoints" / f"step_{step + 1:06d}.pt",
                            model, optimizer, step + 1, resolved, scene.summary(),
                            generator)
            save_checkpoint(output / "checkpoints" / "latest.pt", model,
                            optimizer, step + 1, resolved, scene.summary(), generator)

    save_checkpoint(output / "checkpoints" / "final.pt", model, optimizer,
                    total, resolved, scene.summary(), generator)
    stats = export_current_mesh(
        model, output / "mesh" / "final.ply",
        resolution=train_cfg.get("export_resolution"),
        cameras=scene.train_cameras,
        normalized_to_world=scene.normalized_to_world,
        normalized_path=output / "mesh" / "final_normalized.ply")
    save_json(output / "metrics" / "topology_train.json", stats)
    print(f"Training complete: {output}")


if __name__ == "__main__":
    main()
