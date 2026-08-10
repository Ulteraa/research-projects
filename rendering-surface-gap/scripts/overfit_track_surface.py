#!/usr/bin/env python3
"""Fast capacity gate for calibrated track-to-SDF supervision.

This intentionally bypasses rendering, topology refreshes, and every competing
loss.  It answers one narrow question: can the bounded residual attached to a
frozen initializer reduce distance to a fixed set of calibrated 3D tracks?
The resulting checkpoint is diagnostic and must not be used as a paper model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

import numpy as np
import torch

from rootsplat import Config, RootSplat
from rootsplat import losses as L
from rootsplat.experiment import (load_checkpoint, load_config,
                                  save_checkpoint, save_yaml, seed_everything)
from rootsplat.tracks import TrackStore


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p90=float(np.quantile(values, .90)),
                p95=float(np.quantile(values, .95)),
                max=float(values.max()))


def evaluate(model, points, chunk=8192):
    distances, eikonal, corrections = [], [], []
    for start in range(0, len(points), chunk):
        x = torch.as_tensor(points[start:start + chunk], dtype=torch.float32,
                            device=model.device)
        value, gradient = model.sdf.s_and_grad(x, create_graph=False)
        norm = gradient.norm(dim=-1).clamp_min(1e-6)
        distances.append((value.abs() / norm).detach().cpu().numpy())
        eikonal.append((norm - 1.0).abs().detach().cpu().numpy())
        with torch.no_grad():
            corrections.append(
                (model.sdf(x) - model.sdf.base(x)).abs().cpu().numpy())
    return dict(distance=distribution(np.concatenate(distances)),
                eikonal_abs=distribution(np.concatenate(eikonal)),
                correction_abs=distribution(np.concatenate(corrections)))


def point_distances(model, points, chunk=8192):
    values = []
    for start in range(0, len(points), chunk):
        x = torch.as_tensor(points[start:start + chunk], dtype=torch.float32,
                            device=model.device)
        value, gradient = model.sdf.s_and_grad(x, create_graph=False)
        values.append((value.abs() / gradient.norm(dim=-1).clamp_min(1e-6))
                      .detach().cpu().numpy())
    return np.concatenate(values)


def spatially_balance(points, confidence, bound, voxel_size, max_per_voxel,
                      rng):
    if max_per_voxel <= 0:
        return points, confidence
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    cells = np.floor((points + float(bound)) / float(voxel_size)).astype(np.int64)
    counts, selected = {}, []
    for index in rng.permutation(len(points)):
        key = tuple(cells[index].tolist())
        count = counts.get(key, 0)
        if count >= int(max_per_voxel):
            continue
        counts[key] = count + 1
        selected.append(int(index))
    selected = np.asarray(selected, dtype=np.int64)
    return points[selected], confidence[selected]


def gradient_norm(parameters):
    squared = []
    for parameter in parameters:
        if parameter.grad is not None:
            squared.append(parameter.grad.detach().float().square().sum())
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-initial-distance", type=float, default=0.0,
                        help="Keep only anchors reachable by the bounded residual")
    parser.add_argument("--voxel-size", type=float, default=.02)
    parser.add_argument("--max-per-voxel", type=int, default=0)
    parser.add_argument("--eikonal-weight", type=float, default=0.0)
    parser.add_argument("--residual-gradient-weight", type=float, default=0.0)
    parser.add_argument("--jitter", type=float, default=.01)
    parser.add_argument("--max-train-ratio", type=float, default=.90)
    parser.add_argument("--max-validation-ratio", type=float, default=.95)
    parser.add_argument("--max-eikonal-increase", type=float, default=.02)
    parser.add_argument("--max-saturation-fraction", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.steps <= 0 or args.batch_size <= 0 or args.max_points < 1000:
        raise ValueError("Invalid point-only optimization budget")
    if not 0.05 <= args.validation_fraction <= 0.5:
        raise ValueError("validation_fraction must lie in [0.05,0.5]")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    for name in ("eikonal_weight", "residual_gradient_weight", "jitter"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0 < args.max_train_ratio <= 1 or \
            not 0 < args.max_validation_ratio <= 1:
        raise ValueError("distance-ratio thresholds must lie in (0,1]")
    if args.max_eikonal_increase < 0 or \
            not 0 <= args.max_saturation_fraction <= 1:
        raise ValueError("invalid preservation thresholds")

    seed_everything(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    (output / "metrics").mkdir()

    cfg = load_config(args.config)
    model_cfg = dict(cfg["model"])
    if not bool(model_cfg.get("bounded_residual", False)):
        raise RuntimeError("Point-only gate requires a bounded residual SDF")
    model = RootSplat(Config(**model_cfg), device=args.device)
    reference_path = Path(args.reference_checkpoint).resolve()
    checkpoint = load_checkpoint(reference_path, model, map_location=args.device,
                                 restore_rng=False)
    reference_step = int(checkpoint["step"])
    base_reference = {
        key: value.detach().cpu().clone()
        for key, value in model.sdf.base.state_dict().items()}

    store = TrackStore(args.tracks, min_confidence=.5)
    if store.point3d is None:
        raise RuntimeError("Point-only gate requires calibrated point3d tracks")
    ids = np.asarray(sorted(store._index), dtype=np.int64)
    points = np.asarray([store.point3d[store._index[int(track)][0]]
                         for track in ids], dtype=np.float32)
    confidence = np.asarray([
        store.confidence[slice(*store._index[int(track)])].mean()
        for track in ids], dtype=np.float32)
    bound = float(model_cfg["bound"])
    keep = np.isfinite(points).all(-1) & (np.abs(points).max(-1) <= bound)
    points, confidence = points[keep], confidence[keep]
    rng = np.random.default_rng(args.seed)
    source_point_count = int(len(points))
    if args.max_initial_distance > 0:
        initial_distance = point_distances(model, points)
        reachable = initial_distance <= float(args.max_initial_distance)
        points, confidence = points[reachable], confidence[reachable]
    reachable_point_count = int(len(points))
    points, confidence = spatially_balance(
        points, confidence, bound, args.voxel_size, args.max_per_voxel, rng)
    balanced_point_count = int(len(points))
    order = rng.permutation(len(points))[:args.max_points]
    points, confidence = points[order], confidence[order]
    validation_count = max(1, int(round(len(points) * args.validation_fraction)))
    validation_points = points[:validation_count]
    train_points = points[validation_count:]
    train_confidence = confidence[validation_count:]
    if len(train_points) < 500 or len(validation_points) < 100:
        raise RuntimeError("Too few in-bound points for a deterministic split")

    initial_train = evaluate(model, train_points)
    initial_validation = evaluate(model, validation_points)

    model.requires_grad_(False)
    model.sdf.residual.requires_grad_(True)
    parameters = list(model.sdf.residual.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    trace = []
    first_gradient_norm = None
    for step in range(args.steps):
        sample = rng.integers(0, len(train_points),
                              size=min(args.batch_size, len(train_points)))
        x = torch.as_tensor(train_points[sample], dtype=torch.float32,
                            device=args.device)
        weight = torch.as_tensor(train_confidence[sample], dtype=torch.float32,
                                 device=args.device)
        track_loss = L.track_surface(
            model.sdf, x, weight,
            distance_scale=float(model_cfg.get("track_distance_scale", .02)))
        probe = (x + torch.randn_like(x) * float(args.jitter)).clamp(
            -bound, bound)
        eikonal_loss = L.eikonal(model.sdf, probe) \
            if args.eikonal_weight > 0 else x.new_zeros(())
        gradient_loss = L.residual_gradient(model.sdf, probe) \
            if args.residual_gradient_weight > 0 else x.new_zeros(())
        loss = track_loss + args.eikonal_weight * eikonal_loss + \
            args.residual_gradient_weight * gradient_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = gradient_norm(parameters)
        if first_gradient_norm is None:
            first_gradient_norm = norm
        torch.nn.utils.clip_grad_norm_(parameters, 5.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            row = dict(step=step + 1, loss=float(loss.detach()),
                       track_surface=float(track_loss.detach()),
                       eikonal=float(eikonal_loss.detach()),
                       residual_gradient=float(gradient_loss.detach()),
                       gradient_norm=float(norm))
            trace.append(row)
            print(json.dumps(row), flush=True)

    final_train = evaluate(model, train_points)
    final_validation = evaluate(model, validation_points)
    base_exact = all(torch.equal(value, model.sdf.base.state_dict()[key].cpu())
                     for key, value in base_reference.items())
    train_ratio = final_train["distance"]["mean"] / \
        max(initial_train["distance"]["mean"], 1e-12)
    validation_ratio = final_validation["distance"]["mean"] / \
        max(initial_validation["distance"]["mean"], 1e-12)
    correction_max = max(final_train["correction_abs"]["max"],
                         final_validation["correction_abs"]["max"])
    displacement_limit = float(model_cfg["max_surface_displacement"])
    saturation_threshold = .95 * displacement_limit
    correction_values = []
    for values in (train_points, validation_points):
        for start in range(0, len(values), 8192):
            x = torch.as_tensor(values[start:start + 8192], dtype=torch.float32,
                                device=args.device)
            with torch.no_grad():
                correction_values.append(
                    (model.sdf(x) - model.sdf.base(x)).abs().cpu().numpy())
    saturation_fraction = float(
        (np.concatenate(correction_values) >= saturation_threshold).mean())
    eikonal_increase = final_validation["eikonal_abs"]["mean"] - \
        initial_validation["eikonal_abs"]["mean"]
    checks = dict(
        initial_gradient_nonzero=dict(
            value=float(first_gradient_norm or 0.0), relation=">=",
            threshold=1e-8, passed=float(first_gradient_norm or 0.0) >= 1e-8),
        train_distance_ratio=dict(
            value=float(train_ratio), relation="<=",
            threshold=float(args.max_train_ratio),
            passed=train_ratio <= args.max_train_ratio),
        validation_distance_ratio=dict(
            value=float(validation_ratio), relation="<=",
            threshold=float(args.max_validation_ratio),
            passed=validation_ratio <= args.max_validation_ratio),
        validation_eikonal_preservation=dict(
            value=float(eikonal_increase), relation="<=",
            threshold=float(args.max_eikonal_increase),
            passed=eikonal_increase <= args.max_eikonal_increase),
        correction_saturation_fraction=dict(
            value=saturation_fraction, relation="<=",
            threshold=float(args.max_saturation_fraction),
            passed=saturation_fraction <= args.max_saturation_fraction),
        frozen_base_exact=dict(value=bool(base_exact), relation="==",
                               threshold=True, passed=bool(base_exact)),
        correction_bound=dict(
            value=float(correction_max), relation="<=",
            threshold=displacement_limit + 1e-6,
            passed=correction_max <= displacement_limit + 1e-6))
    failures = [name for name, check in checks.items() if not check["passed"]]
    report = dict(
        schema="rootsplat.track_surface_capacity_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        checks=checks, reference_checkpoint=str(reference_path),
        reference_checkpoint_sha256=hashlib.sha256(
            reference_path.read_bytes()).hexdigest(),
        reference_step=reference_step,
        tracks=store.summary(), points=dict(source=source_point_count,
            reachable=reachable_point_count,
            reachable_fraction=float(reachable_point_count /
                                     max(source_point_count, 1)),
            spatially_balanced=balanced_point_count, total=int(len(points)),
            train=int(len(train_points)), validation=int(len(validation_points))),
        initial=dict(train=initial_train, validation=initial_validation),
        final=dict(train=final_train, validation=final_validation), trace=trace,
        options=vars(args))
    report_path = output / "metrics" / "track_surface_capacity.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    resolved = dict(cfg, diagnostic=dict(
        schema=report["schema"], reference_checkpoint=str(reference_path),
        tracks=str(Path(args.tracks).resolve()), options=vars(args)))
    save_yaml(output / "config.resolved.yaml", resolved)
    save_checkpoint(output / "checkpoints" / "final.pt", model, optimizer,
                    args.steps, resolved, store.summary(), generator)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
