#!/usr/bin/env python3
"""Preregistered gate for correspondence-certified bounded SDF refinement."""
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.experiment import load_checkpoint, load_config
from rootsplat.tracks import TrackStore


def triangulate(observations, cameras):
    rows = []
    for view, uv01 in observations:
        camera = cameras[int(view)]
        u, v = uv01 * np.array([camera.W, camera.H], dtype=np.float64)
        K = camera.K.detach().cpu().numpy().astype(np.float64)
        R = camera.R.detach().cpu().numpy().astype(np.float64)
        t = camera.t.detach().cpu().numpy().astype(np.float64)
        P = K @ np.concatenate([R, t[:, None]], axis=1)
        rows.extend([u * P[2] - P[0], v * P[2] - P[1]])
    _u, _s, vh = np.linalg.svd(np.stack(rows), full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    return homogeneous[:3] / homogeneous[3]


def surface_distance(model, points, chunk=8192):
    values, eikonal = [], []
    for start in range(0, len(points), chunk):
        x = torch.as_tensor(points[start:start + chunk], dtype=torch.float32,
                            device=model.device)
        s, grad = model.sdf.s_and_grad(x, create_graph=False)
        norm = grad.norm(dim=-1).clamp_min(1e-6)
        values.append((s.abs() / norm).detach().cpu().numpy())
        eikonal.append((norm - 1.0).abs().detach().cpu().numpy())
    return np.concatenate(values), np.concatenate(eikonal)


def distribution(value):
    return dict(mean=float(value.mean()), median=float(np.median(value)),
                p90=float(np.quantile(value, .90)),
                p95=float(np.quantile(value, .95)))


def metric_iou(path, key):
    payload = json.loads(Path(path).read_text())["aggregate"]
    group = payload.get(key)
    if group is None:
        raise ValueError(f"{path} does not contain {key} mask metrics")
    return float(group["iou"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--reference-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tracks", type=int, default=20000)
    parser.add_argument("--min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--max-alpha-iou-drop", type=float, default=0.01)
    parser.add_argument("--max-root-iou-drop", type=float, default=0.02)
    parser.add_argument("--max-eikonal-mean", type=float, default=0.2)
    parser.add_argument("--max-eikonal-increase", type=float, default=0.02)
    parser.add_argument("--min-active-log-rows", type=int, default=10)
    args = parser.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.resolved.yaml")
    dcfg, mcfg = cfg["dataset"], cfg["model"]
    scene = DTUScene(
        dcfg["scene"], device=args.device,
        downscale=dcfg.get("downscale", .25),
        test_every=dcfg.get("test_every", 8),
        require_masks=True, require_scale_matrices=True)
    tracks = TrackStore(dcfg["tracks"], view_count=len(scene.views),
                        min_confidence=dcfg.get("track_min_confidence", 0.0))
    ids = np.asarray(sorted(tracks._index))
    if args.max_tracks > 0 and len(ids) > args.max_tracks:
        ids = np.random.default_rng(0).choice(ids, args.max_tracks, replace=False)
    points = []
    for track in ids:
        lo, hi = tracks._index[int(track)]
        point = tracks.point3d[lo] if tracks.point3d is not None else triangulate(
            list(zip(tracks.view_id[lo:hi], tracks.uv01[lo:hi])),
            scene.view_cameras)
        if point is not None and np.isfinite(point).all() \
                and np.max(np.abs(point)) <= float(mcfg["bound"]):
            points.append(point)
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 500:
        raise RuntimeError("Fewer than 500 in-bound calibrated track points remain")

    reference = RootSplat(Config(**mcfg), device=args.device)
    candidate = RootSplat(Config(**mcfg), device=args.device)
    load_checkpoint(args.reference_checkpoint, reference,
                    map_location=args.device, restore_rng=False)
    load_checkpoint(args.candidate_checkpoint, candidate,
                    map_location=args.device, restore_rng=False)
    reference.eval(); candidate.eval()
    reference_distance, reference_eikonal = surface_distance(reference, points)
    candidate_distance, candidate_eikonal = surface_distance(candidate, points)
    base_exact = all(
        torch.equal(reference.sdf.base.state_dict()[key].cpu(), value.cpu())
        for key, value in candidate.sdf.base.state_dict().items())
    with torch.no_grad():
        x = torch.as_tensor(points, dtype=torch.float32, device=args.device)
        correction = candidate.sdf(x) - candidate.sdf.base(x)
        correction_max = float(correction.abs().max())

    reference_alpha = metric_iou(args.reference_metrics, "alpha_mask")
    candidate_alpha = metric_iou(args.candidate_metrics, "alpha_mask")
    reference_root = metric_iou(args.reference_metrics, "root_mask")
    candidate_root = metric_iou(args.candidate_metrics, "root_mask")
    active_rows = 0
    log_path = run / "training.jsonl"
    for line in log_path.read_text().splitlines():
        row = json.loads(line)
        if float(row.get("track_pairs", 0.0)) > 0 or \
                float(row.get("track_surface_tracks", 0.0)) > 0:
            active_rows += 1
    ratio = float(candidate_distance.mean() /
                  max(reference_distance.mean(), 1e-12))
    displacement_limit = float(mcfg["max_surface_displacement"])
    checks = dict(
        frozen_base_exact=dict(value=bool(base_exact), relation="==",
                               threshold=True, passed=bool(base_exact)),
        track_surface_mean_ratio=dict(
            value=ratio, relation="<=",
            threshold=1.0 - args.min_relative_improvement,
            passed=ratio <= 1.0 - args.min_relative_improvement),
        alpha_iou_preservation=dict(
            value=candidate_alpha - reference_alpha, relation=">=",
            threshold=-args.max_alpha_iou_drop,
            passed=candidate_alpha - reference_alpha >= -args.max_alpha_iou_drop),
        root_iou_preservation=dict(
            value=candidate_root - reference_root, relation=">=",
            threshold=-args.max_root_iou_drop,
            passed=candidate_root - reference_root >= -args.max_root_iou_drop),
        eikonal_abs_mean=dict(
            value=float(candidate_eikonal.mean()), relation="<=",
            threshold=args.max_eikonal_mean,
            passed=float(candidate_eikonal.mean()) <= args.max_eikonal_mean),
        eikonal_preservation=dict(
            value=float(candidate_eikonal.mean() - reference_eikonal.mean()),
            relation="<=", threshold=args.max_eikonal_increase,
            passed=float(candidate_eikonal.mean() - reference_eikonal.mean()) <=
            args.max_eikonal_increase),
        correction_bound=dict(
            value=correction_max, relation="<=",
            threshold=displacement_limit + 1e-6,
            passed=correction_max <= displacement_limit + 1e-6),
        active_track_log_rows=dict(
            value=int(active_rows), relation=">=",
            threshold=int(args.min_active_log_rows),
            passed=active_rows >= args.min_active_log_rows))
    failures = [key for key, check in checks.items() if not check["passed"]]
    report = dict(
        schema="rootsplat.correspondence_surface_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        checks=checks, evaluated_tracks=int(len(points)),
        reference_surface_distance=distribution(reference_distance),
        reference_eikonal_abs=distribution(reference_eikonal),
        candidate_surface_distance=distribution(candidate_distance),
        candidate_eikonal_abs=distribution(candidate_eikonal),
        render_masks=dict(
            reference_alpha_iou=reference_alpha, candidate_alpha_iou=candidate_alpha,
            reference_root_iou=reference_root, candidate_root_iou=candidate_root))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
