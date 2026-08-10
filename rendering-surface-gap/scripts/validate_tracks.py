#!/usr/bin/env python3
"""Calibrated geometric gate for a RootSplat track cache."""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.data import DTUScene
from rootsplat.tracks import TrackStore, save_tracks


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
    x = vh[-1]
    if abs(x[3]) < 1e-12:
        return None
    return x[:3] / x[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filtered-output",
                        help="Write only calibrated, mask-supported tracks")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Use 1.0 so reprojection gates are in source pixels")
    parser.add_argument("--max-tracks", type=int, default=100_000)
    parser.add_argument("--min-tracks", type=int, default=500)
    parser.add_argument("--max-median-reprojection", type=float, default=1.5)
    parser.add_argument("--max-p95-reprojection", type=float, default=5.0)
    parser.add_argument("--min-positive-depth", type=float, default=0.98)
    parser.add_argument("--min-median-ray-angle", type=float, default=1.0)
    parser.add_argument("--min-mask-support", type=float, default=0.8)
    parser.add_argument("--min-acceptance-fraction", type=float, default=0.01)
    parser.add_argument(
        "--train-only", action="store_true",
        help="Triangulate and export using training views only; required for NVS")
    args = parser.parse_args()

    scene = DTUScene(args.scene, device="cpu", downscale=args.downscale,
                     test_every=8, require_masks=True,
                     require_scale_matrices=True)
    store = TrackStore(args.tracks, view_count=len(scene.views))
    ids = np.asarray(sorted(store._index))
    if len(ids) > args.max_tracks > 0:
        rng = np.random.default_rng(0)
        ids = rng.choice(ids, args.max_tracks, replace=False)
    errors, positive, angles, accepted = [], [], [], 0
    triangulated = 0
    accepted_observations, accepted_points = [], []
    for track in ids:
        lo, hi = store._index[int(track)]
        observation_ids = np.arange(lo, hi)
        if args.train_only:
            observation_ids = observation_ids[np.isin(
                store.view_id[observation_ids], scene.train_ids)]
        if len(observation_ids) < 2:
            continue
        observations = list(zip(
            store.view_id[observation_ids], store.uv01[observation_ids]))
        point = triangulate(observations, scene.view_cameras)
        if point is None or not np.isfinite(point).all():
            continue
        triangulated += 1
        rays, track_errors, track_positive, track_mask = [], [], [], []
        for observation_index, (view, uv01) in enumerate(observations):
            camera = scene.view_cameras[int(view)]
            K = camera.K.detach().cpu().numpy()
            R = camera.R.detach().cpu().numpy()
            t = camera.t.detach().cpu().numpy()
            x_cam = R @ point + t
            track_positive.append(float(x_cam[2] > 0))
            projected = K @ x_cam
            denominator = projected[2] if abs(projected[2]) > 1e-12 else 1e-12
            uv = projected[:2] / denominator
            target = uv01 * np.array([camera.W, camera.H])
            track_errors.append(float(np.linalg.norm(uv - target)))
            x = min(max(int(np.floor(target[0])), 0), camera.W - 1)
            y = min(max(int(np.floor(target[1])), 0), camera.H - 1)
            mask = scene.views[int(view)].mask
            track_mask.append(float(mask is not None and mask[y, x] > 0.5))
            center = camera.center.detach().cpu().numpy()
            ray = point - center
            ray /= np.linalg.norm(ray).clip(1e-12)
            rays.append(ray)
        ray_angle = 0.0
        if len(rays) >= 2:
            cosine = np.clip(np.asarray(rays) @ np.asarray(rays).T, -1.0, 1.0)
            upper = np.triu_indices(len(rays), 1)
            ray_angle = float(np.degrees(np.arccos(cosine[upper])).max())
        track_errors = np.asarray(track_errors)
        track_positive = np.asarray(track_positive)
        track_mask = np.asarray(track_mask)
        track_ok = float(np.median(track_errors)) <= args.max_median_reprojection \
            and float(np.quantile(track_errors, .95)) <= args.max_p95_reprojection \
            and float(track_positive.mean()) >= args.min_positive_depth \
            and ray_angle >= args.min_median_ray_angle \
            and float(track_mask.mean()) >= args.min_mask_support
        if not track_ok:
            continue
        errors.extend(track_errors.tolist())
        positive.extend(track_positive.tolist())
        angles.append(ray_angle)
        accepted_observations.extend(observation_ids.tolist())
        accepted_points.extend([point.tolist()] * len(observation_ids))
        accepted += 1
    if not errors:
        raise RuntimeError("No tracks could be triangulated")
    errors, positive, angles = map(np.asarray, (errors, positive, angles))
    acceptance_fraction = accepted / max(triangulated, 1)
    checks = dict(
        track_count=dict(value=int(accepted), relation=">=",
                         threshold=int(args.min_tracks),
                         passed=accepted >= args.min_tracks),
        acceptance_fraction=dict(value=float(acceptance_fraction), relation=">=",
                         threshold=args.min_acceptance_fraction,
                         passed=acceptance_fraction >= args.min_acceptance_fraction),
        reprojection_median_pixels=dict(value=float(np.median(errors)), relation="<=",
                         threshold=args.max_median_reprojection,
                         passed=float(np.median(errors)) <= args.max_median_reprojection),
        reprojection_p95_pixels=dict(value=float(np.quantile(errors, .95)), relation="<=",
                         threshold=args.max_p95_reprojection,
                         passed=float(np.quantile(errors, .95)) <= args.max_p95_reprojection),
        positive_depth_fraction=dict(value=float(positive.mean()), relation=">=",
                         threshold=args.min_positive_depth,
                         passed=float(positive.mean()) >= args.min_positive_depth),
        median_max_ray_angle_degrees=dict(value=float(np.median(angles)), relation=">=",
                         threshold=args.min_median_ray_angle,
                         passed=float(np.median(angles)) >= args.min_median_ray_angle))
    failures = [name for name, check in checks.items() if not check["passed"]]
    report = dict(schema="rootsplat.track_geometry_gate.v1",
                  status="pass" if not failures else "fail", failures=failures,
                  train_only=bool(args.train_only),
                  checks=checks, triangulated_tracks=int(triangulated),
                  accepted_tracks=int(accepted), track_cache=store.summary())
    if args.filtered_output:
        if not accepted_observations:
            raise RuntimeError("No observations survive calibrated track filtering")
        selected = np.asarray(accepted_observations, dtype=np.int64)
        original = store.track_id[selected]
        _unique, remapped = np.unique(original, return_inverse=True)
        save_tracks(
            args.filtered_output, remapped, store.view_id[selected],
            store.uv01[selected], store.confidence[selected],
            store.sigma01[selected],
            point3d=np.asarray(accepted_points, dtype=np.float32),
            metadata={**store.metadata, "calibrated_filter": True,
                      "train_only": bool(args.train_only),
                      "geometry_gate": str(Path(args.output).resolve())})
        report["filtered_output"] = str(Path(args.filtered_output).resolve())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
