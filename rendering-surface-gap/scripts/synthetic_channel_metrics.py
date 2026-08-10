#!/usr/bin/env python3
"""Evaluate analytic sphere geometry in saved per-pixel render channels.

Unlike ``synthetic_geometry_metrics.py``, which evaluates the exported mesh,
this script measures the certified roots and continuous SDF-gradient normals.
Reporting both is necessary: a smooth extracted mesh can coexist with a noisy
differential field used by the renderer.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np


def normalize(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return dict(mean=None, median=None, p90=None, p95=None, p99=None, max=None)
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x, .90)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), max=float(x.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--radius", type=float, default=.5)
    p.add_argument("--camera-radius", type=float, default=1.7)
    p.add_argument("--view-index", type=int, default=0)
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--focal-scale", type=float, default=.9)
    a = p.parse_args()

    z = np.load(a.channels)
    alpha = np.asarray(z["alpha"])
    H, W = alpha.shape
    az = 2 * np.pi * a.view_index / a.views
    el = .18 * np.sin(2 * az)
    C = np.array([a.camera_radius * np.cos(az) * np.cos(el),
                  a.camera_radius * np.sin(el),
                  a.camera_radius * np.sin(az) * np.cos(el)])
    forward = normalize((-C)[None])[0]
    up = np.array([0., 1., 0.])
    right = normalize(np.cross(up, forward)[None])[0]
    R = np.stack([right, np.cross(forward, right), forward], 0)
    focal = a.focal_scale * W
    K = np.array([[focal, 0., W / 2], [0., focal, H / 2], [0., 0., 1.]])
    yy, xx = np.meshgrid(np.arange(H) + .5, np.arange(W) + .5, indexing="ij")
    hom = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3)
    d = normalize(hom @ np.linalg.inv(K).T @ R).reshape(H, W, 3)

    b = np.sum(d * C, axis=-1)
    c = C @ C - a.radius ** 2
    disc = b * b - c
    gt = disc >= 0
    true_depth = -b - np.sqrt(np.maximum(disc, 0))
    gt &= true_depth > 0
    inferred_gt = np.min(z["gt_rgb"], axis=-1) < 1.0 - 1e-6

    depth = np.asarray(z["depth"])
    proposal = np.asarray(z["proposal_depth"])
    valid = np.asarray(z["valid"], dtype=bool)
    root = C[None, None] + depth[..., None] * d
    proposed = C[None, None] + proposal[..., None] * d
    radius_root = np.linalg.norm(root, axis=-1)
    radius_proposal = np.linalg.norm(proposed, axis=-1)
    certified = valid & gt

    n = normalize(np.asarray(z["normal"])[certified])
    radial = normalize(root[certified])
    dot = np.sum(n * radial, axis=-1).clip(-1, 1)
    oriented = np.degrees(np.arccos(dot))
    unoriented = np.degrees(np.arccos(np.abs(dot)))

    result = {
        "channels": str(a.channels),
        "image_size": [int(H), int(W)],
        "analytic_mask_matches_saved_gt": bool(np.array_equal(gt, inferred_gt)),
        "ground_truth_foreground_pixels": int(gt.sum()),
        "silhouette_equivalent_radius_pixels": {
            "ground_truth": float(math.sqrt(gt.sum() / math.pi)),
            **{f"alpha_gt_{threshold:g}":
               float(math.sqrt(np.sum(alpha > threshold) / math.pi))
               for threshold in (.01, .1, .25, .5, .75, .9, .99)}
        },
        "certification": {
            "valid_foreground_pixels": int(certified.sum()),
            "valid_foreground_fraction": float(certified.sum() / max(gt.sum(), 1)),
            "clipped_foreground_fraction":
                float(np.asarray(z["clipped"], dtype=bool)[gt].mean()),
            "confidence_foreground_mean": float(np.asarray(z["confidence"])[gt].mean())
        },
        "certified_root": {
            "signed_radial_error_mean": float((radius_root[certified] - a.radius).mean()),
            "absolute_radial_error": stats(np.abs(radius_root[certified] - a.radius)),
            "absolute_depth_error": stats(np.abs(depth[certified] - true_depth[certified]))
        },
        "proposal_foreground": {
            "signed_radial_error_mean": float((radius_proposal[gt] - a.radius).mean()),
            "absolute_radial_error": stats(np.abs(radius_proposal[gt] - a.radius)),
            "absolute_depth_error": stats(np.abs(proposal[gt] - true_depth[gt]))
        },
        "sdf_normal_oriented_degrees": stats(oriented),
        "sdf_normal_unoriented_degrees": stats(unoriented),
        "ray_derivative_absolute": stats(np.abs(np.asarray(z["ray_derivative"])[gt]))
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
