#!/usr/bin/env python3
"""Fail-fast coordinate-frame and silhouette gate for a COLMAP surface."""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.data import DTUScene
from rootsplat.initialization import (filter_oriented_surface_by_masks,
                                      load_oriented_surface)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--surface-space", choices=("world", "normalized"),
                        default="normalized")
    parser.add_argument("--output", required=True)
    parser.add_argument("--downscale", type=float, default=0.125)
    parser.add_argument("--samples", type=int, default=250000)
    parser.add_argument("--mask-fraction", type=float, default=0.8)
    parser.add_argument("--min-views", type=int, default=4)
    parser.add_argument("--min-retained", type=int, default=10000)
    parser.add_argument("--min-retained-fraction", type=float, default=0.01)
    parser.add_argument("--bound", type=float, default=1.0)
    args = parser.parse_args()

    scene = DTUScene(args.scene, device="cpu", downscale=args.downscale,
                     test_every=8, require_masks=True,
                     require_scale_matrices=True)
    transform = scene.world_to_normalized if args.surface_space == "world" else None
    points, normals, loader = load_oriented_surface(
        args.surface, transform=transform, samples=args.samples,
        seed=0, bound=args.bound)
    points, normals, mask = filter_oriented_surface_by_masks(
        points, normals, scene.view_cameras,
        [view.mask for view in scene.views],
        min_fraction=args.mask_fraction, min_views=args.min_views)
    max_abs = np.max(np.abs(points), axis=-1)
    extents = points.max(0) - points.min(0)
    checks = dict(
        retained_samples=dict(value=int(len(points)), relation=">=",
                              threshold=int(args.min_retained),
                              passed=len(points) >= args.min_retained),
        retained_fraction=dict(value=float(mask["retained_fraction"]),
                               relation=">=",
                               threshold=float(args.min_retained_fraction),
                               passed=mask["retained_fraction"] >=
                               args.min_retained_fraction),
        normalized_bound_p99=dict(value=float(np.quantile(max_abs, .99)),
                                  relation="<=", threshold=float(args.bound),
                                  passed=float(np.quantile(max_abs, .99)) <= args.bound),
        nondegenerate_extent=dict(value=float(extents.min()), relation=">=",
                                  threshold=0.1,
                                  passed=float(extents.min()) >= 0.1))
    failures = [name for name, check in checks.items() if not check["passed"]]
    report = dict(
        schema="rootsplat.colmap_initializer_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        checks=checks, loader=loader, mask_filter=mask,
        filtered_bounds=[points.min(0).tolist(), points.max(0).tolist()],
        surface_space=args.surface_space)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
