#!/usr/bin/env python3
"""Validate a DTU scene and write a deterministic split manifest."""
from pathlib import Path
import argparse
import json

from rootsplat.data import DTUScene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--downscale", type=float, default=0.25)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--priors-dir")
    p.add_argument("--depth-scale", type=float, default=1.0)
    p.add_argument("--depth-type", choices=("ray", "z"), default="ray")
    p.add_argument("--normal-space", choices=("world", "camera"), default="world")
    p.add_argument("--require-masks", action="store_true")
    p.add_argument("--require-scale-matrices", action="store_true")
    a = p.parse_args()
    scene = DTUScene(a.scene, device="cpu", downscale=a.downscale,
                     test_every=a.test_every, priors_dir=a.priors_dir,
                     depth_scale=a.depth_scale, depth_type=a.depth_type,
                     normal_space=a.normal_space,
                     require_masks=a.require_masks,
                     require_scale_matrices=a.require_scale_matrices)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    payload = scene.write_manifest(out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
