#!/usr/bin/env python3
"""Render held-out views, export the zero-set mesh, and write metrics.json."""
from pathlib import Path
import argparse

import torch

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.artifacts import save_json
from rootsplat.evaluation import (aggregate_view_metrics, export_current_mesh,
                                  render_view)
from rootsplat.experiment import load_checkpoint, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", choices=("train", "test", "all"), default="test")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-views", type=int, default=0)
    p.add_argument("--export-resolution", type=int, default=0)
    p.add_argument("--tag", default="",
                   help="Optional artifact suffix for matched checkpoint comparisons")
    a = p.parse_args()
    run = Path(a.run)
    cfg = load_config(run / "config.resolved.yaml")
    dcfg, mcfg = cfg["dataset"], cfg["model"]
    scene = DTUScene(dcfg["scene"], device=a.device,
                     downscale=dcfg.get("downscale", 0.25),
                     test_every=dcfg.get("test_every", 8),
                     priors_dir=dcfg.get("priors_dir"),
                     depth_type=dcfg.get("depth_type", "ray"),
                     normal_space=dcfg.get("normal_space", "world"),
                     depth_scale=dcfg.get("depth_scale", 1.0),
                     require_masks=dcfg.get("require_masks", False),
                     require_scale_matrices=dcfg.get(
                         "require_scale_matrices", False))
    model = RootSplat(Config(**mcfg), device=a.device)
    checkpoint = Path(a.checkpoint) if a.checkpoint else run / "checkpoints" / "final.pt"
    load_checkpoint(checkpoint, model, map_location=a.device, restore_rng=False)
    model.eval()
    ids = scene.test_ids if a.split == "test" else scene.train_ids
    if a.split == "all":
        ids = list(range(len(scene.views)))
    if not ids:
        raise RuntimeError(f"Split {a.split} is empty")
    if a.max_views > 0:
        ids = ids[:a.max_views]
    if a.tag and not all(c.isalnum() or c in "-_" for c in a.tag):
        raise ValueError("--tag may contain only letters, digits, '-' and '_'")
    artifact_split = a.split if not a.tag else f"{a.split}_{a.tag}"
    rows = []
    for idx in ids:
        view = scene.views[idx]
        print(f"Rendering {idx}: {view.name}", flush=True)
        rows.append(render_view(
            model, view, run / "renders" / artifact_split / view.name))
    aggregate = aggregate_view_metrics(rows)
    save_json(run / "metrics" / f"{artifact_split}.json",
              dict(aggregate=aggregate, per_view=rows))
    resolution = a.export_resolution or cfg.get("training", {}).get("export_resolution")
    stats = export_current_mesh(model, run / "mesh" / "final_eval.ply",
                                resolution=resolution, cameras=scene.train_cameras,
                                normalized_to_world=scene.normalized_to_world,
                                normalized_path=run / "mesh" /
                                    "final_eval_normalized.ply")
    save_json(run / "metrics" / "topology.json", stats)
    print(f"PSNR {aggregate.get('psnr', float('nan')):.3f}, "
          f"SSIM {aggregate.get('ssim', float('nan')):.4f}")


if __name__ == "__main__":
    main()
