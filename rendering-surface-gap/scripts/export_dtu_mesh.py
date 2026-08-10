#!/usr/bin/env python3
"""Export a checkpoint as dataset-world input for the official DTU evaluator.

This script performs coordinate conversion and writes an auditable handoff
manifest.  It intentionally does not reimplement or rename itself as the
released DTU visibility/crop evaluator.
"""
from pathlib import Path
import argparse
import json
import re

import torch

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.artifacts import save_json
from rootsplat.evaluation import export_current_mesh
from rootsplat.experiment import load_checkpoint, load_config


def scan_id_from_path(path):
    matches = re.findall(r"scan[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--export-resolution", type=int, default=0)
    p.add_argument(
        "--allow-identity-transform", action="store_true",
        help="Permit missing scale_mat_i entries. Never use this for a DTU paper result.")
    a = p.parse_args()
    if str(a.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    run = Path(a.run)
    cfg = load_config(run / "config.resolved.yaml")
    dcfg, mcfg = cfg["dataset"], cfg["model"]
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
    if scene.scale_matrices_present != len(scene.views) and \
            not a.allow_identity_transform:
        raise RuntimeError(
            "Official-evaluation export requires scale_mat_i for every view. "
            "Use --allow-identity-transform only for a verified non-DTU scene.")

    model = RootSplat(Config(**mcfg), device=a.device)
    checkpoint = Path(a.checkpoint) if a.checkpoint else \
        run / "checkpoints" / "final.pt"
    load_checkpoint(checkpoint, model, map_location=a.device, restore_rng=False)
    model.eval()
    output = Path(a.output) if a.output else run / "mesh" / "final_dtu_world.ply"
    normalized = output.with_name(output.stem + "_normalized.ply")
    resolution = a.export_resolution or \
        cfg.get("training", {}).get("export_resolution")
    stats = export_current_mesh(
        model, output, resolution=resolution, cameras=scene.train_cameras,
        normalized_to_world=scene.normalized_to_world,
        normalized_path=normalized)
    payload = dict(
        schema="rootsplat.dtu_official_handoff.v1",
        scan_id=scan_id_from_path(scene.scene_dir),
        scene=str(scene.scene_dir), checkpoint=str(checkpoint),
        evaluator_input_mesh=str(output),
        diagnostic_normalized_mesh=str(normalized),
        coordinate_space="dataset_world",
        coordinate_unit="dataset_native",
        normalized_to_world=scene.normalized_to_world.tolist(),
        scale_matrices_present=scene.scale_matrices_present,
        scale_matrix_count=len(scene.views),
        official_evaluation_status="not_run",
        note=("Pass evaluator_input_mesh to the benchmark's released DTU "
              "visibility/crop evaluator; uniform_surface metrics are not official."),
        topology=stats)
    handoff = run / "metrics" / "dtu_official_handoff.json"
    save_json(handoff, payload)
    print(json.dumps(payload, indent=2))
    print(f"Handoff manifest: {handoff}")


if __name__ == "__main__":
    main()
