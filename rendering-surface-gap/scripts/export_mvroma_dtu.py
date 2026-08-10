#!/usr/bin/env python3
"""Run the official MV-RoMa model and save RootSplat-compatible predictions.

This script imports the authors' unmodified repository at runtime.  MV-RoMa is
never trained or fine-tuned by RootSplat; it is a frozen observation provider.
The saved files are subsequently calibrated, triangulated, and filtered by
``validate_tracks.py`` before any SDF optimization.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

from rootsplat.data import DTUScene, _image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvroma-repo", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", type=int, default=4)
    parser.add_argument("--height", type=int, default=672)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--coarse", type=int, default=672)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-angle", type=float, default=1.0)
    parser.add_argument("--max-angle", type=float, default=70.0)
    parser.add_argument("--max-sources", type=int, default=0,
                        help="0 processes every source; use 2 for a smoke test")
    parser.add_argument("--resume", action="store_true",
                        help="Skip prediction files that already exist")
    args = parser.parse_args()
    if args.targets < 1 or args.height <= 0 or args.width <= 0:
        raise ValueError("Invalid MV-RoMa group or output resolution")
    repository = Path(args.mvroma_repo).resolve()
    if not (repository / "demo.py").is_file():
        raise FileNotFoundError("MV-RoMa demo.py was not found")
    sys.path.insert(0, str(repository))
    from demo import build_model_matcher  # noqa: E402
    from src.run_model import run_model_test  # noqa: E402

    scene = DTUScene(args.scene, device="cpu", downscale=1.0, test_every=0,
                     require_masks=False, require_scale_matrices=True)
    image_dir = next((Path(args.scene) / name for name in ("image", "images")
                      if (Path(args.scene) / name).is_dir()), None)
    images = _image_files(image_dir)
    centers = np.stack([
        camera.center.detach().cpu().numpy()
        for camera in scene.view_cameras], axis=0).astype(np.float64)
    directions = centers / np.linalg.norm(centers, axis=-1, keepdims=True).clip(1e-12)
    angles = np.degrees(np.arccos(np.clip(directions @ directions.T, -1, 1)))
    if len(images) != len(centers):
        raise ValueError("Scene image/camera counts differ")

    prematch_name, prematch, model = build_model_matcher(
        device=args.device, weight_path=args.weights)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = []
    source_items = list(enumerate(images))
    if args.max_sources > 0:
        source_items = source_items[:args.max_sources]
    with torch.inference_mode():
        for progress, (source, image) in enumerate(source_items):
            candidates = np.arange(len(images))
            valid = (candidates != source) & (angles[source] >= args.min_angle) \
                & (angles[source] <= args.max_angle)
            candidates = candidates[valid]
            candidates = candidates[np.argsort(angles[source, candidates])]
            selected = candidates[:args.targets]
            if len(selected) < args.targets:
                raise RuntimeError(f"Too few MV-RoMa targets for {image.name}")
            destination = output / f"source_{source:03d}.npz"
            if args.resume and destination.is_file():
                groups.append(dict(source=source, targets=selected.tolist(),
                                   file=destination.name, skipped=True))
                print(f"[{progress + 1}/{len(source_items)}] skip {destination}",
                      flush=True)
                continue
            predictions = run_model_test(
                model,
                {"query_img_path": str(image),
                 "ref_img_paths": [str(images[int(i)]) for i in selected]},
                coarse_res_hw=(args.coarse, args.coarse),
                target_res_hw=(args.height, args.width),
                prematch_model=prematch,
                prematch_model_name=prematch_name,
                upsample_preds=True, num_cluster=512, device=args.device)
            finest = min(predictions.keys())
            flow = predictions[finest]["flow"][0].detach().cpu().float().numpy()
            certainty = predictions[finest]["certainty"][0].detach().cpu().float().numpy()
            np.savez_compressed(
                destination, source_view=np.asarray(source, dtype=np.int64),
                target_views=selected.astype(np.int64), flow=flow,
                certainty=certainty)
            groups.append(dict(source=source, targets=selected.tolist(),
                               shape=list(flow.shape), file=destination.name))
            print(f"[{progress + 1}/{len(source_items)}] {destination}", flush=True)
    report = dict(
        schema="rootsplat.mvroma_predictions.v1", frozen=True,
        official_repository=str(repository), weights=str(Path(args.weights).resolve()),
        scene=str(Path(args.scene).resolve()), groups=groups,
        target_resolution=[args.height, args.width], targets=args.targets)
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
