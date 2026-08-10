#!/usr/bin/env python3
"""Run the official frozen VGGT model and save portable DTU predictions.

The output deliberately contains raw VGGT-frame geometry.  Camera alignment,
confidence filtering, and all acceptance decisions happen in the independent
``prepare_vggt_initializer.py`` stage and therefore remain testable without
VGGT or its weights.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import json

import numpy as np
import torch

from rootsplat.data import _image_files


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-views", type=int, default=16,
                        help="Evenly spaced views; 0 uses every image")
    args = parser.parse_args()
    if args.max_views < 0:
        raise ValueError("--max-views must be non-negative")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("VGGT CUDA inference requested but CUDA is unavailable")

    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except ImportError as error:
        raise ImportError(
            "Install the official facebookresearch/vggt package in an isolated "
            "environment before running this exporter") from error

    scene = Path(args.scene)
    image_dir = next((scene / name for name in ("image", "images")
                      if (scene / name).is_dir()), None)
    images = _image_files(image_dir)
    if len(images) < 4:
        raise RuntimeError("VGGT initialization requires at least four images")
    if args.max_views and len(images) > args.max_views:
        view_ids = np.unique(np.rint(np.linspace(
            0, len(images) - 1, int(args.max_views))).astype(np.int64))
    else:
        view_ids = np.arange(len(images), dtype=np.int64)
    selected = [images[int(index)] for index in view_ids]
    input_images = load_and_preprocess_images(
        [str(path) for path in selected]).to(args.device)

    model = VGGT.from_pretrained(args.checkpoint).to(args.device).eval()
    if str(args.device).startswith("cuda"):
        major = torch.cuda.get_device_capability(torch.device(args.device))[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        autocast = torch.cuda.amp.autocast(dtype=dtype)
    else:
        dtype = torch.float32
        autocast = nullcontext()
    with torch.inference_mode(), autocast:
        prediction = model(input_images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            prediction["pose_enc"], input_images.shape[-2:])
        depth = prediction["depth"]
        points = unproject_depth_map_to_point_map(
            depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

    points = _numpy(points)
    extrinsic = _numpy(extrinsic).squeeze(0)
    intrinsic = _numpy(intrinsic).squeeze(0)
    depth = _numpy(depth).squeeze(0)
    confidence = _numpy(prediction["depth_conf"]).squeeze(0)
    color = _numpy(input_images).transpose(0, 2, 3, 1)
    color = np.rint(np.clip(color, 0, 1) * 255).astype(np.uint8)
    if points.shape[:3] != confidence.shape[:3] or points.shape != color.shape:
        raise RuntimeError(
            "Unexpected official VGGT output shapes: "
            f"points={points.shape}, confidence={confidence.shape}, "
            f"images={color.shape}")
    if extrinsic.shape != (len(view_ids), 3, 4):
        raise RuntimeError(f"Unexpected VGGT extrinsic shape: {extrinsic.shape}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, view_ids=view_ids.astype(np.int32),
        world_points=points.astype(np.float32),
        world_points_conf=confidence.astype(np.float32),
        images=color, extrinsic=extrinsic.astype(np.float32),
        intrinsic=intrinsic.astype(np.float32), depth=depth.astype(np.float32))
    report = dict(
        schema="rootsplat.vggt_raw.v1", scene=str(scene.resolve()),
        checkpoint=args.checkpoint, device=str(args.device), dtype=str(dtype),
        total_scene_views=int(len(images)), selected_views=view_ids.tolist(),
        prediction_resolution=list(points.shape[1:3]),
        output=str(output.resolve()), output_bytes=int(output.stat().st_size))
    report_path = Path(args.report) if args.report else output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
