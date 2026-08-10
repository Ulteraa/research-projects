#!/usr/bin/env python3
"""Export frozen VGGT point maps for every DTU training view in batches.

VGGT predictions from different forward passes live in different similarity
frames.  Every batch therefore repeats a small set of anchor cameras.  The
downstream calibration stage aligns each batch independently to the released
DTU cameras and emits each training view exactly once.  Held-out views are
never loaded by this script.
"""
from contextlib import nullcontext
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from rootsplat.data import _image_files


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _sha256(path, block_size=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(int(block_size))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _batches(train_ids, batch_size, anchor_count):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    if len(train_ids) < 4:
        raise ValueError("Full-training-view VGGT export requires four views")
    anchor_count = min(int(anchor_count), len(train_ids) - 1)
    if anchor_count < 4:
        raise ValueError("At least four alignment anchors are required")
    if int(batch_size) <= anchor_count:
        raise ValueError("--batch-size must exceed --anchor-views")
    anchor_slots = np.unique(np.rint(np.linspace(
        0, len(train_ids) - 1, anchor_count)).astype(np.int64))
    anchors = train_ids[anchor_slots]
    remaining = np.asarray(
        [value for value in train_ids if value not in set(anchors.tolist())],
        dtype=np.int64)
    capacity = int(batch_size) - len(anchors)
    groups = [remaining[start:start + capacity]
              for start in range(0, len(remaining), capacity)]
    if not groups:
        groups = [np.empty(0, dtype=np.int64)]
    result = []
    for index, group in enumerate(groups):
        selected = np.unique(np.concatenate([anchors, group])).astype(np.int64)
        emitted = selected if index == 0 else group.astype(np.int64)
        result.append((selected, emitted))
    emitted = np.concatenate([value for _, value in result])
    if sorted(emitted.tolist()) != sorted(train_ids.tolist()) or \
            len(np.unique(emitted)) != len(train_ids):
        raise RuntimeError("Internal batching error: train views are not emitted once")
    return anchors, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True,
                        help="New directory for batch NPZ files and manifest")
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--anchor-views", type=int, default=4)
    args = parser.parse_args()
    if args.test_every <= 0:
        raise ValueError("--test-every must be positive for leakage-safe export")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("VGGT CUDA inference requested but CUDA is unavailable")

    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except ImportError as error:
        raise ImportError(
            "Run this script in the isolated official VGGT environment") from error

    scene = Path(args.scene)
    image_dir = next((scene / name for name in ("image", "images")
                      if (scene / name).is_dir()), None)
    images = _image_files(image_dir)
    test_ids = np.asarray(
        [index for index in range(len(images)) if index % args.test_every == 0],
        dtype=np.int64)
    train_ids = np.asarray(
        [index for index in range(len(images)) if index not in set(test_ids.tolist())],
        dtype=np.int64)
    anchors, batches = _batches(
        train_ids, batch_size=args.batch_size, anchor_count=args.anchor_views)

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite VGGT batches: {output}")
    output.mkdir(parents=True)
    model = VGGT.from_pretrained(args.checkpoint).to(args.device).eval()
    if str(args.device).startswith("cuda"):
        major = torch.cuda.get_device_capability(torch.device(args.device))[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
    else:
        dtype = torch.float32

    batch_reports = []
    for batch_index, (selected_ids, output_ids) in enumerate(batches):
        selected = [images[int(index)] for index in selected_ids]
        input_images = load_and_preprocess_images(
            [str(path) for path in selected]).to(args.device)
        autocast = (torch.cuda.amp.autocast(dtype=dtype)
                    if str(args.device).startswith("cuda") else nullcontext())
        with torch.inference_mode(), autocast:
            prediction = model(input_images)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                prediction["pose_enc"], input_images.shape[-2:])
            depth = prediction["depth"]
            points = unproject_depth_map_to_point_map(
                depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
        points = _numpy(points)
        confidence = _numpy(prediction["depth_conf"]).squeeze(0)
        extrinsic = _numpy(extrinsic).squeeze(0)
        if points.shape[:3] != confidence.shape[:3] or \
                points.shape[0] != len(selected_ids):
            raise RuntimeError(
                f"Unexpected VGGT shapes in batch {batch_index}: "
                f"points={points.shape}, confidence={confidence.shape}")
        path = output / f"batch_{batch_index:03d}.npz"
        np.savez_compressed(
            path, view_ids=selected_ids.astype(np.int32),
            output_view_ids=output_ids.astype(np.int32),
            world_points=points.astype(np.float32),
            world_points_conf=confidence.astype(np.float32),
            extrinsic=extrinsic.astype(np.float32))
        row = dict(
            index=int(batch_index), path=path.name,
            selected_view_ids=selected_ids.tolist(),
            output_view_ids=output_ids.tolist(),
            prediction_resolution=list(points.shape[1:3]),
            bytes=int(path.stat().st_size), sha256=_sha256(path))
        batch_reports.append(row)
        print(json.dumps(row))
        del input_images, prediction, points, confidence, extrinsic, depth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = dict(
        schema="rootsplat.vggt_fulltrain_raw.v1",
        scene=str(scene.resolve()), checkpoint=args.checkpoint,
        device=str(args.device), dtype=str(dtype),
        total_scene_views=int(len(images)), test_every=int(args.test_every),
        train_view_ids=train_ids.tolist(), heldout_view_ids=test_ids.tolist(),
        anchor_view_ids=anchors.tolist(), batch_size=int(args.batch_size),
        batches=batch_reports)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
