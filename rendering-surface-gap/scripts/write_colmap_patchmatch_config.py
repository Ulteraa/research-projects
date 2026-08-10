#!/usr/bin/env python3
"""Write deterministic source-view lists for known-camera COLMAP MVS.

COLMAP's automatic PatchMatch source selection relies on sparse covisibility.
Our known-camera model intentionally contains no invented sparse points, so it
needs an explicit ``stereo/patch-match.cfg``.  For an object-centric capture,
nearby viewing directions provide the useful baselines.  We rank them by angle
about the normalized scene origin and reject nearly coincident cameras.
"""
from pathlib import Path
import argparse
import itertools
import json
import struct

import numpy as np

from rootsplat.data import DTUScene, _image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--workspace", required=True,
                        help="COLMAP dense workspace after image_undistorter")
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--min-angle", type=float, default=1.0)
    parser.add_argument("--max-angle", type=float, default=60.0)
    parser.add_argument("--bound", type=float, default=1.05,
                        help="Half-extent of the normalized reconstruction box")
    parser.add_argument("--depth-padding", type=float, default=0.1,
                        help="Fractional padding of the global box-depth span")
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.neighbors < 2 or not 0 <= args.min_angle < args.max_angle <= 180:
        raise ValueError("Invalid neighbor count or angular interval")
    if args.bound <= 0 or not 0 <= args.depth_padding <= 1:
        raise ValueError("Invalid reconstruction bound or depth padding")

    scene = DTUScene(args.scene, device="cpu", downscale=1.0, test_every=0,
                     require_masks=False, require_scale_matrices=True)
    image_dir = next((Path(args.scene) / name for name in ("image", "images")
                      if (Path(args.scene) / name).is_dir()), None)
    images = _image_files(image_dir)
    if len(images) != len(scene.views):
        raise ValueError("Scene image/camera counts differ")
    workspace = Path(args.workspace)
    registered_path = workspace / "sparse" / "images.bin"
    if not registered_path.is_file():
        raise RuntimeError(
            "Dense workspace has no sparse/images.bin; image undistortion "
            "did not finish")
    with registered_path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise RuntimeError("Dense sparse/images.bin has a truncated header")
    registered_images = struct.unpack("<Q", header)[0]
    if registered_images != len(images):
        raise RuntimeError(
            "Dense workspace registered-image mismatch: "
            f"{registered_images} != {len(images)}")
    centers = np.stack([
        camera.center.detach().cpu().numpy()
        for camera in scene.view_cameras], axis=0).astype(np.float64)
    directions = centers / np.linalg.norm(centers, axis=-1, keepdims=True).clip(1e-12)
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))

    corners = np.asarray(list(itertools.product(
        [-float(args.bound), float(args.bound)], repeat=3)), dtype=np.float64)
    box_depths = np.concatenate([
        (corners @ camera.R.detach().cpu().numpy().astype(np.float64).T +
         camera.t.detach().cpu().numpy().astype(np.float64))[:, 2]
        for camera in scene.view_cameras])
    unpadded_min = float(box_depths.min())
    unpadded_max = float(box_depths.max())
    if not np.isfinite(box_depths).all() or unpadded_min <= 0:
        raise RuntimeError(
            "The normalized reconstruction box is not entirely in front of "
            "every camera; refusing to invent a PatchMatch depth range")
    depth_span = unpadded_max - unpadded_min
    depth_min = max(1e-4, unpadded_min - args.depth_padding * depth_span)
    depth_max = unpadded_max + args.depth_padding * depth_span

    lines, counts = [], []
    for reference, image in enumerate(images):
        candidates = np.arange(len(images))
        valid = (candidates != reference) \
            & (angles[reference] >= args.min_angle) \
            & (angles[reference] <= args.max_angle)
        candidates = candidates[valid]
        candidates = candidates[np.argsort(angles[reference, candidates])]
        if len(candidates) < 2:
            raise RuntimeError(
                f"Fewer than two source views pass the angular gate for {image.name}")
        selected = candidates[:args.neighbors]
        lines.extend([image.name, ", ".join(images[int(i)].name for i in selected)])
        counts.append(len(selected))

    config = workspace / "stereo" / "patch-match.cfg"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("\n".join(lines) + "\n")
    depth_environment = config.parent / "depth-range.env"
    depth_environment.write_text(
        f"depth_min={depth_min:.17g}\n"
        f"depth_max={depth_max:.17g}\n")
    report = dict(
        schema="rootsplat.colmap_patchmatch_sources.v1",
        scene=str(Path(args.scene).resolve()), workspace=str(Path(args.workspace).resolve()),
        references=len(images), neighbors_min=int(min(counts)),
        neighbors_max=int(max(counts)), min_angle_degrees=float(args.min_angle),
        max_angle_degrees=float(args.max_angle), config=str(config.resolve()))
    report.update(
        reconstruction_bound=float(args.bound),
        depth_padding_fraction=float(args.depth_padding),
        depth_min=float(depth_min), depth_max=float(depth_max),
        depth_environment=str(depth_environment.resolve()))
    report_path = Path(args.report) if args.report else config.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
