#!/usr/bin/env python3
"""Convert a COLMAP text sparse model into RootSplat's portable track cache.

Convert a binary model first with:

  colmap model_converter --input_path sparse/0 --output_path sparse_txt \
      --output_type TXT
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
from PIL import Image

from rootsplat.tracks import save_tracks


def image_sizes(image_dir):
    result = {}
    for path in Path(image_dir).iterdir():
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                result[path.name] = (image.width, image.height)
                result[path.stem] = (image.width, image.height)
        except Exception:
            pass
    return result


def natural_key(path):
    return [int(value) if value.isdigit() else value.lower()
            for value in re.split(r"(\d+)", path.stem)]


def read_colmap_images(path):
    lines = [line.strip() for line in Path(path).read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) % 2:
        raise ValueError("COLMAP images.txt must contain two lines per image")
    images = []
    for index in range(0, len(lines), 2):
        header, points = lines[index].split(), lines[index + 1].split()
        if len(header) < 10 or len(points) % 3:
            raise ValueError("Malformed COLMAP images.txt")
        image_id, name = int(header[0]), " ".join(header[9:])
        observations = []
        for i in range(0, len(points), 3):
            point_id = int(points[i + 2])
            if point_id >= 0:
                observations.append((float(points[i]), float(points[i + 1]),
                                     point_id))
        images.append((image_id, name, observations))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-txt", required=True)
    parser.add_argument("--image-dir", required=True,
                        help="Images used by COLMAP, for coordinate normalization")
    parser.add_argument("--scene-image-dir", required=True,
                        help="RootSplat image directory; defines sequential view IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--sigma-pixels", type=float, default=1.0)
    parser.add_argument("--min-track-length", type=int, default=2)
    args = parser.parse_args()

    if args.sigma_pixels <= 0 or args.min_track_length < 2:
        raise ValueError("sigma-pixels must be positive and min-track-length >= 2")
    sizes = image_sizes(args.image_dir)
    readable = image_sizes(args.scene_image_dir)
    scene_files = sorted(
        [path for path in Path(args.scene_image_dir).iterdir()
         if path.is_file() and (path.name in readable or path.stem in readable)],
        key=natural_key)
    stem_to_view = {path.stem: i for i, path in enumerate(scene_files)}
    name_to_view = {path.name: i for i, path in enumerate(scene_files)}
    grouped = {}
    missing_images = []
    for _image_id, name, observations in read_colmap_images(args.images_txt):
        view = name_to_view.get(name, stem_to_view.get(Path(name).stem))
        size = sizes.get(name, sizes.get(Path(name).stem))
        if view is None or size is None:
            missing_images.append(name)
            continue
        width, height = size
        for u, v, point_id in observations:
            if 0 <= u < width and 0 <= v < height:
                grouped.setdefault(point_id, []).append(
                    (view, u / width, v / height,
                     args.sigma_pixels / width, args.sigma_pixels / height))

    track_id, view_id, uv01, confidence, sigma01 = [], [], [], [], []
    duplicate_observations = 0
    local = 0
    for _point_id, observations in grouped.items():
        unique = {}
        for observation in observations:
            if observation[0] in unique:
                duplicate_observations += 1
            unique[observation[0]] = observation
        observations = list(unique.values())
        if len(observations) < args.min_track_length:
            continue
        for view, u, v, su, sv in observations:
            track_id.append(local); view_id.append(view); uv01.append((u, v))
            confidence.append(1.0); sigma01.append((su, sv))
        local += 1
    if not track_id:
        raise RuntimeError("No valid COLMAP tracks were produced")
    # Unique point order is deterministic under images.txt order.
    old = np.asarray(track_id, dtype=np.int64)
    _, track_id = np.unique(old, return_inverse=True)
    metadata = dict(source="COLMAP-SIFT", sigma_pixels=float(args.sigma_pixels),
                    images_txt=str(Path(args.images_txt).resolve()))
    save_tracks(args.output, track_id, view_id, uv01, confidence, sigma01,
                metadata=metadata)
    lengths = np.unique(track_id, return_counts=True)[1]
    report = dict(schema="rootsplat.colmap_tracks.v1", output=str(Path(args.output)),
                  tracks=int(len(lengths)), observations=int(len(track_id)),
                  views=int(len(np.unique(view_id))),
                  track_length_mean=float(lengths.mean()),
                  track_length_max=int(lengths.max()),
                  missing_images=missing_images,
                  duplicate_observations=int(duplicate_observations))
    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
