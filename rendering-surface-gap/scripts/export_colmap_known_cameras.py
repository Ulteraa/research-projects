#!/usr/bin/env python3
"""Export IDR/NeuS DTU cameras as a registered COLMAP text model.

The exported world frame is RootSplat's normalized SDF frame induced by
``world_mat_i @ scale_mat_i``.  Consequently a fused COLMAP PLY generated from
this model must use ``initialization.surface_space: normalized``.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

from rootsplat.data import DTUScene, _image_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    scene = DTUScene(args.scene, device="cpu", downscale=1.0, test_every=0,
                     require_masks=False, require_scale_matrices=True)
    image_dir = next((Path(args.scene) / name for name in ("image", "images")
                      if (Path(args.scene) / name).is_dir()), None)
    images = _image_files(image_dir)
    if len(images) != len(scene.views):
        raise ValueError("Scene image/camera counts differ")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    camera_lines, image_lines = [], []
    for index, (view, image_path) in enumerate(zip(scene.views, images), start=1):
        camera = view.camera
        K = camera.K.detach().cpu().numpy()
        camera_lines.append(
            f"{index} PINHOLE {camera.W} {camera.H} "
            f"{K[0,0]:.17g} {K[1,1]:.17g} {K[0,2]:.17g} {K[1,2]:.17g}")
        R = camera.R.detach().cpu().numpy()
        t = camera.t.detach().cpu().numpy()
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        image_lines.append(
            f"{index} {qw:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {index} {image_path.name}")
    (output / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" +
        "\n".join(camera_lines) + "\n")
    (output / "images.txt").write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n" +
        "\n\n".join(image_lines) + "\n\n")
    (output / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    report = dict(schema="rootsplat.colmap_known_cameras.v1",
                  coordinate_frame="rootsplat_normalized",
                  cameras=len(camera_lines), images=len(image_lines),
                  output=str(output.resolve()))
    report_path = Path(args.report) if args.report else output / "export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
