#!/usr/bin/env python3
"""Export train-only VGGT depths as a COLMAP dense-Delaunay workspace."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rootsplat.data import DTUScene
from rootsplat.delaunay_sdf import (
    depth_evidence, visibility_lists, voxel_merge, write_fused_ply,
    write_visibility)
from rootsplat.tsdf import check, sha256_file


def _write_model(path, views):
    path.mkdir(parents=True, exist_ok=True)
    camera_lines, image_lines = [], []
    for registered_id, view in enumerate(views, start=1):
        camera = view.camera
        K = camera.K.detach().cpu().numpy()
        camera_lines.append(
            f"{registered_id} PINHOLE {camera.W} {camera.H} "
            f"{K[0,0]:.17g} {K[1,1]:.17g} {K[0,2]:.17g} {K[1,2]:.17g}")
        R = camera.R.detach().cpu().numpy()
        t = camera.t.detach().cpu().numpy()
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        image_lines.append(
            f"{registered_id} {qw:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {registered_id} "
            f"{view.name}.png")
    (path / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" +
        "\n".join(camera_lines) + "\n", encoding="utf8")
    (path / "images.txt").write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n" +
        "\n\n".join(image_lines) + "\n\n", encoding="utf8")
    (path / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n", encoding="utf8")


def _write_images(path, views):
    path.mkdir(parents=True, exist_ok=True)
    for view in views:
        rgb = np.rint(view.rgb.detach().cpu().numpy() * 255.0) \
            .clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(path / f"{view.name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--depths", required=True)
    parser.add_argument("--depth-gate", required=True)
    parser.add_argument("--output", required=True,
                        help="New COLMAP dense workspace; existing paths are refused")
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, default=.01)
    parser.add_argument("--visibility-tolerance", type=float, default=.03)
    parser.add_argument("--visibility-radius", type=int, default=1)
    parser.add_argument("--min-visible-views", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Delaunay workspace: {output}")
    depths_path, gate_path = Path(args.depths), Path(args.depth_gate)
    report = json.loads(gate_path.read_text(encoding="utf8"))
    if report.get("schema") != "rootsplat.vggt_fulltrain_depth_gate.v1" or \
            report.get("status") != "pass" or report.get("failures"):
        raise RuntimeError("A passing full-training-view depth gate is required")
    if report.get("output_sha256") != sha256_file(depths_path):
        raise RuntimeError("Full-training-view depth archive digest mismatch")

    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    expected = np.asarray(scene.train_ids, dtype=np.int64)
    heldout = set(int(value) for value in scene.test_ids)
    with np.load(depths_path) as archive:
        required = {"view_ids", "depth", "weight", "support"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError("Depth archive misses: " + ", ".join(sorted(missing)))
        view_ids = archive["view_ids"].astype(np.int64)
        depth = archive["depth"].astype(np.float32)
        weight = archive["weight"].astype(np.float32)
    if not np.array_equal(view_ids, expected):
        raise RuntimeError("Delaunay export requires every training view in order")
    if set(view_ids.tolist()) & heldout:
        raise RuntimeError("Held-out views reached the Delaunay constructor")
    views = [scene.views[int(index)] for index in view_ids]
    cameras = [view.camera for view in views]
    rgbs = [view.rgb.detach().cpu().numpy() for view in views]
    raw_points, raw_colours, raw_weights, raw_sources = depth_evidence(
        depth, weight, cameras, rgbs, bound=float(args.bound))
    points, colours, sources, masses = voxel_merge(
        raw_points, raw_colours, raw_weights, raw_sources,
        voxel_size=float(args.voxel_size), max_points=int(args.max_points),
        seed=int(args.seed))
    visibility = visibility_lists(
        points, sources, depth, cameras,
        tolerance=float(args.visibility_tolerance),
        radius=int(args.visibility_radius))
    visible_count = np.asarray([len(row) for row in visibility], dtype=np.int32)
    keep = visible_count >= int(args.min_visible_views)
    points, colours, masses = points[keep], colours[keep], masses[keep]
    visibility = [row for row, accepted in zip(visibility, keep) if accepted]
    visible_count = visible_count[keep]
    if len(points) < 10_000:
        raise RuntimeError("Too few multi-view points remain for Delaunay meshing")

    output.mkdir(parents=True)
    _write_model(output / "sparse", views)
    _write_images(output / "images", views)
    cloud_path = write_fused_ply(output / "fused.ply", points, colours)
    visibility_path = write_visibility(output / "fused.ply.vis", visibility)
    checks = dict(
        exact_training_view_coverage=check(len(view_ids), "==", len(expected)),
        heldout_geometry_views=check(
            len(set(view_ids.tolist()) & heldout), "==", 0),
        delaunay_points=check(len(points), ">=", 10_000),
        minimum_visibility=check(int(visible_count.min()), ">=",
                                 int(args.min_visible_views)),
        median_visibility=check(float(np.median(visible_count)), ">=", 2.0))
    failures = [name for name, value in checks.items() if not value["passed"]]
    audit = dict(
        schema="rootsplat.vggt_delaunay_workspace_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        protocol="train_only", scene=str(Path(args.scene).resolve()),
        depths=str(depths_path.resolve()), depth_gate=str(gate_path.resolve()),
        depth_sha256=sha256_file(depths_path), view_ids=view_ids.tolist(),
        heldout_view_ids=sorted(heldout), registration_order=view_ids.tolist(),
        input_depth_points=int(len(raw_points)), merged_points=int(len(keep)),
        output_points=int(len(points)), visibility=dict(
            minimum=int(visible_count.min()), median=float(np.median(visible_count)),
            p95=float(np.quantile(visible_count, .95)),
            maximum=int(visible_count.max())),
        point_mass=dict(median=float(np.median(masses)),
                        p95=float(np.quantile(masses, .95))),
        bounds=[points.min(axis=0).tolist(), points.max(axis=0).tolist()],
        checks=checks, fused_ply=str(cloud_path.resolve()),
        fused_ply_sha256=sha256_file(cloud_path),
        visibility_file=str(visibility_path.resolve()),
        visibility_sha256=sha256_file(visibility_path), options=vars(args))
    audit_path = output / "workspace_gate.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
