#!/usr/bin/env python3
"""Align batched VGGT predictions and emit all train-only calibrated depths."""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.data import DTUScene
from rootsplat.tsdf import (check, cross_view_filter, reproject_point_map,
                            sha256_file)
from rootsplat.vggt_initializer import (
    camera_centers_from_extrinsics, robust_camera_alignment)


def _require_alignment(audit, batch_index):
    checks = dict(
        inlier_fraction=check(audit["inlier_fraction"], ">=", .75),
        residual_median=check(audit["residual_normalized_median"], "<=", .05),
        inlier_residual_p95=check(
            audit["residual_normalized_inlier_p95"], "<=", .08),
        rotation_determinant_error=check(
            abs(audit["rotation_determinant"] - 1.0), "<=", 1e-5))
    failures = [name for name, value in checks.items() if not value["passed"]]
    if failures:
        raise RuntimeError(
            f"VGGT batch {batch_index} camera alignment failed: {failures}")
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--input", required=True,
                        help="Directory from export_vggt_dtu_fulltrain.py")
    parser.add_argument("--output", required=True,
                        help="New calibrated depth NPZ")
    parser.add_argument("--report", required=True)
    parser.add_argument("--downscale", type=float, default=.25)
    parser.add_argument("--confidence-quantile", type=float, default=.5)
    parser.add_argument("--alignment-threshold", type=float, default=.08)
    parser.add_argument("--alignment-trials", type=int, default=512)
    parser.add_argument("--cross-view-neighbors", type=int, default=6)
    parser.add_argument("--cross-view-distance", type=float, default=.03)
    parser.add_argument("--cross-view-min-support", type=int, default=1)
    parser.add_argument("--bound", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input)
    manifest_path = input_dir / "manifest.json"
    output, report_path = Path(args.output), Path(args.report)
    if output.exists() or report_path.exists():
        raise FileExistsError("Refusing to overwrite calibrated depth artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    if manifest.get("schema") != "rootsplat.vggt_fulltrain_raw.v1":
        raise RuntimeError("Expected rootsplat.vggt_fulltrain_raw.v1 manifest")

    scene = DTUScene(
        args.scene, device="cpu", downscale=float(args.downscale), test_every=8,
        require_masks=True, require_scale_matrices=True)
    expected_train = [int(value) for value in scene.train_ids]
    heldout = set(int(value) for value in scene.test_ids)
    if manifest.get("train_view_ids") != expected_train:
        raise RuntimeError("VGGT manifest train split differs from DTUScene")
    if set(manifest.get("heldout_view_ids", [])) != heldout:
        raise RuntimeError("VGGT manifest held-out split differs from DTUScene")

    by_view = {}
    batch_audits = []
    for batch_row in manifest.get("batches", []):
        index = int(batch_row["index"])
        path = input_dir / batch_row["path"]
        if sha256_file(path) != batch_row.get("sha256"):
            raise RuntimeError(f"VGGT batch integrity mismatch: {path}")
        with np.load(path) as batch:
            required = {"view_ids", "output_view_ids", "world_points",
                        "world_points_conf", "extrinsic"}
            missing = required.difference(batch.files)
            if missing:
                raise KeyError(f"Batch {index} misses {sorted(missing)}")
            view_ids = batch["view_ids"].astype(np.int64)
            output_ids = batch["output_view_ids"].astype(np.int64)
            if set(view_ids.tolist()) & heldout or set(output_ids.tolist()) & heldout:
                raise RuntimeError(f"Batch {index} contains held-out views")
            if not set(output_ids.tolist()).issubset(set(view_ids.tolist())):
                raise RuntimeError(f"Batch {index} output IDs are not selected IDs")
            extrinsic = batch["extrinsic"].astype(np.float64)
            predicted = camera_centers_from_extrinsics(extrinsic)
            known = np.stack([
                scene.view_cameras[int(view_id)].center.detach().cpu().numpy()
                for view_id in view_ids])
            similarity, alignment = robust_camera_alignment(
                predicted, known,
                threshold_fraction=float(args.alignment_threshold),
                trials=int(args.alignment_trials), seed=int(args.seed) + index)
            alignment_checks = _require_alignment(alignment, index)
            inlier = np.asarray(alignment["inlier_mask"], dtype=bool)
            slot_for_id = {int(view_id): slot
                           for slot, view_id in enumerate(view_ids)}
            output_slots = [slot_for_id[int(view_id)] for view_id in output_ids]
            bad_outputs = [int(view_ids[slot]) for slot in output_slots
                           if not inlier[slot]]
            if bad_outputs:
                raise RuntimeError(
                    f"Batch {index} has outlying emitted cameras: {bad_outputs}")
            point_maps = batch["world_points"]
            confidence_maps = batch["world_points_conf"]
            raster = []
            for view_id, slot in zip(output_ids, output_slots):
                view_id = int(view_id)
                if view_id in by_view:
                    raise RuntimeError(f"Training view {view_id} is emitted twice")
                points = similarity.apply_points(
                    point_maps[int(slot)]).astype(np.float32)
                view = scene.views[view_id]
                depth, weight, audit = reproject_point_map(
                    points, confidence_maps[int(slot)], view.camera,
                    mask=view.mask,
                    confidence_quantile=float(args.confidence_quantile),
                    bound=float(args.bound))
                audit.update(view_id=view_id, view_name=view.name)
                by_view[view_id] = (depth, weight, view.camera)
                raster.append(audit)
        batch_audits.append(dict(
            index=index, selected_view_ids=view_ids.tolist(),
            output_view_ids=output_ids.tolist(), alignment=alignment,
            checks=alignment_checks, rasterization=raster))

    if sorted(by_view) != expected_train:
        missing = sorted(set(expected_train) - set(by_view))
        extra = sorted(set(by_view) - set(expected_train))
        raise RuntimeError(f"Full-view calibration mismatch; missing={missing}, extra={extra}")
    view_ids = np.asarray(expected_train, dtype=np.int32)
    raw_depth = np.stack([by_view[index][0] for index in expected_train])
    raw_weight = np.stack([by_view[index][1] for index in expected_train])
    cameras = [by_view[index][2] for index in expected_train]
    depth, weight, support, cross_audit = cross_view_filter(
        raw_depth, raw_weight, cameras,
        neighbors=int(args.cross_view_neighbors),
        tolerance=float(args.cross_view_distance),
        min_support_views=int(args.cross_view_min_support))
    pixels_per_view = np.count_nonzero(depth, axis=(1, 2))
    checks = dict(
        exact_training_view_coverage=check(len(view_ids), "==", len(expected_train)),
        heldout_geometry_views=check(
            len(set(view_ids.tolist()) & heldout), "==", 0),
        minimum_pixels_per_view=check(int(pixels_per_view.min()), ">=", 500),
        accepted_depth_pixels=check(int(np.count_nonzero(depth)), ">=", 100_000),
        cross_view_retained_fraction=check(
            cross_audit["retained_fraction"], ">=", .10))
    failures = [name for name, value in checks.items() if not value["passed"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, view_ids=view_ids, depth=depth.astype(np.float32),
        weight=weight.astype(np.float32), support=support.astype(np.uint8))
    report = dict(
        schema="rootsplat.vggt_fulltrain_depth_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        scene=str(Path(args.scene).resolve()), input=str(input_dir.resolve()),
        output=str(output.resolve()), output_sha256=sha256_file(output),
        train_view_ids=view_ids.tolist(), heldout_view_ids=sorted(heldout),
        batch_alignment=batch_audits, cross_view=cross_audit,
        depth_pixels_per_view=pixels_per_view.tolist(), checks=checks,
        options=vars(args))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
