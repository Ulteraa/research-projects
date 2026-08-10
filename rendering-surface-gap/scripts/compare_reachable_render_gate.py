#!/usr/bin/env python3
"""Gate the v0.7.3 point fit before spending compute on appearance.

The capacity report establishes correspondence fit on a deterministic
train/validation split.  This script adds the missing full held-out-view and
mesh-topology checks.  RGB gain is deliberately not required because both
checkpoints still contain the same untrained appearance network.
"""
from pathlib import Path
import argparse
import json
import math

from rootsplat.artifacts import save_json


def load_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return math.isfinite(float(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-report", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-alpha-iou-drop", type=float, default=0.01)
    parser.add_argument("--max-root-iou-drop", type=float, default=0.02)
    parser.add_argument("--max-psnr-drop", type=float, default=0.25)
    args = parser.parse_args()

    capacity_path = Path(args.capacity_report)
    reference_path = Path(args.reference)
    candidate_path = Path(args.candidate)
    topology_path = Path(args.topology)
    capacity = load_json(capacity_path)
    reference = load_json(reference_path)["aggregate"]
    candidate = load_json(candidate_path)["aggregate"]
    topology = load_json(topology_path)
    if int(reference.get("num_views", -1)) != \
            int(candidate.get("num_views", -2)):
        raise ValueError("Reference and candidate evaluate different view counts")

    validation_ratio = float(
        capacity["checks"]["validation_distance_ratio"]["value"])
    eikonal_increase = float(
        capacity["checks"]["validation_eikonal_preservation"]["value"])
    saturation = float(
        capacity["checks"]["correction_saturation_fraction"]["value"])
    alpha_gain = float(candidate["alpha_mask"]["iou"] -
                       reference["alpha_mask"]["iou"])
    root_gain = float(candidate["root_mask"]["iou"] -
                      reference["root_mask"]["iou"])
    psnr_gain = float(candidate["psnr"] - reference["psnr"])
    closed = bool(topology.get("closed_oriented_manifold", False))

    checks = dict(
        capacity_gate_passed=dict(
            value=capacity.get("status"), relation="==", threshold="pass",
            passed=capacity.get("status") == "pass"),
        heldout_track_distance_ratio=dict(
            value=validation_ratio, relation="<=", threshold=.85,
            passed=finite(validation_ratio) and validation_ratio <= .85),
        heldout_track_eikonal_increase=dict(
            value=eikonal_increase, relation="<=", threshold=.02,
            passed=finite(eikonal_increase) and eikonal_increase <= .02),
        correction_saturation_fraction=dict(
            value=saturation, relation="<=", threshold=.05,
            passed=finite(saturation) and saturation <= .05),
        heldout_alpha_iou_preservation=dict(
            value=alpha_gain, relation=">=",
            threshold=-float(args.max_alpha_iou_drop),
            passed=finite(alpha_gain) and
            alpha_gain >= -float(args.max_alpha_iou_drop)),
        heldout_root_iou_preservation=dict(
            value=root_gain, relation=">=",
            threshold=-float(args.max_root_iou_drop),
            passed=finite(root_gain) and
            root_gain >= -float(args.max_root_iou_drop)),
        untrained_rgb_non_regression=dict(
            value=psnr_gain, relation=">=",
            threshold=-float(args.max_psnr_drop),
            passed=finite(psnr_gain) and
            psnr_gain >= -float(args.max_psnr_drop)),
        closed_oriented_manifold=dict(
            value=closed, relation="==", threshold=True, passed=closed))
    failures = [name for name, check in checks.items() if not check["passed"]]
    payload = dict(
        schema="rootsplat.reachable_render_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        view_count=int(candidate["num_views"]), checks=checks,
        capacity_report=str(capacity_path), reference=str(reference_path),
        candidate=str(candidate_path), topology=str(topology_path))
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
