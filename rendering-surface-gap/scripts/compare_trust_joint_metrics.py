#!/usr/bin/env python3
"""Gate a trust-region continuation on detail and silhouette preservation."""
from pathlib import Path
import argparse
import json
import math

from rootsplat.artifacts import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-psnr-gain", type=float, default=-0.10)
    parser.add_argument("--min-foreground-gain", type=float, default=-0.10)
    parser.add_argument("--min-boundary-gain", type=float, default=0.25)
    parser.add_argument("--min-edge-ratio-gain", type=float, default=0.03)
    parser.add_argument("--max-alpha-iou-drop", type=float, default=0.01)
    parser.add_argument("--max-root-iou-drop", type=float, default=0.02)
    args = parser.parse_args()

    reference_path, candidate_path = Path(args.reference), Path(args.candidate)
    reference = json.loads(reference_path.read_text())["aggregate"]
    candidate = json.loads(candidate_path.read_text())["aggregate"]
    if int(reference.get("num_views", -1)) != int(candidate.get("num_views", -2)):
        raise ValueError("Reference and candidate evaluate different view counts")

    checks = {}

    def gain_check(name, key, threshold):
        before, after = float(reference[key]), float(candidate[key])
        gain = after - before
        checks[name] = dict(reference=before, candidate=after, gain=gain,
                            relation=">=", threshold=float(threshold),
                            passed=bool(math.isfinite(gain) and
                                        gain >= float(threshold)))

    gain_check("psnr_non_regression", "psnr", args.min_psnr_gain)
    gain_check("foreground_psnr_non_regression", "psnr_foreground",
               args.min_foreground_gain)
    gain_check("boundary_psnr_gain", "psnr_foreground_boundary",
               args.min_boundary_gain)
    gain_check("edge_energy_ratio_gain", "edge_energy_ratio_foreground",
               args.min_edge_ratio_gain)

    for name, group, drop in (
            ("alpha_iou_preservation", "alpha_mask", args.max_alpha_iou_drop),
            ("root_iou_preservation", "root_mask", args.max_root_iou_drop)):
        before = float(reference[group]["iou"])
        after = float(candidate[group]["iou"])
        gain = after - before
        checks[name] = dict(reference=before, candidate=after, gain=gain,
                            relation=">=", threshold=-float(drop),
                            passed=bool(math.isfinite(gain) and
                                        gain >= -float(drop)))

    failures = [name for name, check in checks.items() if not check["passed"]]
    payload = dict(
        schema="rootsplat.trust_joint_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        view_count=int(candidate["num_views"]), checks=checks,
        reference=str(reference_path), candidate=str(candidate_path))
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
