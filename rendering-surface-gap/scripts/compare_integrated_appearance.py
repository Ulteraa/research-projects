#!/usr/bin/env python3
"""Final feasibility gate for fixed-geometry multi-view appearance training."""
from pathlib import Path
import argparse
import json
import math

import torch

from rootsplat.artifacts import save_json


def trusted_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sdf_exact(reference_checkpoint, candidate_checkpoint):
    reference = trusted_load(reference_checkpoint)["model"]
    candidate = trusted_load(candidate_checkpoint)["model"]
    ref_keys = {key for key in reference if key.startswith("sdf.")}
    can_keys = {key for key in candidate if key.startswith("sdf.")}
    mismatches = []
    if ref_keys != can_keys:
        mismatches.append("sdf_state_keys")
    for key in sorted(ref_keys & can_keys):
        if reference[key].shape != candidate[key].shape:
            mismatches.append(f"{key}:shape")
        elif not torch.equal(reference[key], candidate[key]):
            mismatches.append(f"{key}:value")
    return not mismatches, mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-gate", required=True)
    parser.add_argument("--geometry-checkpoint", required=True)
    parser.add_argument("--appearance-checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-psnr-gain", type=float, default=3.0)
    parser.add_argument("--min-foreground-psnr-gain", type=float, default=3.0)
    parser.add_argument("--min-ssim-gain", type=float, default=0.0)
    parser.add_argument("--min-candidate-psnr", type=float, default=15.0)
    parser.add_argument("--min-candidate-foreground-psnr", type=float,
                        default=14.0)
    parser.add_argument("--max-mask-iou-drift", type=float, default=1e-4)
    args = parser.parse_args()

    geometry_gate_path = Path(args.geometry_gate)
    reference_path, candidate_path = Path(args.reference), Path(args.candidate)
    geometry_gate = json.loads(geometry_gate_path.read_text())
    reference = json.loads(reference_path.read_text())["aggregate"]
    candidate = json.loads(candidate_path.read_text())["aggregate"]
    if int(reference.get("num_views", -1)) != \
            int(candidate.get("num_views", -2)):
        raise ValueError("Reference and candidate evaluate different view counts")
    exact, mismatches = sdf_exact(
        args.geometry_checkpoint, args.appearance_checkpoint)

    def gain(key):
        return float(candidate[key] - reference[key])

    psnr_gain = gain("psnr")
    foreground_gain = gain("psnr_foreground")
    ssim_gain = gain("ssim")
    alpha_drift = abs(float(candidate["alpha_mask"]["iou"] -
                            reference["alpha_mask"]["iou"]))
    root_drift = abs(float(candidate["root_mask"]["iou"] -
                           reference["root_mask"]["iou"]))

    def lower(value, threshold):
        return math.isfinite(float(value)) and float(value) >= float(threshold)

    checks = dict(
        geometry_gate_passed=dict(
            value=geometry_gate.get("status"), relation="==", threshold="pass",
            passed=geometry_gate.get("status") == "pass"),
        geometry_checkpoint_exact=dict(
            value=exact, relation="==", threshold=True, passed=exact),
        psnr_gain=dict(value=psnr_gain, relation=">=",
                       threshold=float(args.min_psnr_gain),
                       passed=lower(psnr_gain, args.min_psnr_gain)),
        foreground_psnr_gain=dict(
            value=foreground_gain, relation=">=",
            threshold=float(args.min_foreground_psnr_gain),
            passed=lower(foreground_gain, args.min_foreground_psnr_gain)),
        ssim_gain=dict(value=ssim_gain, relation=">=",
                       threshold=float(args.min_ssim_gain),
                       passed=lower(ssim_gain, args.min_ssim_gain)),
        candidate_psnr=dict(value=float(candidate["psnr"]), relation=">=",
                            threshold=float(args.min_candidate_psnr),
                            passed=lower(candidate["psnr"],
                                         args.min_candidate_psnr)),
        candidate_foreground_psnr=dict(
            value=float(candidate["psnr_foreground"]), relation=">=",
            threshold=float(args.min_candidate_foreground_psnr),
            passed=lower(candidate["psnr_foreground"],
                         args.min_candidate_foreground_psnr)),
        alpha_iou_fixed_geometry=dict(
            value=alpha_drift, relation="<=",
            threshold=float(args.max_mask_iou_drift),
            passed=math.isfinite(alpha_drift) and
            alpha_drift <= float(args.max_mask_iou_drift)),
        root_iou_fixed_geometry=dict(
            value=root_drift, relation="<=",
            threshold=float(args.max_mask_iou_drift),
            passed=math.isfinite(root_drift) and
            root_drift <= float(args.max_mask_iou_drift)))
    failures = [name for name, check in checks.items() if not check["passed"]]
    payload = dict(
        schema="rootsplat.integrated_appearance_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        view_count=int(candidate["num_views"]), checks=checks,
        geometry_mismatches=mismatches,
        geometry_gate=str(geometry_gate_path),
        geometry_checkpoint=str(Path(args.geometry_checkpoint)),
        appearance_checkpoint=str(Path(args.appearance_checkpoint)),
        reference=str(reference_path), candidate=str(candidate_path))
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
