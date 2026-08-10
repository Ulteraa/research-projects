#!/usr/bin/env python3
"""Matched gate for a mask-only continuation of a validated SDF bootstrap."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def _read(path):
    path = Path(path)
    return path, json.loads(path.read_text())


def _check(reference, candidate, gain, relation, threshold):
    passed = gain >= threshold if relation == ">=" else candidate <= threshold
    return dict(reference=float(reference), candidate=float(candidate),
                gain=float(gain), relation=relation,
                threshold=float(threshold), passed=bool(passed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-iou-gain", type=float, default=0.01)
    p.add_argument("--heldout-iou-gain", type=float, default=0.005)
    p.add_argument("--eikonal-ratio", type=float, default=1.25)
    p.add_argument("--normal-margin", type=float, default=2.0)
    a = p.parse_args()

    reference_path, reference = _read(a.reference)
    candidate_path, candidate = _read(a.candidate)
    for name, report in (("reference", reference), ("candidate", candidate)):
        # The clean v0.4.5 reference predates the explicit stage field; its
        # schema is bootstrap-only and therefore unambiguous.
        stage = report.get("stage")
        if stage is None and report.get("schema") == "rootsplat.bootstrap_gate.v1":
            stage = "bootstrap"
        if stage != "bootstrap":
            raise ValueError(f"{name} must be a bootstrap-stage geometry report")
        if report.get("decision", {}).get("status") != "pass":
            raise ValueError(f"{name} absolute geometry gate did not pass")

    rt = float(reference["learned"]["train"]["iou"])
    ct = float(candidate["learned"]["train"]["iou"])
    rh = float(reference["learned"]["heldout"]["iou"])
    ch = float(candidate["learned"]["heldout"]["iou"])
    re = float(reference["eikonal"]["mean"])
    ce = float(candidate["eikonal"]["mean"])
    rn = float(reference["mesh_normal_degrees"]["mean"])
    cn = float(candidate["mesh_normal_degrees"]["mean"])
    values = (rt, ct, rh, ch, re, ce, rn, cn)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Geometry comparison contains non-finite metrics")

    # Relative constraints are intentionally tighter than the absolute gate.
    # A continuation is useful only when it improves silhouettes without
    # spending the SDF metric/normal quality inherited from the reference.
    eikonal_limit = min(0.20, max(1e-8, re) * float(a.eikonal_ratio))
    normal_limit = min(15.0, rn + float(a.normal_margin))
    checks = {
        "train_iou_gain": _check(rt, ct, ct - rt, ">=", a.train_iou_gain),
        "heldout_iou_gain": _check(
            rh, ch, ch - rh, ">=", a.heldout_iou_gain),
        "eikonal_abs_mean": _check(
            re, ce, ce - re, "<=", eikonal_limit),
        "mesh_normal_mean_degrees": _check(
            rn, cn, cn - rn, "<=", normal_limit),
    }
    failures = [name for name, check in checks.items() if not check["passed"]]
    payload = dict(
        schema="rootsplat.mask_refinement_gate.v1",
        status="pass" if not failures else "fail",
        failures=failures,
        checks=checks,
        reference=str(reference_path), candidate=str(candidate_path),
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest())
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
