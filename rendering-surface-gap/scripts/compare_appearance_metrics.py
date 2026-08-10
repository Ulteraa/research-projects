#!/usr/bin/env python3
"""Non-inferential gate for fixed-geometry appearance learnability."""
from pathlib import Path
import argparse
import json
import math

from rootsplat.artifacts import save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-psnr-gain", type=float, default=3.0)
    p.add_argument("--min-foreground-psnr-gain", type=float, default=3.0)
    a = p.parse_args()
    reference = json.loads(Path(a.reference).read_text())["aggregate"]
    candidate = json.loads(Path(a.candidate).read_text())["aggregate"]
    if int(reference.get("num_views", -1)) != int(candidate.get("num_views", -2)):
        raise ValueError("Reference and candidate evaluate different view counts")

    checks = {}
    for key, threshold in (("psnr", a.min_psnr_gain),
                           ("psnr_foreground", a.min_foreground_psnr_gain)):
        before, after = float(reference[key]), float(candidate[key])
        gain = after - before
        checks[f"{key}_gain"] = dict(
            reference=before, candidate=after, gain=gain,
            relation=">=", threshold=float(threshold),
            passed=bool(math.isfinite(gain) and gain >= float(threshold)))
    failures = [name for name, item in checks.items() if not item["passed"]]
    payload = dict(
        schema="rootsplat.appearance_gate.v1",
        status="pass" if not failures else "fail", failures=failures,
        view_count=int(candidate["num_views"]), checks=checks,
        reference=str(Path(a.reference)), candidate=str(Path(a.candidate)))
    save_json(a.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
