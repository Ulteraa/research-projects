#!/usr/bin/env bash
set -euo pipefail

scene="${ROOTSPLAT_SCENE:-/workspace/data/DTU/dtu_scan24}"
depths="${ROOTSPLAT_VGGT_DEPTHS:-$scene/vggt/poisson_v076/calibrated_depths.npz}"
output="${ROOTSPLAT_NEURAL_PULL_OUTPUT:-$scene/vggt/neural_pull_v077}"
python_bin="${ROOTSPLAT_PYTHON:-python}"

test -f "$depths" || {
  echo "STOP: calibrated VGGT depth evidence is missing: $depths"
  exit 1
}

test ! -e "$output" || {
  echo "STOP: output already exists: $output"
  exit 1
}

echo "===== DIRECT VGGT POINT/RAY -> NEURAL SDF ====="
"$python_bin" -u scripts/fit_vggt_neural_pull_sdf.py \
  --scene "$scene" \
  --depths "$depths" \
  --output "$output" \
  --steps 1200 \
  --batch-size 2048 \
  --learning-rate 0.0005 \
  --query-sigma-start 0.04 \
  --query-sigma-end 0.006 \
  --ray-sign-offset 0.03 \
  --ray-sign-margin 0.005 \
  --max-points 250000 \
  --balance-voxel 0.005 \
  --grid-resolution 192 \
  --device cuda

echo "===== GATE SUMMARY ====="
"$python_bin" - "$output" <<'PY'
from pathlib import Path
import json
import sys

output = Path(sys.argv[1])
report = json.loads((output / "neural_pull_sdf_gate.json").read_text())
print("status:", report["status"])
print("failures:", report["failures"])
print("surface p95:", report["field"]["surface_abs_p95"])
print("inside fraction:", report["grid_sdf"]["inside_fraction"])
print("ray thickness median:", report["ray_solid"]["thickness_median"])
print("thin-ray fraction:", report["ray_solid"]["thin_fraction_005"])
print("evidence gate:", report["evidence_gate"]["status"])
print("mesh:", report["mesh"])
print("preview:", report["preview"])
if report["status"] != "pass":
    raise SystemExit(2)
PY

echo "VGGT NEURAL-PULL SDF GATE: PASS"
echo "Do not train appearance until the preview has also been inspected."
