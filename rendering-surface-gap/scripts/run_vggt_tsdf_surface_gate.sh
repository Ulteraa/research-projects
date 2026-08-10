#!/usr/bin/env bash
set -euo pipefail

# Cheap, training-free surface construction gate.  Expected inputs are the
# frozen VGGT export and its already-passing camera-alignment report.
scene="${ROOTSPLAT_SCENE:-/workspace/data/DTU/dtu_scan24}"
output="${ROOTSPLAT_TSDF_OUTPUT:-$scene/vggt/tsdf_v075}"
python_bin="${ROOTSPLAT_PYTHON:-python}"

raw="$scene/vggt/raw_predictions.npz"
initializer_gate="$scene/vggt/initializer_gate.json"

test -f "$raw" || {
  echo "STOP: missing VGGT raw predictions: $raw"
  exit 1
}
test -f "$initializer_gate" || {
  echo "STOP: missing VGGT alignment gate: $initializer_gate"
  exit 1
}
test ! -e "$output" || {
  echo "STOP: output already exists; preserve it and choose a new ROOTSPLAT_TSDF_OUTPUT"
  echo "$output"
  exit 1
}

"$python_bin" -u scripts/prepare_vggt_tsdf.py \
  --scene "$scene" \
  --raw "$raw" \
  --initializer-gate "$initializer_gate" \
  --output "$output" \
  --device cuda \
  --resolution 256 \
  --confidence-quantile 0.5 \
  --cross-view-neighbors 4 \
  --cross-view-distance 0.03 \
  --cross-view-min-support 1 \
  --truncation-voxels 4 \
  --visual-hull-fraction 0.8 \
  --visual-hull-min-views 4

"$python_bin" - <<'PY'
from pathlib import Path
import json
import os

scene = Path(os.environ.get(
    "ROOTSPLAT_SCENE", "/workspace/data/DTU/dtu_scan24"))
output = Path(os.environ.get(
    "ROOTSPLAT_TSDF_OUTPUT", str(scene / "vggt" / "tsdf_v075")))
report = json.loads((output / "tsdf_gate.json").read_text())
print("\n===== VGGT -> SDF SURFACE GATE =====")
print("status:", report["status"])
print("failures:", report["failures"])
print("mesh:", report["mesh"])
print("preview:", report["preview"])
print("point -> surface p95:",
      report["evidence_gate"]["point_to_surface"]["p95"])
print("surface -> point p95:",
      report["evidence_gate"]["surface_to_point"]["p95"])
print("unsupported surface fraction:",
      report["evidence_gate"]["unsupported_surface_fraction"])
print("components:", report["evidence_gate"]["components"])
print("DO NOT TRAIN YET. Inspect surface_normalized.ply and surface_preview.png.")
PY

