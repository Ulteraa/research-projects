#!/usr/bin/env bash
set -euo pipefail

# Training-free replacement for the failed v0.7.5 visual-hull sign fill.
scene="${ROOTSPLAT_SCENE:-/workspace/data/DTU/dtu_scan24}"
output="${ROOTSPLAT_POISSON_OUTPUT:-$scene/vggt/poisson_v076}"
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
  echo "STOP: output already exists; preserve it and choose a new ROOTSPLAT_POISSON_OUTPUT"
  echo "$output"
  exit 1
}
"$python_bin" - <<'PY'
try:
    import open3d
except ImportError as error:
    raise SystemExit(
        "STOP: Open3D is missing. Run: python -m pip install -e '.[poisson]'") \
        from error
print("Open3D:", open3d.__version__)
PY

"$python_bin" -u scripts/prepare_vggt_poisson_sdf.py \
  --scene "$scene" \
  --raw "$raw" \
  --initializer-gate "$initializer_gate" \
  --output "$output" \
  --resolution 256 \
  --confidence-quantile 0.5 \
  --cross-view-neighbors 4 \
  --cross-view-distance 0.03 \
  --cross-view-min-support 1 \
  --normal-edge-length 0.04 \
  --oriented-voxel-size 0.005 \
  --max-oriented-points 250000 \
  --poisson-depth 9 \
  --poisson-scale 1.02 \
  --sign-samples 3

"$python_bin" - <<'PY'
from pathlib import Path
import json
import os

scene = Path(os.environ.get(
    "ROOTSPLAT_SCENE", "/workspace/data/DTU/dtu_scan24"))
output = Path(os.environ.get(
    "ROOTSPLAT_POISSON_OUTPUT", str(scene / "vggt" / "poisson_v076")))
report = json.loads((output / "poisson_sdf_gate.json").read_text())
print("\n===== VGGT -> SCREENED-POISSON -> SDF GATE =====")
print("status:", report["status"])
print("failures:", report["failures"])
print("mesh:", report["mesh"])
print("preview:", report["preview"])
print("oriented evidence:", report["oriented_evidence"])
print("point -> surface p95:",
      report["evidence_gate"]["point_to_surface"]["p95"])
print("surface -> point p95:",
      report["evidence_gate"]["surface_to_point"]["p95"])
print("unsupported surface fraction:",
      report["evidence_gate"]["unsupported_surface_fraction"])
print("inside fraction:", report["grid_sdf"]["inside_fraction"])
print("DO NOT TRAIN YET. Inspect surface_normalized.ply and surface_preview.png.")
PY
