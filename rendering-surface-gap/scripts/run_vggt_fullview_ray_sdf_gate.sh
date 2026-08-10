#!/usr/bin/env bash
set -euo pipefail

scene="${ROOTSPLAT_SCENE:-/workspace/data/DTU/dtu_scan24}"
output="${ROOTSPLAT_RAY_SDF_OUTPUT:-$scene/vggt/fullview_ray_sdf_v081}"
python_bin="${ROOTSPLAT_PYTHON:-python}"
depths="$scene/vggt/fulltrain_depths_v078.npz"
depth_gate="$scene/vggt/fulltrain_depth_gate_v078.json"

test -f "$depths" || { echo "STOP: full-view VGGT depths missing: $depths"; exit 1; }
test -f "$depth_gate" || { echo "STOP: full-view depth gate missing: $depth_gate"; exit 1; }
test ! -e "$output" || { echo "STOP: output already exists: $output"; exit 1; }

echo "===== FULL-42-VIEW RAY-SUPERVISED SDF ====="
"$python_bin" -u scripts/fit_vggt_fullview_ray_sdf.py \
  --scene "$scene" \
  --depths "$depths" \
  --depth-gate "$depth_gate" \
  --output "$output" \
  --steps 1600 \
  --batch-size 2048 \
  --grid-resolution 192

echo "===== GATE SUMMARY ====="
"$python_bin" - "$output/fullview_ray_sdf_gate.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print("status:", report["status"])
print("failures:", report["failures"])
print("train/validation rays:", report["split"])
print("validation field:", report["validation_field"])
print("ray solid:", report["ray_solid"])
print("evidence checks:", report["evidence_gate"]["checks"])
print("topology:", report["mesh_topology"])
if report["status"] != "pass":
    raise SystemExit(2)
PY

echo "FULL-VIEW RAY-SDF GATE: PASS"
echo "Do not train appearance yet. Upload the gate, preview, mesh, and trace."
