#!/usr/bin/env bash
set -euo pipefail

rootsplat_scene=${ROOTSPLAT_SCENE:-/workspace/data/DTU/dtu_scan24}
rootsplat_python=${ROOTSPLAT_PYTHON:-python}
vggt_python=${VGGT_PYTHON:-/workspace/venvs/vggt/bin/python}
vggt_source=${VGGT_SOURCE:-/workspace/external/vggt}
vggt_views=${VGGT_VIEWS:-16}
vggt_dir="$rootsplat_scene/vggt"

test -f "$rootsplat_scene/cameras_sphere.npz" || {
  echo "STOP: missing $rootsplat_scene/cameras_sphere.npz"
  exit 1
}
test -x "$vggt_python" || {
  echo "STOP: VGGT Python is missing: $vggt_python"
  exit 1
}
test -f "$vggt_source/vggt/models/vggt.py" || {
  echo "STOP: official VGGT source is missing: $vggt_source"
  exit 1
}

mkdir -p "$vggt_dir"

echo "===== ROOTSPLAT CPU CONTRACT ====="
"$rootsplat_python" tests/test_torch_runtime.py

echo "===== FROZEN VGGT INFERENCE ====="
PYTHONPATH="$vggt_source${PYTHONPATH:+:$PYTHONPATH}" \
  "$vggt_python" scripts/export_vggt_dtu.py \
    --scene "$rootsplat_scene" \
    --output "$vggt_dir/raw_predictions.npz" \
    --report "$vggt_dir/raw_predictions.json" \
    --max-views "$vggt_views"

echo "===== CALIBRATED INITIALIZER GATE ====="
"$rootsplat_python" scripts/prepare_vggt_initializer.py \
  --raw "$vggt_dir/raw_predictions.npz" \
  --scene "$rootsplat_scene" \
  --surface "$vggt_dir/initializer.ply" \
  --report "$vggt_dir/initializer_gate.json" \
  --preview "$vggt_dir/initializer_preview.png"

echo "===== GATE SUMMARY ====="
"$rootsplat_python" - "$vggt_dir/initializer_gate.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf8"))
print("status:", report["status"])
print("failures:", report["failures"])
print("surface points:", report["output"]["points"])
print("alignment median:", report["alignment"]["residual_normalized_median"])
print("alignment inlier p95:", report["alignment"]["residual_normalized_inlier_p95"])
print("cross-view retained:", report["cross_view"]["retained_fraction"])
print("mask retained:", report["masks"]["retained_fraction"])
print("surface SHA-256:", report["surface_sha256"])
PY

echo "VGGT INITIALIZER SMOKE: PASS"
echo "Do not train yet. Upload initializer_gate.json and initializer_preview.png."
