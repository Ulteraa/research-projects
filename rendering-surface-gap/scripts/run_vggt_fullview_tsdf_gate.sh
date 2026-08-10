#!/usr/bin/env bash
set -euo pipefail

scene=${ROOTSPLAT_SCENE:?set ROOTSPLAT_SCENE}
raw=${ROOTSPLAT_VGGT_FULLTRAIN_RAW:?set ROOTSPLAT_VGGT_FULLTRAIN_RAW}
output=${ROOTSPLAT_FULLVIEW_TSDF_OUTPUT:?set ROOTSPLAT_FULLVIEW_TSDF_OUTPUT}
python_bin=${ROOTSPLAT_PYTHON:-python}

depths="$scene/vggt/fulltrain_depths_v078.npz"
depth_gate="$scene/vggt/fulltrain_depth_gate_v078.json"

test ! -e "$depths" || {
  echo "STOP: calibrated full-training depth archive already exists: $depths"
  exit 1
}
test ! -e "$depth_gate" || {
  echo "STOP: calibrated full-training depth gate already exists: $depth_gate"
  exit 1
}
test ! -e "$output" || {
  echo "STOP: full-view TSDF output already exists: $output"
  exit 1
}

echo "===== CALIBRATE ALL TRAINING-VIEW VGGT DEPTHS ====="
"$python_bin" -u scripts/calibrate_vggt_fulltrain_depths.py \
  --scene "$scene" \
  --input "$raw" \
  --output "$depths" \
  --report "$depth_gate"

echo "===== CONSENSUS TSDF -> MARCHING CUBES -> GRID SDF ====="
"$python_bin" -u scripts/prepare_vggt_fullview_tsdf.py \
  --scene "$scene" \
  --depths "$depths" \
  --depth-gate "$depth_gate" \
  --output "$output"

echo "VGGT FULL-VIEW TSDF GATE: COMPLETE"
echo "Do not train appearance until the JSON status is pass and the preview is correct."
