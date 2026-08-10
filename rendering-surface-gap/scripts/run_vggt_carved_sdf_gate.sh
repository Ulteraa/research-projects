#!/usr/bin/env bash
set -euo pipefail

scene=${ROOTSPLAT_SCENE:?set ROOTSPLAT_SCENE}
output=${ROOTSPLAT_CARVED_SDF_OUTPUT:?set ROOTSPLAT_CARVED_SDF_OUTPUT}
python_bin=${ROOTSPLAT_PYTHON:-python}

depths="$scene/vggt/fulltrain_depths_v078.npz"
depth_gate="$scene/vggt/fulltrain_depth_gate_v078.json"

test -f "$depths" || {
  echo "STOP: cached v0.7.8 training-view depths are missing: $depths"
  exit 1
}
test -f "$depth_gate" || {
  echo "STOP: cached v0.7.8 depth gate is missing: $depth_gate"
  exit 1
}
test ! -e "$output" || {
  echo "STOP: carved-SDF output already exists: $output"
  exit 1
}

echo "===== STRICT VISUAL HULL + VGGT FREE-SPACE CARVING ====="
"$python_bin" -u scripts/prepare_vggt_carved_sdf.py \
  --scene "$scene" \
  --depths "$depths" \
  --depth-gate "$depth_gate" \
  --output "$output"

echo "VGGT CARVED-SDF GATE: COMPLETE"
echo "Do not train unless the JSON status is pass and the mesh is visually correct."
