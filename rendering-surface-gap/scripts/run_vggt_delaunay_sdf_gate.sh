#!/usr/bin/env bash
set -euo pipefail

scene=${ROOTSPLAT_SCENE:?set ROOTSPLAT_SCENE}
output=${ROOTSPLAT_DELAUNAY_OUTPUT:?set ROOTSPLAT_DELAUNAY_OUTPUT}
python_bin=${ROOTSPLAT_PYTHON:-python}
open3d_python=${ROOTSPLAT_OPEN3D_PYTHON:-$python_bin}
colmap_bin=${COLMAP_BIN:-/workspace/venvs/colmap/bin/colmap}
depths=${ROOTSPLAT_DEPTHS:-$scene/vggt/fulltrain_depths_v078.npz}
depth_gate=${ROOTSPLAT_DEPTH_GATE:-$scene/vggt/fulltrain_depth_gate_v078.json}

test -f "$depths" || { echo "STOP: depth cache missing: $depths"; exit 1; }
test -f "$depth_gate" || { echo "STOP: depth gate missing: $depth_gate"; exit 1; }
test -x "$colmap_bin" || { echo "STOP: COLMAP missing: $colmap_bin"; exit 1; }
test ! -e "$output" || { echo "STOP: output already exists: $output"; exit 1; }

"$colmap_bin" delaunay_mesher -h >/dev/null 2>&1 || {
  echo "STOP: this COLMAP build has no CGAL Delaunay mesher"
  exit 1
}
PYTHONPATH="$PWD" "$open3d_python" -c \
  'import open3d, rootsplat; print("Open3D:", open3d.__version__, "RootSplat:", rootsplat.__version__)'

echo "===== EXPORT 42-VIEW DEPTH + VISIBILITY WORKSPACE ====="
"$python_bin" -u scripts/export_vggt_delaunay_workspace.py \
  --scene "$scene" \
  --depths "$depths" \
  --depth-gate "$depth_gate" \
  --output "$output/workspace"

echo "===== COLMAP DELAUNAY VISIBILITY GRAPH CUT ====="
"$colmap_bin" delaunay_mesher \
  --input_path "$output/workspace" \
  --input_type dense \
  --output_path "$output/surface_raw.ply"

test -s "$output/surface_raw.ply" || {
  echo "STOP: COLMAP did not emit a non-empty mesh"
  exit 1
}

echo "===== CLOSED MESH -> METRIC GRID SDF -> EVIDENCE GATE ====="
PYTHONPATH="$PWD" "$open3d_python" -u scripts/prepare_vggt_delaunay_sdf.py \
  --scene "$scene" \
  --depths "$depths" \
  --depth-gate "$depth_gate" \
  --workspace-gate "$output/workspace/workspace_gate.json" \
  --mesh "$output/surface_raw.ply" \
  --output "$output/result"

echo "VGGT DELAUNAY-SDF GATE: COMPLETE"
echo "Do not train appearance until result/delaunay_sdf_gate.json passes and the PLY is visually correct."
