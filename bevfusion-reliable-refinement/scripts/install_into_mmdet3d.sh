#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/mmdetection3d" >&2
  exit 2
fi

ROOT="$(cd "$1" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/projects/BEVFusion"

if [[ ! -d "$DEST/bevfusion" || ! -d "$DEST/configs" ]]; then
  echo "Expected MMDetection3D BEVFusion project under: $DEST" >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
cp "$DEST/bevfusion/transfusion_head.py" \
   "$DEST/bevfusion/transfusion_head.py.backup_$stamp"
cp "$DEST/bevfusion/__init__.py" \
   "$DEST/bevfusion/__init__.py.backup_$stamp"

cp "$HERE/src/transfusion_head.py" "$DEST/bevfusion/transfusion_head.py"
cp "$HERE/src/__init__.py" "$DEST/bevfusion/__init__.py"
cp "$HERE/configs/"*.py "$DEST/configs/"

echo "Installed custom BEVFusion modules and configs into: $DEST"
echo "Backups use suffix: backup_$stamp"
