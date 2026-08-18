#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/research-projects" >&2
  exit 2
fi

REPO="$(cd "$1" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/bevfusion-reliable-refinement"
DEST="$REPO/bevfusion-reliable-refinement"
README="$REPO/README.md"

if [[ ! -d "$REPO/.git" || ! -f "$README" ]]; then
  echo "Not a research-projects Git checkout: $REPO" >&2
  exit 1
fi

if [[ -e "$DEST" ]]; then
  echo "Destination already exists: $DEST" >&2
  echo "Move or remove it before running this installer." >&2
  exit 1
fi

cp -a "$SRC" "$DEST"

python - "$README" "$HERE/RESEARCH_PROJECTS_README_ENTRY.md" <<'PY'
from pathlib import Path
import sys

readme = Path(sys.argv[1])
entry_file = Path(sys.argv[2])
text = readme.read_text()
entry = entry_file.read_text().rstrip() + "\n\n"
heading = "### [Reliable Camera–LiDAR BEV Fusion with Proposal-Level 3D Box Refinement]"

if heading not in text:
    marker = "## About"
    if marker not in text:
        raise SystemExit("Could not find '## About' in the root README.")
    text = text.replace(marker, entry + marker, 1)
    readme.write_text(text)
PY

echo "Added project folder and updated root README."
echo
printf '%s\n' \
  "cd '$REPO'" \
  "git add README.md bevfusion-reliable-refinement" \
  "git commit -m 'Add reliable BEVFusion refinement project'" \
  "git push origin main"
