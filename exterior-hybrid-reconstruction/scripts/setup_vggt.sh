#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VGGT_DIR="${ROOT_DIR}/external/vggt"
mkdir -p "${ROOT_DIR}/external"
if [[ ! -d "${VGGT_DIR}/.git" ]]; then
  git clone https://github.com/facebookresearch/vggt.git "${VGGT_DIR}"
fi
python -m pip install --upgrade pip
python -m pip install -e "${VGGT_DIR}"
echo "VGGT installed at ${VGGT_DIR}"
