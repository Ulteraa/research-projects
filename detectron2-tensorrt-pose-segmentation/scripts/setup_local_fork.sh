#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:-${PROJECT_ROOT}/.vendor/detectron2}"
UPSTREAM="https://github.com/facebookresearch/detectron2.git"
COMMIT="80307d2d5e06f06a8a677cc2653f23a4c56402ac"
PATCH_FILE="${PROJECT_ROOT}/patches/detectron2-keypoints-onnx.patch"
PYTHON_BIN="${PYTHON:-python}"

if [[ -e "${TARGET}" && ! -d "${TARGET}/.git" ]]; then
  echo "Refusing to use an existing non-git path: ${TARGET}" >&2
  exit 1
fi

if [[ ! -d "${TARGET}/.git" ]]; then
  mkdir -p "$(dirname "${TARGET}")"
  git clone --filter=blob:none --no-checkout "${UPSTREAM}" "${TARGET}"
fi

git -C "${TARGET}" fetch --depth 1 origin "${COMMIT}"
git -C "${TARGET}" checkout --detach "${COMMIT}"

if git -C "${TARGET}" apply --unidiff-zero --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "Detectron2 ONNX patch is already applied."
elif git -C "${TARGET}" apply --unidiff-zero --check "${PATCH_FILE}"; then
  git -C "${TARGET}" apply --unidiff-zero "${PATCH_FILE}"
else
  echo "The Detectron2 checkout has overlapping changes; patch not applied." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install -e "${TARGET}"
"${PYTHON_BIN}" - <<'PY'
import detectron2
from detectron2.structures import Keypoints

print("Detectron2", detectron2.__version__, "is importable; Keypoints:", Keypoints.__name__)
PY

echo "Local Detectron2 fork ready at ${TARGET}"
