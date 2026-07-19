#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/third_party/ultralytics"
UPSTREAM_TAG="v8.2.74"
UPSTREAM_COMMIT="9f593318542e9fc38de6b30b070673104a3c6f28"

python -m pip install -r "${ROOT_DIR}/requirements.txt"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${UPSTREAM_DIR}")"
  git clone --depth 1 --branch "${UPSTREAM_TAG}" \
    https://github.com/ultralytics/ultralytics.git "${UPSTREAM_DIR}"
fi

actual_commit="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${UPSTREAM_COMMIT}" ]]; then
  echo "Expected ${UPSTREAM_TAG} at ${UPSTREAM_COMMIT}, found ${actual_commit}." >&2
  echo "Remove third_party/ultralytics and run this script again." >&2
  exit 1
fi

cp -a "${ROOT_DIR}/overlay/ultralytics/." "${UPSTREAM_DIR}/ultralytics/"
python -m pip install --no-deps -e "${UPSTREAM_DIR}"

python - <<'PY'
from ultralytics import __version__
from ultralytics.nn.modules import PoseSegment

print(f"Installed YOLOv8+ overlay on Ultralytics {__version__}")
print(f"Joint head available: {PoseSegment.__name__}")
PY
