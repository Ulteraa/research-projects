#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "Usage: $0 SCENE_DIR OUTPUT_DIR {feedforward|ba}"; exit 2; fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SCENE_DIR="$(realpath "$1")"; mkdir -p "$2"; OUTPUT_DIR="$(realpath "$2")"; MODE="$3"; VGGT_DIR="${ROOT_DIR}/external/vggt"
[[ -f "${VGGT_DIR}/demo_colmap.py" ]] || { echo "Run scripts/setup_vggt.sh first" >&2; exit 1; }
STAGE_DIR="${OUTPUT_DIR}/scene"; rm -rf "${STAGE_DIR}"; mkdir -p "${STAGE_DIR}"; cp -a "${SCENE_DIR}/images" "${STAGE_DIR}/images"
START=$(date +%s); pushd "${VGGT_DIR}" >/dev/null
if [[ "${MODE}" == "feedforward" ]]; then python demo_colmap.py --scene_dir "${STAGE_DIR}";
elif [[ "${MODE}" == "ba" ]]; then python demo_colmap.py --scene_dir "${STAGE_DIR}" --use_ba --max_query_pts 2048 --query_frame_num 5;
else echo "Mode must be feedforward or ba" >&2; exit 2; fi
popd >/dev/null
[[ -d "${STAGE_DIR}/sparse" ]] || { echo "VGGT sparse model missing" >&2; exit 1; }
rm -rf "${OUTPUT_DIR}/sparse"; mv "${STAGE_DIR}/sparse" "${OUTPUT_DIR}/sparse"; rm -rf "${STAGE_DIR}"
END=$(date +%s); printf '{"pipeline":"vggt","mode":"%s","runtime_seconds":%d}
' "$MODE" "$((END-START))" > "${OUTPUT_DIR}/runtime.json"
