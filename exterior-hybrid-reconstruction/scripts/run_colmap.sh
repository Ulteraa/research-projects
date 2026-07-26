#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then echo "Usage: $0 SCENE_DIR OUTPUT_DIR {sparse|dense}"; exit 2; fi
SCENE_DIR="$(realpath "$1")"; mkdir -p "$2"; OUTPUT_DIR="$(realpath "$2")"; MODE="$3"
IMAGE_DIR="${SCENE_DIR}/images"; DATABASE_PATH="${OUTPUT_DIR}/database.db"; SPARSE_DIR="${OUTPUT_DIR}/sparse"; DENSE_DIR="${OUTPUT_DIR}/dense"
command -v colmap >/dev/null 2>&1 || { echo "COLMAP not found" >&2; exit 1; }
[[ -d "${IMAGE_DIR}" ]] || { echo "Missing ${IMAGE_DIR}" >&2; exit 1; }
mkdir -p "${SPARSE_DIR}"; rm -f "${DATABASE_PATH}"
START=$(date +%s)
colmap feature_extractor --database_path "${DATABASE_PATH}" --image_path "${IMAGE_DIR}" --ImageReader.camera_model OPENCV --FeatureExtraction.use_gpu 1
colmap exhaustive_matcher --database_path "${DATABASE_PATH}" --FeatureMatching.use_gpu 1
colmap mapper --database_path "${DATABASE_PATH}" --image_path "${IMAGE_DIR}" --output_path "${SPARSE_DIR}"
[[ -d "${SPARSE_DIR}/0" ]] || { echo "Sparse model missing" >&2; exit 1; }
if [[ "${MODE}" == "dense" ]]; then
  mkdir -p "${DENSE_DIR}"
  colmap image_undistorter --image_path "${IMAGE_DIR}" --input_path "${SPARSE_DIR}/0" --output_path "${DENSE_DIR}" --output_type COLMAP --max_image_size 2400
  colmap patch_match_stereo --workspace_path "${DENSE_DIR}" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true --PatchMatchStereo.max_image_size 2400 --PatchMatchStereo.cache_size 8
  colmap stereo_fusion --workspace_path "${DENSE_DIR}" --workspace_format COLMAP --input_type geometric --output_path "${DENSE_DIR}/fused.ply" --StereoFusion.max_image_size 2400 --StereoFusion.cache_size 8
elif [[ "${MODE}" != "sparse" ]]; then echo "Mode must be sparse or dense" >&2; exit 2; fi
END=$(date +%s); printf '{"pipeline":"colmap","mode":"%s","runtime_seconds":%d}
' "$MODE" "$((END-START))" > "${OUTPUT_DIR}/runtime.json"
