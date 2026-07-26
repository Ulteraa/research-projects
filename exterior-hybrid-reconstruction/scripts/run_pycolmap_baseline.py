from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a classical sparse COLMAP baseline using PyCOLMAP."
    )
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    image_dir = args.scene_dir / "images"
    output_dir = args.output_dir
    database_path = output_dir / "database.db"
    sparse_root = output_dir / "sparse"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images: {image_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    sparse_root.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()

    print("Step 1/3: Extracting CPU SIFT features")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        camera_model="SIMPLE_RADIAL",
        device=pycolmap.Device.cpu,
    )

    print("Step 2/3: Exhaustive CPU feature matching")
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        device=pycolmap.Device.cpu,
    )

    print("Step 3/3: Incremental SfM reconstruction")
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(sparse_root),
    )

    if not reconstructions:
        raise RuntimeError("COLMAP did not produce a reconstruction.")

    best_id, best_model = max(
        reconstructions.items(),
        key=lambda item: item[1].num_reg_images(),
    )

    canonical_model_dir = sparse_root / "0"
    canonical_model_dir.mkdir(parents=True, exist_ok=True)
    best_model.write(str(canonical_model_dir))

    elapsed = time.perf_counter() - start

    runtime = {
        "pipeline": "pycolmap",
        "mode": "classical_sparse",
        "runtime_seconds": round(elapsed, 3),
        "num_models": len(reconstructions),
        "selected_model_id": int(best_id),
        "registered_images": int(best_model.num_reg_images()),
        "points3D": int(best_model.num_points3D()),
    }

    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
