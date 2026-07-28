"""Run the Phase 2 zero-noise AX=XB correctness benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibgraph.baselines.opencv_hand_eye import solve_all_opencv_methods
from calibgraph.evaluation.metrics import evaluate_hand_eye_result
from calibgraph.simulation.eye_in_hand import generate_eye_in_hand_dataset


def main() -> None:
    dataset = generate_eye_in_hand_dataset(num_poses=25, seed=7)
    results = solve_all_opencv_methods(dataset)
    rows = [evaluate_hand_eye_result(dataset, result) for result in results]

    frame = pd.DataFrame(rows).sort_values(
        ["translation_error_mm", "rotation_error_deg"]
    )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "phase2_zero_noise.csv"
    frame.to_csv(output_path, index=False)

    print("Phase 2: zero-noise eye-in-hand benchmark")
    print("-----------------------------------------")
    print(f"Poses: {dataset.num_poses}")
    print(frame.to_string(index=False, float_format=lambda x: f"{x:.6e}"))
    print()
    print(f"Saved: {output_path}")

    max_translation_mm = float(frame["translation_error_mm"].max())
    max_rotation_deg = float(frame["rotation_error_deg"].max())

    if max_translation_mm > 1e-3 or max_rotation_deg > 1e-5:
        raise RuntimeError(
            "Zero-noise correctness gate failed: "
            f"max translation={max_translation_mm:.6e} mm, "
            f"max rotation={max_rotation_deg:.6e} deg"
        )

    print("Zero-noise correctness gate: PASS")


if __name__ == "__main__":
    main()
