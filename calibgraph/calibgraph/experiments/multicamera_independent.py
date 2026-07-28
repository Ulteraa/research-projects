"""Benchmark independent calibration for cameras on different robot links."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.evaluation.multicamera_metrics import (
    evaluate_independent_multicamera,
)
from calibgraph.evaluation.observability import analyze_motion_observability
from calibgraph.simulation.articulated_multicamera import (
    generate_articulated_multicamera_dataset,
)
from calibgraph.simulation.multicamera_noise import (
    add_multicamera_target_pose_noise,
)


def run_multicamera_benchmark(
    *,
    trials: int = 20,
    num_poses: int = 30,
    method: str = "PARK",
    translation_sigma_mm: float = 0.50,
    rotation_sigma_deg: float = 0.25,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for trial in range(trials):
        clean = generate_articulated_multicamera_dataset(
            num_poses=num_poses,
            seed=40_000 + trial,
        )
        observed = add_multicamera_target_pose_noise(
            clean,
            seed=50_000 + trial,
            translation_sigma_m=translation_sigma_mm / 1000.0,
            rotation_sigma_deg=rotation_sigma_deg,
        )

        start = perf_counter()
        result = solve_independent_multicamera(
            observed,
            method=method,
        )
        runtime_ms = (perf_counter() - start) * 1000.0

        metrics = evaluate_independent_multicamera(clean, result)
        for row in metrics:
            camera_dataset = clean.as_hand_eye_dataset(str(row["camera_name"]))
            observability = analyze_motion_observability(camera_dataset)
            rows.append(
                {
                    "trial": trial,
                    "num_poses": num_poses,
                    "translation_sigma_mm": translation_sigma_mm,
                    "rotation_sigma_deg_input": rotation_sigma_deg,
                    "runtime_ms_total": runtime_ms,
                    **observability.to_dict(),
                    **row,
                }
            )

    return pd.DataFrame(rows)


def summarize_multicamera_trials(trials: pd.DataFrame) -> pd.DataFrame:
    grouped = trials.groupby(
        ["method", "camera_name", "link_name", "quality"],
        as_index=False,
    )
    summary = grouped.agg(
        trials=("trial", "count"),
        translation_error_mm_mean=("translation_error_mm", "mean"),
        translation_error_mm_std=("translation_error_mm", "std"),
        rotation_error_deg_mean=("rotation_error_deg", "mean"),
        rotation_error_deg_std=("rotation_error_deg", "std"),
        mean_target_error_mm_mean=("mean_target_error_mm", "mean"),
        mean_cross_camera_target_disagreement_mm=(
            "mean_cross_camera_target_disagreement_mm",
            "mean",
        ),
        runtime_ms_total_mean=("runtime_ms_total", "mean"),
        max_relative_rotation_deg=("max_relative_rotation_deg", "mean"),
        rotation_axis_diversity_ratio=(
            "rotation_axis_diversity_ratio",
            "mean",
        ),
        rotation_design_rank=("rotation_design_rank", "min"),
    )
    return summary.sort_values("camera_name").reset_index(drop=True)


def plot_multicamera_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary.sort_values("camera_name")

    plt.figure(figsize=(8.5, 5.2))
    plt.bar(
        ordered["camera_name"],
        ordered["translation_error_mm_mean"],
        yerr=ordered["translation_error_mm_std"].fillna(0.0),
        capsize=4,
    )
    plt.ylabel("Camera-to-link translation error (mm)")
    plt.xlabel("Camera mounting link")
    plt.title("Independent Multi-Camera Calibration Error")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase5_translation_error_by_camera.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    plt.bar(
        ordered["camera_name"],
        ordered["rotation_error_deg_mean"],
        yerr=ordered["rotation_error_deg_std"].fillna(0.0),
        capsize=4,
    )
    plt.ylabel("Camera-to-link rotation error (degrees)")
    plt.xlabel("Camera mounting link")
    plt.title("Rotation Error by Camera Mount")
    plt.xticks(rotation=15)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase5_rotation_error_by_camera.png",
        dpi=180,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark independent calibration of multiple robot-mounted cameras."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num-poses", type=int, default=30)
    parser.add_argument("--method", type=str, default="PARK")
    parser.add_argument("--translation-sigma-mm", type=float, default=0.50)
    parser.add_argument("--rotation-sigma-deg", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_multicamera_benchmark(
        trials=args.trials,
        num_poses=args.num_poses,
        method=args.method,
        translation_sigma_mm=args.translation_sigma_mm,
        rotation_sigma_deg=args.rotation_sigma_deg,
    )
    summary = summarize_multicamera_trials(trials)

    trials_path = output_dir / "phase5_multicamera_trials.csv"
    summary_path = output_dir / "phase5_multicamera_summary.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_multicamera_summary(summary, output_dir)

    print("Phase 5: articulated multi-camera independent calibration")
    print("---------------------------------------------------------")
    print(f"Method: {args.method.upper()}")
    print(f"Trials: {args.trials}")
    print(f"Robot poses per trial: {args.num_poses}")
    print(
        f"Observation noise: {args.translation_sigma_mm:g} mm / "
        f"{args.rotation_sigma_deg:g} deg"
    )
    print()
    print(
        summary[
            [
                "camera_name",
                "link_name",
                "quality",
                "rotation_design_rank",
                "rotation_axis_diversity_ratio",
                "translation_error_mm_mean",
                "rotation_error_deg_mean",
                "mean_target_error_mm_mean",
                "mean_cross_camera_target_disagreement_mm",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Saved: {trials_path}")
    print(f"Saved: {summary_path}")
    print(
        "Saved: "
        f"{output_dir / 'phase5_translation_error_by_camera.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase5_rotation_error_by_camera.png'}"
    )

    if not bool((summary["quality"] == "GOOD").all()):
        raise RuntimeError("one or more camera-link trajectories are not observable")
    if float(summary["translation_error_mm_mean"].max()) > 2.0:
        raise RuntimeError("multi-camera translation gate failed")
    if float(summary["rotation_error_deg_mean"].max()) > 0.5:
        raise RuntimeError("multi-camera rotation gate failed")

    print("Phase 5 benchmark gate: PASS")


if __name__ == "__main__":
    main()
