"""Monte Carlo benchmark under synthetic target-pose observation noise."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibgraph.baselines.opencv_hand_eye import (
    HAND_EYE_METHODS,
    solve_opencv_hand_eye,
)
from calibgraph.evaluation.metrics import evaluate_hand_eye_result
from calibgraph.simulation.eye_in_hand import generate_eye_in_hand_dataset
from calibgraph.simulation.noise import add_target_pose_noise


NOISE_REGIMES = (
    {
        "noise_level": 0,
        "noise_label": "ideal",
        "translation_sigma_mm": 0.0,
        "rotation_sigma_deg_input": 0.0,
    },
    {
        "noise_level": 1,
        "noise_label": "very-low",
        "translation_sigma_mm": 0.10,
        "rotation_sigma_deg_input": 0.05,
    },
    {
        "noise_level": 2,
        "noise_label": "low",
        "translation_sigma_mm": 0.25,
        "rotation_sigma_deg_input": 0.10,
    },
    {
        "noise_level": 3,
        "noise_label": "moderate",
        "translation_sigma_mm": 0.50,
        "rotation_sigma_deg_input": 0.25,
    },
    {
        "noise_level": 4,
        "noise_label": "high",
        "translation_sigma_mm": 1.00,
        "rotation_sigma_deg_input": 0.50,
    },
    {
        "noise_level": 5,
        "noise_label": "severe",
        "translation_sigma_mm": 2.00,
        "rotation_sigma_deg_input": 1.00,
    },
)


def run_benchmark(
    *,
    trials: int = 30,
    num_poses: int = 25,
    noise_regimes: tuple[dict[str, object], ...] = NOISE_REGIMES,
) -> pd.DataFrame:
    """Run all OpenCV solvers across trajectories and noise realizations."""
    if trials < 1:
        raise ValueError("trials must be positive")
    if num_poses < 3:
        raise ValueError("num_poses must be at least 3")

    rows: list[dict[str, object]] = []

    for regime in noise_regimes:
        for trial in range(trials):
            trajectory_seed = 10_000 + trial
            noise_seed = (
                100_000
                + int(regime["noise_level"]) * 10_000
                + trial
            )

            clean_dataset = generate_eye_in_hand_dataset(
                num_poses=num_poses,
                seed=trajectory_seed,
            )
            observed_dataset = add_target_pose_noise(
                clean_dataset,
                seed=noise_seed,
                translation_sigma_m=(
                    float(regime["translation_sigma_mm"]) / 1000.0
                ),
                rotation_sigma_deg=float(
                    regime["rotation_sigma_deg_input"]
                ),
            )

            for method in HAND_EYE_METHODS:
                start = perf_counter()
                try:
                    result = solve_opencv_hand_eye(
                        observed_dataset,
                        method=method,
                    )
                    metrics = evaluate_hand_eye_result(
                        clean_dataset,
                        result,
                    )
                    success = bool(
                        np.isfinite(metrics["translation_error_mm"])
                        and np.isfinite(metrics["rotation_error_deg"])
                    )
                    error_message = ""
                except Exception as exc:  # benchmark must record failures
                    metrics = {
                        "translation_error_mm": np.nan,
                        "rotation_error_deg": np.nan,
                        "mean_target_error_mm": np.nan,
                        "max_target_error_mm": np.nan,
                        "mean_target_rotation_error_deg": np.nan,
                        "max_target_rotation_error_deg": np.nan,
                    }
                    success = False
                    error_message = f"{type(exc).__name__}: {exc}"

                elapsed_ms = (perf_counter() - start) * 1000.0
                rows.append(
                    {
                        **regime,
                        "trial": trial,
                        "num_poses": num_poses,
                        "trajectory_seed": trajectory_seed,
                        "noise_seed": noise_seed,
                        "method": method,
                        "success": success,
                        "runtime_ms": elapsed_ms,
                        "error_message": error_message,
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def summarize_trials(trials_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean, standard deviation, median, runtime, and failure rate."""
    grouped = trials_frame.groupby(
        [
            "noise_level",
            "noise_label",
            "translation_sigma_mm",
            "rotation_sigma_deg_input",
            "method",
        ],
        as_index=False,
        dropna=False,
    )

    summary = grouped.agg(
        trials=("trial", "count"),
        successful_trials=("success", "sum"),
        translation_error_mm_mean=("translation_error_mm", "mean"),
        translation_error_mm_std=("translation_error_mm", "std"),
        translation_error_mm_median=("translation_error_mm", "median"),
        rotation_error_deg_mean=("rotation_error_deg", "mean"),
        rotation_error_deg_std=("rotation_error_deg", "std"),
        rotation_error_deg_median=("rotation_error_deg", "median"),
        mean_target_error_mm_mean=("mean_target_error_mm", "mean"),
        runtime_ms_mean=("runtime_ms", "mean"),
    )
    summary["failure_rate"] = (
        1.0 - summary["successful_trials"] / summary["trials"]
    )
    return summary.sort_values(["noise_level", "method"]).reset_index(drop=True)


def _noise_tick_labels(summary: pd.DataFrame) -> tuple[list[int], list[str]]:
    unique = (
        summary[
            [
                "noise_level",
                "translation_sigma_mm",
                "rotation_sigma_deg_input",
            ]
        ]
        .drop_duplicates()
        .sort_values("noise_level")
    )
    positions = unique["noise_level"].astype(int).tolist()
    labels = [
        f"{row.translation_sigma_mm:g} mm\n{row.rotation_sigma_deg_input:g}°"
        for row in unique.itertuples()
    ]
    return positions, labels


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create separate translation- and rotation-error figures."""
    positions, labels = _noise_tick_labels(summary)

    plt.figure(figsize=(9, 5.5))
    for method, method_frame in summary.groupby("method"):
        ordered = method_frame.sort_values("noise_level")
        plt.errorbar(
            ordered["noise_level"],
            ordered["translation_error_mm_mean"],
            yerr=ordered["translation_error_mm_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xticks(positions, labels)
    plt.xlabel("Injected target-pose noise: translation σ / rotation σ")
    plt.ylabel("Camera-to-gripper translation error (mm)")
    plt.title("Hand–Eye Calibration Robustness to Target-Pose Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase3_translation_error_vs_noise.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for method, method_frame in summary.groupby("method"):
        ordered = method_frame.sort_values("noise_level")
        plt.errorbar(
            ordered["noise_level"],
            ordered["rotation_error_deg_mean"],
            yerr=ordered["rotation_error_deg_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=method,
        )
    plt.xticks(positions, labels)
    plt.xlabel("Injected target-pose noise: translation σ / rotation σ")
    plt.ylabel("Camera-to-gripper rotation error (degrees)")
    plt.title("Hand–Eye Rotation Robustness to Target-Pose Noise")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase3_rotation_error_vs_noise.png",
        dpi=180,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hand-eye solvers under target-pose noise."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Monte Carlo trials per noise regime (default: 10)",
    )
    parser.add_argument(
        "--num-poses",
        type=int,
        default=25,
        help="Robot poses per trial (default: 25)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_frame = run_benchmark(
        trials=args.trials,
        num_poses=args.num_poses,
    )
    summary = summarize_trials(trials_frame)

    trials_path = output_dir / "phase3_target_pose_noise_trials.csv"
    summary_path = output_dir / "phase3_target_pose_noise_summary.csv"
    trials_frame.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, output_dir)

    columns = [
        "noise_label",
        "translation_sigma_mm",
        "rotation_sigma_deg_input",
        "method",
        "translation_error_mm_mean",
        "translation_error_mm_std",
        "rotation_error_deg_mean",
        "rotation_error_deg_std",
        "failure_rate",
        "runtime_ms_mean",
    ]

    print("Phase 3: target-pose noise Monte Carlo benchmark")
    print("------------------------------------------------")
    print(f"Trials per regime and method: {args.trials}")
    print(f"Robot poses per trial: {args.num_poses}")
    print(
        summary[columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Saved: {trials_path}")
    print(f"Saved: {summary_path}")
    print(
        "Saved: "
        f"{output_dir / 'phase3_translation_error_vs_noise.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase3_rotation_error_vs_noise.png'}"
    )

    if not bool(trials_frame["success"].all()):
        failures = int((~trials_frame["success"]).sum())
        raise RuntimeError(f"{failures} solver trials failed")

    ideal = summary[summary["noise_level"] == 0]
    if float(ideal["translation_error_mm_mean"].max()) > 1e-3:
        raise RuntimeError("ideal translation correctness gate failed")
    if float(ideal["rotation_error_deg_mean"].max()) > 1e-5:
        raise RuntimeError("ideal rotation correctness gate failed")

    print("Phase 3 benchmark gate: PASS")

