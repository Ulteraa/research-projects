"""Benchmark camera/robot time synchronization under articulated motion."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.evaluation.time_sync_metrics import (
    evaluate_time_sync_estimate,
)
from calibgraph.graph.time_offset import (
    solve_time_aware_multicamera,
)
from calibgraph.simulation.time_sync import (
    generate_time_offset_dataset,
)


OFFSET_LEVELS_MS: tuple[int, ...] = (0, 20, 50, 100)


def _offset_pattern(level_ms: int) -> dict[str, float]:
    """Assign distinct offsets so synchronization is identifiable per camera."""
    scale_s = level_ms / 1000.0
    return {
        "upper_arm_camera": 0.30 * scale_s,
        "forearm_camera": 1.00 * scale_s,
        "wrist_camera": -0.80 * scale_s,
    }


def run_time_sync_benchmark(
    *,
    trials: int = 3,
    num_poses: int = 24,
    duration_s: float = 5.0,
    iterations: int = 3,
    offset_levels_ms: tuple[int, ...] = OFFSET_LEVELS_MS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for level_ms in offset_levels_ms:
        true_offsets = _offset_pattern(level_ms)

        for trial in range(trials):
            dataset = generate_time_offset_dataset(
                num_poses=num_poses,
                duration_s=duration_s,
                time_offsets_s=true_offsets,
                observation_translation_sigma_mm=0.25,
                observation_rotation_sigma_deg=0.10,
                seed=90_000 + level_ms * 100 + trial,
            )

            zero_data = dataset.as_zero_offset_multicamera_dataset()
            start = perf_counter()
            baseline = solve_independent_multicamera(
                zero_data,
                method="PARK",
            )
            baseline_runtime_ms = (
                perf_counter() - start
            ) * 1000.0
            baseline_offsets = {
                name: 0.0 for name in dataset.camera_names
            }
            baseline_rows = evaluate_time_sync_estimate(
                dataset,
                method="PARK_ZERO_OFFSET",
                camera_extrinsics=baseline.camera_extrinsics,
                time_offsets_s=baseline_offsets,
            )
            for row in baseline_rows:
                rows.append(
                    {
                        "offset_level_ms": level_ms,
                        "trial": trial,
                        "num_poses": num_poses,
                        "runtime_ms": baseline_runtime_ms,
                        "success": True,
                        "nfev": 0,
                        **row,
                    }
                )

            time_aware = solve_time_aware_multicamera(
                dataset,
                translation_sigma_m=0.00025,
                rotation_sigma_deg=0.10,
                iterations=iterations,
            )
            time_aware_rows = evaluate_time_sync_estimate(
                dataset,
                method=time_aware.method,
                camera_extrinsics=time_aware.camera_extrinsics,
                time_offsets_s=time_aware.time_offsets_s,
            )
            for row in time_aware_rows:
                rows.append(
                    {
                        "offset_level_ms": level_ms,
                        "trial": trial,
                        "num_poses": num_poses,
                        "runtime_ms": time_aware.runtime_ms,
                        "success": time_aware.success,
                        "nfev": time_aware.nfev,
                        **row,
                    }
                )

    return pd.DataFrame(rows)


def summarize_time_sync_benchmark(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        trials.groupby(
            ["offset_level_ms", "method"],
            as_index=False,
        )
        .agg(
            trials=("trial", "nunique"),
            success_rate=("success", "mean"),
            translation_error_mm_mean=(
                "translation_error_mm",
                "mean",
            ),
            rotation_error_deg_mean=(
                "rotation_error_deg",
                "mean",
            ),
            time_offset_abs_error_ms_mean=(
                "time_offset_abs_error_ms",
                "mean",
            ),
            target_error_mm_mean=(
                "mean_target_error_mm",
                "mean",
            ),
            cross_camera_disagreement_mm_mean=(
                "mean_cross_camera_target_disagreement_mm",
                "mean",
            ),
            runtime_ms_mean=("runtime_ms", "mean"),
            nfev_mean=("nfev", "mean"),
        )
        .sort_values(["offset_level_ms", "method"])
        .reset_index(drop=True)
    )
    return summary


def plot_time_sync_summary(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    methods = tuple(summary["method"].unique())

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        frame = summary[summary["method"] == method]
        plt.plot(
            frame["offset_level_ms"],
            frame["translation_error_mm_mean"],
            marker="o",
            label=method,
        )
    plt.xlabel("Maximum injected camera time offset (ms)")
    plt.ylabel("Mean camera-to-link translation error (mm)")
    plt.title("Calibration Sensitivity to Camera/Robot Time Offset")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase8_translation_error_vs_time_offset.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    time_aware = summary[
        summary["method"] == "TIME_AWARE_PARK"
    ]
    plt.plot(
        time_aware["offset_level_ms"],
        time_aware["time_offset_abs_error_ms_mean"],
        marker="o",
    )
    plt.xlabel("Maximum injected camera time offset (ms)")
    plt.ylabel("Mean absolute estimated offset error (ms)")
    plt.title("Camera Time-Offset Estimation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase8_time_offset_estimation_error.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8.5, 5.2))
    for method in methods:
        frame = summary[summary["method"] == method]
        plt.plot(
            frame["offset_level_ms"],
            frame["cross_camera_disagreement_mm_mean"],
            marker="o",
            label=method,
        )
    plt.xlabel("Maximum injected camera time offset (ms)")
    plt.ylabel("Mean cross-camera target disagreement (mm)")
    plt.title("Cross-Camera Consistency After Time Synchronization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase8_consistency_vs_time_offset.png",
        dpi=180,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark camera/robot timestamp synchronization."
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--num-poses", type=int, default=24)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_time_sync_benchmark(
        trials=args.trials,
        num_poses=args.num_poses,
        duration_s=args.duration_s,
        iterations=args.iterations,
    )
    summary = summarize_time_sync_benchmark(trials)

    trials_path = output_dir / "phase8_time_sync_trials.csv"
    summary_path = output_dir / "phase8_time_sync_summary.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_time_sync_summary(summary, output_dir)

    print("Phase 8: camera/robot time synchronization")
    print("------------------------------------------")
    print(f"Trials per offset level: {args.trials}")
    print(f"Robot poses per trial: {args.num_poses}")
    print(f"Alternating refinement iterations: {args.iterations}")
    print()
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Saved: {trials_path}")
    print(f"Saved: {summary_path}")
    print(
        "Saved: "
        f"{output_dir / 'phase8_translation_error_vs_time_offset.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase8_time_offset_estimation_error.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase8_consistency_vs_time_offset.png'}"
    )

    zero = summary[
        (summary["offset_level_ms"] == 0)
        & (summary["method"] == "TIME_AWARE_PARK")
    ].iloc[0]
    severe_baseline = summary[
        (summary["offset_level_ms"] == 100)
        & (summary["method"] == "PARK_ZERO_OFFSET")
    ].iloc[0]
    severe_time_aware = summary[
        (summary["offset_level_ms"] == 100)
        & (summary["method"] == "TIME_AWARE_PARK")
    ].iloc[0]

    if float(zero["time_offset_abs_error_ms_mean"]) > 3.0:
        raise RuntimeError("zero-offset estimation gate failed")
    if float(severe_time_aware["success_rate"]) < 1.0:
        raise RuntimeError("time-aware solver convergence gate failed")
    if not (
        float(severe_time_aware["translation_error_mm_mean"])
        < 0.35
        * float(severe_baseline["translation_error_mm_mean"])
    ):
        raise RuntimeError("time-aware calibration improvement gate failed")

    print("Phase 8 benchmark gate: PASS")


if __name__ == "__main__":
    main()
