"""Held-out downstream task-space validation across calibration failure modes."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd

from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.evaluation.task_space import (
    evaluate_multicamera_task_space,
    evaluate_time_sync_task_space,
)
from calibgraph.graph.joint_multicamera import (
    solve_joint_multicamera,
)
from calibgraph.graph.time_offset import (
    solve_time_aware_multicamera,
)
from calibgraph.simulation.articulated_multicamera import (
    generate_articulated_multicamera_dataset,
)
from calibgraph.simulation.multicamera_noise import (
    add_multicamera_mixed_noise,
)
from calibgraph.simulation.time_sync import (
    generate_time_offset_dataset,
)


def _time_offset_pattern() -> dict[str, float]:
    return {
        "upper_arm_camera": 0.030,
        "forearm_camera": 0.100,
        "wrist_camera": -0.080,
    }


def _append_rows(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    trial: int,
    calibration_runtime_ms: float,
    task_rows: list[dict[str, object]],
) -> None:
    for row in task_rows:
        rows.append(
            {
                "scenario": scenario,
                "trial": trial,
                "calibration_runtime_ms": calibration_runtime_ms,
                **row,
            }
        )


def run_task_space_benchmark(
    *,
    trials: int = 5,
    calibration_poses: int = 30,
    validation_poses: int = 36,
    time_iterations: int = 3,
) -> pd.DataFrame:
    """Calibrate on one sequence and evaluate on a distinct held-out sequence."""
    rows: list[dict[str, object]] = []

    for trial in range(trials):
        # Scenario 1: ordinary Gaussian target-pose noise.
        clean_calibration = generate_articulated_multicamera_dataset(
            num_poses=calibration_poses,
            seed=100_000 + trial,
        )
        gaussian_calibration = add_multicamera_mixed_noise(
            clean_calibration,
            seed=110_000 + trial,
            translation_sigma_m=0.0005,
            rotation_sigma_deg=0.25,
            outlier_probability=0.0,
            outlier_translation_sigma_m=0.0,
            outlier_rotation_sigma_deg=0.0,
        )
        clean_validation = generate_articulated_multicamera_dataset(
            num_poses=validation_poses,
            seed=120_000 + trial,
        )
        gaussian_validation = add_multicamera_mixed_noise(
            clean_validation,
            seed=130_000 + trial,
            translation_sigma_m=0.00025,
            rotation_sigma_deg=0.10,
            outlier_probability=0.0,
            outlier_translation_sigma_m=0.0,
            outlier_rotation_sigma_deg=0.0,
        )

        start = perf_counter()
        park_gaussian = solve_independent_multicamera(
            gaussian_calibration,
            method="PARK",
        )
        park_runtime = (perf_counter() - start) * 1000.0
        _append_rows(
            rows,
            scenario="gaussian",
            trial=trial,
            calibration_runtime_ms=park_runtime,
            task_rows=evaluate_multicamera_task_space(
                gaussian_validation,
                method="PARK",
                camera_extrinsics=park_gaussian.camera_extrinsics,
            ),
        )

        huber_gaussian = solve_joint_multicamera(
            gaussian_calibration,
            initialization_method="PARK",
            loss="huber",
            translation_sigma_m=0.0005,
            rotation_sigma_deg=0.25,
        )
        _append_rows(
            rows,
            scenario="gaussian",
            trial=trial,
            calibration_runtime_ms=huber_gaussian.runtime_ms,
            task_rows=evaluate_multicamera_task_space(
                gaussian_validation,
                method="JOINT_HUBER",
                camera_extrinsics=huber_gaussian.camera_extrinsics,
            ),
        )

        # Scenario 2: corrupted calibration observations, clean held-out test.
        outlier_calibration = add_multicamera_mixed_noise(
            clean_calibration,
            seed=140_000 + trial,
            translation_sigma_m=0.0005,
            rotation_sigma_deg=0.25,
            outlier_probability=0.08,
            outlier_translation_sigma_m=0.020,
            outlier_rotation_sigma_deg=5.0,
        )

        start = perf_counter()
        park_outliers = solve_independent_multicamera(
            outlier_calibration,
            method="PARK",
        )
        park_outlier_runtime = (
            perf_counter() - start
        ) * 1000.0
        _append_rows(
            rows,
            scenario="outliers",
            trial=trial,
            calibration_runtime_ms=park_outlier_runtime,
            task_rows=evaluate_multicamera_task_space(
                gaussian_validation,
                method="PARK",
                camera_extrinsics=park_outliers.camera_extrinsics,
            ),
        )

        huber_outliers = solve_joint_multicamera(
            outlier_calibration,
            initialization_method="PARK",
            loss="huber",
            translation_sigma_m=0.0005,
            rotation_sigma_deg=0.25,
        )
        _append_rows(
            rows,
            scenario="outliers",
            trial=trial,
            calibration_runtime_ms=huber_outliers.runtime_ms,
            task_rows=evaluate_multicamera_task_space(
                gaussian_validation,
                method="JOINT_HUBER",
                camera_extrinsics=huber_outliers.camera_extrinsics,
            ),
        )

        # Scenario 3: calibration and validation on distinct time segments.
        true_offsets = _time_offset_pattern()
        time_calibration = generate_time_offset_dataset(
            num_poses=24,
            duration_s=5.0,
            start_time_s=0.5,
            time_offsets_s=true_offsets,
            observation_translation_sigma_mm=0.25,
            observation_rotation_sigma_deg=0.10,
            seed=150_000 + trial,
        )
        time_validation = generate_time_offset_dataset(
            num_poses=validation_poses,
            duration_s=4.0,
            start_time_s=6.0,
            time_offsets_s=true_offsets,
            observation_translation_sigma_mm=0.25,
            observation_rotation_sigma_deg=0.10,
            seed=160_000 + trial,
        )

        zero_offset_data = (
            time_calibration.as_zero_offset_multicamera_dataset()
        )
        start = perf_counter()
        zero_offset_park = solve_independent_multicamera(
            zero_offset_data,
            method="PARK",
        )
        zero_runtime = (perf_counter() - start) * 1000.0
        _append_rows(
            rows,
            scenario="time_offset",
            trial=trial,
            calibration_runtime_ms=zero_runtime,
            task_rows=evaluate_time_sync_task_space(
                time_validation,
                method="PARK_ZERO_OFFSET",
                camera_extrinsics=zero_offset_park.camera_extrinsics,
                time_offsets_s={
                    name: 0.0
                    for name in time_validation.camera_names
                },
            ),
        )

        time_aware = solve_time_aware_multicamera(
            time_calibration,
            translation_sigma_m=0.00025,
            rotation_sigma_deg=0.10,
            iterations=time_iterations,
        )
        _append_rows(
            rows,
            scenario="time_offset",
            trial=trial,
            calibration_runtime_ms=time_aware.runtime_ms,
            task_rows=evaluate_time_sync_task_space(
                time_validation,
                method="TIME_AWARE_PARK",
                camera_extrinsics=time_aware.camera_extrinsics,
                time_offsets_s=time_aware.time_offsets_s,
            ),
        )

    return pd.DataFrame(rows)


def summarize_task_space_benchmark(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    return (
        trials.groupby(
            ["scenario", "method", "source"],
            as_index=False,
        )
        .agg(
            trials=("trial", "nunique"),
            grasp_point_error_mm_mean=(
                "grasp_point_error_mm",
                "mean",
            ),
            grasp_point_error_mm_p95=(
                "grasp_point_error_mm",
                lambda values: values.quantile(0.95),
            ),
            surface_point_rmse_mm_mean=(
                "surface_point_rmse_mm",
                "mean",
            ),
            approach_normal_error_deg_mean=(
                "approach_normal_error_deg",
                "mean",
            ),
            precision_action_success_rate=(
                "precision_action_success",
                "mean",
            ),
            standard_action_success_rate=(
                "standard_action_success",
                "mean",
            ),
            calibration_runtime_ms_mean=(
                "calibration_runtime_ms",
                "mean",
            ),
        )
        .sort_values(["scenario", "source", "method"])
        .reset_index(drop=True)
    )


def plot_task_space_summary(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fused = summary[summary["source"] == "FUSED_MEDIAN"]

    for scenario in ("gaussian", "outliers", "time_offset"):
        frame = fused[fused["scenario"] == scenario].sort_values(
            "method"
        )

        plt.figure(figsize=(7.8, 5.0))
        plt.bar(
            frame["method"],
            frame["grasp_point_error_mm_mean"],
        )
        plt.ylabel("Mean held-out grasp-point error (mm)")
        plt.xlabel("Calibration method")
        plt.title(
            f"Task-Space Grasp Error — {scenario.replace('_', ' ').title()}"
        )
        plt.xticks(rotation=12)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"phase9_grasp_error_{scenario}.png",
            dpi=180,
        )
        plt.close()

        plt.figure(figsize=(7.8, 5.0))
        plt.bar(
            frame["method"],
            frame["precision_action_success_rate"],
        )
        plt.ylim(0.0, 1.05)
        plt.ylabel("Precision action acceptance rate")
        plt.xlabel("Calibration method")
        plt.title(
            f"Precision Task Acceptance — "
            f"{scenario.replace('_', ' ').title()}"
        )
        plt.xticks(rotation=12)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"phase9_precision_success_{scenario}.png",
            dpi=180,
        )
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate calibration through held-out robot task-space error."
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--calibration-poses", type=int, default=30)
    parser.add_argument("--validation-poses", type=int, default=36)
    parser.add_argument("--time-iterations", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_task_space_benchmark(
        trials=args.trials,
        calibration_poses=args.calibration_poses,
        validation_poses=args.validation_poses,
        time_iterations=args.time_iterations,
    )
    summary = summarize_task_space_benchmark(trials)

    trials_path = output_dir / "phase9_task_space_trials.csv"
    summary_path = output_dir / "phase9_task_space_summary.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_task_space_summary(summary, output_dir)

    fused = summary[summary["source"] == "FUSED_MEDIAN"]

    print("Phase 9: held-out task-space validation")
    print("---------------------------------------")
    print(f"Trials per scenario: {args.trials}")
    print(f"Calibration poses: {args.calibration_poses}")
    print(f"Validation poses: {args.validation_poses}")
    print()
    print(
        fused[
            [
                "scenario",
                "method",
                "grasp_point_error_mm_mean",
                "grasp_point_error_mm_p95",
                "surface_point_rmse_mm_mean",
                "approach_normal_error_deg_mean",
                "precision_action_success_rate",
                "standard_action_success_rate",
                "calibration_runtime_ms_mean",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Saved: {trials_path}")
    print(f"Saved: {summary_path}")
    for scenario in ("gaussian", "outliers", "time_offset"):
        print(
            "Saved: "
            f"{output_dir / f'phase9_grasp_error_{scenario}.png'}"
        )
        print(
            "Saved: "
            f"{output_dir / f'phase9_precision_success_{scenario}.png'}"
        )

    outlier_park = fused[
        (fused["scenario"] == "outliers")
        & (fused["method"] == "PARK")
    ].iloc[0]
    outlier_huber = fused[
        (fused["scenario"] == "outliers")
        & (fused["method"] == "JOINT_HUBER")
    ].iloc[0]
    time_zero = fused[
        (fused["scenario"] == "time_offset")
        & (fused["method"] == "PARK_ZERO_OFFSET")
    ].iloc[0]
    time_aware = fused[
        (fused["scenario"] == "time_offset")
        & (fused["method"] == "TIME_AWARE_PARK")
    ].iloc[0]

    if not (
        float(outlier_huber["grasp_point_error_mm_mean"])
        < 0.5 * float(outlier_park["grasp_point_error_mm_mean"])
    ):
        raise RuntimeError("outlier task-space improvement gate failed")
    if not (
        float(time_aware["grasp_point_error_mm_mean"])
        < 0.35 * float(time_zero["grasp_point_error_mm_mean"])
    ):
        raise RuntimeError("time-aware task-space improvement gate failed")

    print("Phase 9 benchmark gate: PASS")


if __name__ == "__main__":
    main()
