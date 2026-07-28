"""Benchmark hand-eye solvers under informative and degenerate robot motion."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibgraph.baselines.opencv_hand_eye import (
    HAND_EYE_METHODS,
    solve_opencv_hand_eye,
)
from calibgraph.evaluation.metrics import evaluate_hand_eye_result
from calibgraph.evaluation.observability import (
    analyze_motion_observability,
)
from calibgraph.simulation.motion_regimes import (
    MOTION_REGIMES,
    generate_motion_regime_dataset,
)
from calibgraph.simulation.noise import add_target_pose_noise


def run_motion_benchmark(
    *,
    trials: int = 10,
    num_poses: int = 25,
    translation_sigma_mm: float = 0.50,
    rotation_sigma_deg: float = 0.25,
) -> pd.DataFrame:
    """Run all classical solvers across motion regimes."""
    if trials < 1:
        raise ValueError("trials must be positive")

    rows: list[dict[str, object]] = []

    for motion_regime in MOTION_REGIMES:
        for trial in range(trials):
            clean_dataset = generate_motion_regime_dataset(
                motion_regime=motion_regime,
                num_poses=num_poses,
                seed=20_000 + trial,
            )
            observed_dataset = add_target_pose_noise(
                clean_dataset,
                seed=30_000 + trial,
                translation_sigma_m=translation_sigma_mm / 1000.0,
                rotation_sigma_deg=rotation_sigma_deg,
            )
            observability = analyze_motion_observability(clean_dataset)

            for method in HAND_EYE_METHODS:
                start = perf_counter()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = solve_opencv_hand_eye(
                            observed_dataset,
                            method=method,
                        )
                    metrics = evaluate_hand_eye_result(
                        clean_dataset,
                        result,
                    )
                    finite = bool(
                        np.isfinite(metrics["translation_error_mm"])
                        and np.isfinite(metrics["rotation_error_deg"])
                    )
                    success = finite
                    error_message = ""
                except Exception as exc:
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

                runtime_ms = (perf_counter() - start) * 1000.0
                catastrophic = bool(
                    success
                    and (
                        float(metrics["translation_error_mm"]) > 10.0
                        or float(metrics["rotation_error_deg"]) > 5.0
                    )
                )

                rows.append(
                    {
                        "motion_regime": motion_regime,
                        "trial": trial,
                        "num_poses": num_poses,
                        "translation_sigma_mm": translation_sigma_mm,
                        "rotation_sigma_deg_input": rotation_sigma_deg,
                        "method": method,
                        "success": success,
                        "catastrophic": catastrophic,
                        "runtime_ms": runtime_ms,
                        "error_message": error_message,
                        **observability.to_dict(),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def summarize_motion_benchmark(
    trials_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate solver performance by motion regime."""
    grouped = trials_frame.groupby(
        [
            "motion_regime",
            "quality",
            "method",
        ],
        as_index=False,
        dropna=False,
    )
    summary = grouped.agg(
        trials=("trial", "count"),
        successful_trials=("success", "sum"),
        catastrophic_trials=("catastrophic", "sum"),
        translation_error_mm_median=("translation_error_mm", "median"),
        translation_error_mm_mean=("translation_error_mm", "mean"),
        rotation_error_deg_median=("rotation_error_deg", "median"),
        rotation_error_deg_mean=("rotation_error_deg", "mean"),
        runtime_ms_mean=("runtime_ms", "mean"),
        max_relative_rotation_deg=("max_relative_rotation_deg", "mean"),
        translation_baseline_mm=("translation_baseline_mm", "mean"),
        rotation_axis_diversity_ratio=(
            "rotation_axis_diversity_ratio",
            "mean",
        ),
        rotation_design_rank=("rotation_design_rank", "min"),
    )
    summary["failure_rate"] = (
        1.0 - summary["successful_trials"] / summary["trials"]
    )
    summary["catastrophic_rate"] = (
        summary["catastrophic_trials"] / summary["trials"]
    )

    regime_order = {name: index for index, name in enumerate(MOTION_REGIMES)}
    summary["_order"] = summary["motion_regime"].map(regime_order)
    return (
        summary.sort_values(["_order", "method"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def observability_table(
    trials_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Create one compact motion-quality row per regime."""
    columns = [
        "motion_regime",
        "quality",
        "recommendation",
        "num_poses",
        "max_relative_rotation_deg",
        "mean_relative_rotation_deg",
        "translation_baseline_mm",
        "rotation_axis_diversity_ratio",
        "rotation_design_rank",
        "expected_rotation_design_rank",
        "smallest_informative_singular_value",
    ]
    table = trials_frame[columns].drop_duplicates(
        subset=["motion_regime"]
    )
    regime_order = {name: index for index, name in enumerate(MOTION_REGIMES)}
    table["_order"] = table["motion_regime"].map(regime_order)
    return (
        table.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def _plot_metric(
    summary: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_scale: bool,
) -> None:
    regimes = list(MOTION_REGIMES)
    x = np.arange(len(regimes))

    plt.figure(figsize=(9, 5.5))
    for method, frame in summary.groupby("method"):
        ordered = (
            frame.set_index("motion_regime")
            .reindex(regimes)
        )
        plt.plot(
            x,
            ordered[metric],
            marker="o",
            label=method,
        )
    plt.xticks(x, [name.replace("_", "\n") for name in regimes])
    plt.xlabel("Robot motion regime")
    plt.ylabel(ylabel)
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_motion_results(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    _plot_metric(
        summary,
        metric="translation_error_mm_median",
        ylabel="Median camera-to-gripper translation error (mm)",
        title="Hand–Eye Calibration Sensitivity to Robot Motion",
        output_path=output_dir / "phase4_translation_error_by_motion.png",
        log_scale=True,
    )
    _plot_metric(
        summary,
        metric="rotation_error_deg_median",
        ylabel="Median camera-to-gripper rotation error (degrees)",
        title="Rotation Error Under Degenerate Robot Motion",
        output_path=output_dir / "phase4_rotation_error_by_motion.png",
        log_scale=True,
    )
    _plot_metric(
        summary,
        metric="catastrophic_rate",
        ylabel="Catastrophic estimate rate",
        title="Calibration Failure Risk by Robot Motion Regime",
        output_path=output_dir / "phase4_catastrophic_rate_by_motion.png",
        log_scale=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hand-eye calibration motion observability."
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--num-poses", type=int, default=25)
    parser.add_argument(
        "--translation-sigma-mm",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=0.25,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_motion_benchmark(
        trials=args.trials,
        num_poses=args.num_poses,
        translation_sigma_mm=args.translation_sigma_mm,
        rotation_sigma_deg=args.rotation_sigma_deg,
    )
    summary = summarize_motion_benchmark(trials)
    observability = observability_table(trials)

    trials_path = output_dir / "phase4_motion_trials.csv"
    summary_path = output_dir / "phase4_motion_summary.csv"
    observability_path = output_dir / "phase4_observability_report.csv"

    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    observability.to_csv(observability_path, index=False)
    plot_motion_results(summary, output_dir)

    print("Phase 4: motion observability and degeneracy benchmark")
    print("----------------------------------------------------")
    print(
        f"Noise: {args.translation_sigma_mm:g} mm / "
        f"{args.rotation_sigma_deg:g} deg"
    )
    print(f"Trials per regime and method: {args.trials}")
    print(f"Robot poses per trial: {args.num_poses}")
    print()

    print("Motion observability:")
    print(
        observability[
            [
                "motion_regime",
                "quality",
                "max_relative_rotation_deg",
                "translation_baseline_mm",
                "rotation_axis_diversity_ratio",
                "rotation_design_rank",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()

    print("Solver summary:")
    print(
        summary[
            [
                "motion_regime",
                "quality",
                "method",
                "translation_error_mm_median",
                "rotation_error_deg_median",
                "failure_rate",
                "catastrophic_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()

    print(f"Saved: {trials_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {observability_path}")
    print(
        "Saved: "
        f"{output_dir / 'phase4_translation_error_by_motion.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase4_rotation_error_by_motion.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase4_catastrophic_rate_by_motion.png'}"
    )

    diverse = summary[summary["motion_regime"] == "diverse"]
    if not bool((diverse["failure_rate"] == 0.0).all()):
        raise RuntimeError("diverse-motion correctness gate failed")
    if float(diverse["translation_error_mm_median"].max()) > 2.0:
        raise RuntimeError("diverse-motion translation gate failed")
    if float(diverse["rotation_error_deg_median"].max()) > 0.5:
        raise RuntimeError("diverse-motion rotation gate failed")

    print("Phase 4 benchmark gate: PASS")


if __name__ == "__main__":
    main()
