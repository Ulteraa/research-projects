"""Compare independent AX=XB against joint multi-camera graph refinement."""

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
    evaluate_multicamera_estimate,
)
from calibgraph.graph.joint_multicamera import solve_joint_multicamera
from calibgraph.simulation.articulated_multicamera import (
    generate_articulated_multicamera_dataset,
)
from calibgraph.simulation.multicamera_noise import (
    add_multicamera_mixed_noise,
)


SCENARIOS = {
    "gaussian": {
        "outlier_probability": 0.0,
        "outlier_translation_sigma_m": 0.0,
        "outlier_rotation_sigma_deg": 0.0,
    },
    "outliers": {
        "outlier_probability": 0.08,
        "outlier_translation_sigma_m": 0.020,
        "outlier_rotation_sigma_deg": 5.0,
    },
}


def run_joint_benchmark(
    *,
    trials: int = 10,
    num_poses: int = 30,
    translation_sigma_mm: float = 0.50,
    rotation_sigma_deg: float = 0.25,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for scenario_name, scenario in SCENARIOS.items():
        for trial in range(trials):
            clean = generate_articulated_multicamera_dataset(
                num_poses=num_poses,
                seed=60_000 + trial,
            )
            observed = add_multicamera_mixed_noise(
                clean,
                seed=70_000 + trial + 10_000 * (scenario_name == "outliers"),
                translation_sigma_m=translation_sigma_mm / 1000.0,
                rotation_sigma_deg=rotation_sigma_deg,
                **scenario,
            )

            independent_start = perf_counter()
            independent = solve_independent_multicamera(
                observed,
                method="PARK",
            )
            independent_runtime_ms = (
                perf_counter() - independent_start
            ) * 1000.0

            joint_linear = solve_joint_multicamera(
                observed,
                initialization_method="PARK",
                loss="linear",
                translation_sigma_m=translation_sigma_mm / 1000.0,
                rotation_sigma_deg=rotation_sigma_deg,
            )
            joint_huber = solve_joint_multicamera(
                observed,
                initialization_method="PARK",
                loss="huber",
                translation_sigma_m=translation_sigma_mm / 1000.0,
                rotation_sigma_deg=rotation_sigma_deg,
            )

            methods = (
                (independent, independent_runtime_ms, True, 0, 0.0),
                (
                    joint_linear,
                    joint_linear.runtime_ms,
                    joint_linear.success,
                    joint_linear.nfev,
                    joint_linear.cost,
                ),
                (
                    joint_huber,
                    joint_huber.runtime_ms,
                    joint_huber.success,
                    joint_huber.nfev,
                    joint_huber.cost,
                ),
            )

            for result, runtime_ms, success, nfev, cost in methods:
                for metric in evaluate_multicamera_estimate(clean, result):
                    rows.append(
                        {
                            "scenario": scenario_name,
                            "trial": trial,
                            "num_poses": num_poses,
                            "base_translation_sigma_mm": translation_sigma_mm,
                            "base_rotation_sigma_deg": rotation_sigma_deg,
                            "success": success,
                            "runtime_ms": runtime_ms,
                            "nfev": nfev,
                            "cost": cost,
                            **metric,
                        }
                    )

    return pd.DataFrame(rows)


def summarize_joint_benchmark(trials: pd.DataFrame) -> pd.DataFrame:
    grouped = trials.groupby(
        ["scenario", "method", "camera_name", "link_name"],
        as_index=False,
    )
    summary = grouped.agg(
        trials=("trial", "count"),
        success_rate=("success", "mean"),
        translation_error_mm_mean=("translation_error_mm", "mean"),
        translation_error_mm_std=("translation_error_mm", "std"),
        rotation_error_deg_mean=("rotation_error_deg", "mean"),
        rotation_error_deg_std=("rotation_error_deg", "std"),
        mean_target_error_mm_mean=("mean_target_error_mm", "mean"),
        mean_cross_camera_target_disagreement_mm=(
            "mean_cross_camera_target_disagreement_mm",
            "mean",
        ),
        runtime_ms_mean=("runtime_ms", "mean"),
        nfev_mean=("nfev", "mean"),
    )
    return summary.sort_values(
        ["scenario", "camera_name", "method"]
    ).reset_index(drop=True)


def plot_joint_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    for scenario in SCENARIOS:
        frame = summary[summary["scenario"] == scenario]
        aggregate = (
            frame.groupby("method", as_index=False)
            .agg(
                translation_error_mm=(
                    "translation_error_mm_mean",
                    "mean",
                ),
                cross_camera_disagreement_mm=(
                    "mean_cross_camera_target_disagreement_mm",
                    "mean",
                ),
            )
            .sort_values("method")
        )

        plt.figure(figsize=(7.8, 5.0))
        plt.bar(
            aggregate["method"],
            aggregate["translation_error_mm"],
        )
        plt.ylabel("Mean camera-to-link translation error (mm)")
        plt.xlabel("Calibration method")
        plt.title(
            f"Independent vs Joint Calibration — {scenario.capitalize()}"
        )
        plt.xticks(rotation=12)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"phase6_translation_{scenario}.png",
            dpi=180,
        )
        plt.close()

        plt.figure(figsize=(7.8, 5.0))
        plt.bar(
            aggregate["method"],
            aggregate["cross_camera_disagreement_mm"],
        )
        plt.ylabel("Mean cross-camera target disagreement (mm)")
        plt.xlabel("Calibration method")
        plt.title(
            f"Cross-Camera Consistency — {scenario.capitalize()}"
        )
        plt.xticks(rotation=12)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"phase6_consistency_{scenario}.png",
            dpi=180,
        )
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark joint articulated multi-camera calibration."
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--num-poses", type=int, default=30)
    parser.add_argument("--translation-sigma-mm", type=float, default=0.50)
    parser.add_argument("--rotation-sigma-deg", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = run_joint_benchmark(
        trials=args.trials,
        num_poses=args.num_poses,
        translation_sigma_mm=args.translation_sigma_mm,
        rotation_sigma_deg=args.rotation_sigma_deg,
    )
    summary = summarize_joint_benchmark(trials)

    trials_path = output_dir / "phase6_joint_trials.csv"
    summary_path = output_dir / "phase6_joint_summary.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_joint_summary(summary, output_dir)

    display = (
        summary.groupby(["scenario", "method"], as_index=False)
        .agg(
            translation_error_mm_mean=(
                "translation_error_mm_mean",
                "mean",
            ),
            rotation_error_deg_mean=(
                "rotation_error_deg_mean",
                "mean",
            ),
            target_error_mm_mean=(
                "mean_target_error_mm_mean",
                "mean",
            ),
            cross_camera_disagreement_mm=(
                "mean_cross_camera_target_disagreement_mm",
                "mean",
            ),
            runtime_ms_mean=("runtime_ms_mean", "mean"),
            success_rate=("success_rate", "mean"),
        )
    )

    print("Phase 6: joint multi-camera graph refinement")
    print("--------------------------------------------")
    print(f"Trials per scenario: {args.trials}")
    print(f"Robot poses per trial: {args.num_poses}")
    print()
    print(
        display.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print()
    print(f"Saved: {trials_path}")
    print(f"Saved: {summary_path}")
    for scenario in SCENARIOS:
        print(
            "Saved: "
            f"{output_dir / f'phase6_translation_{scenario}.png'}"
        )
        print(
            "Saved: "
            f"{output_dir / f'phase6_consistency_{scenario}.png'}"
        )

    if float(display["success_rate"].min()) < 1.0:
        raise RuntimeError("joint optimization did not converge in all trials")

    print("Phase 6 benchmark gate: PASS")


if __name__ == "__main__":
    main()
