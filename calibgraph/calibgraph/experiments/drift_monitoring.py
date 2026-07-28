"""Evaluate online calibration health monitoring under mechanical drift."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibgraph.monitoring.calibration_health import (
    fit_calibration_health_monitor,
)
from calibgraph.simulation.articulated_multicamera import CAMERA_NAMES
from calibgraph.simulation.drift import (
    generate_multicamera_drift_sequence,
)


DRIFT_SCENARIOS = {
    "none": (0.0, 0.0),
    "small": (1.0, 0.20),
    "medium": (3.0, 0.50),
    "large": (8.0, 1.50),
}


def run_drift_benchmark(
    *,
    trials: int = 20,
    num_steps: int = 80,
    calibration_window: int = 20,
    drift_start_index: int = 45,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []

    for scenario_index, (
        scenario_name,
        (translation_mm, rotation_deg),
    ) in enumerate(DRIFT_SCENARIOS.items()):
        for trial in range(trials):
            drift_camera = None
            drift_start = None
            if scenario_name != "none":
                drift_camera = CAMERA_NAMES[trial % len(CAMERA_NAMES)]
                drift_start = drift_start_index

            sequence = generate_multicamera_drift_sequence(
                num_steps=num_steps,
                calibration_window=calibration_window,
                drift_start_index=drift_start,
                drift_camera=drift_camera,
                drift_translation_mm=translation_mm,
                drift_rotation_deg=rotation_deg,
                seed=80_000 + 10_000 * scenario_index + trial,
            )
            monitor = fit_calibration_health_monitor(
                sequence.dataset,
                calibration_window=calibration_window,
                warning_threshold=4.0,
                critical_threshold=8.0,
                persistence=3,
            )
            report = monitor.evaluate(sequence.dataset)

            detections = report.first_detection_by_camera
            pre_drift_limit = (
                drift_start_index
                if scenario_name != "none"
                else num_steps
            )
            false_alarm = any(
                detection is not None
                and detection < pre_drift_limit
                for detection in detections.values()
            )

            if scenario_name == "none":
                detected = False
                detection_delay = np.nan
                localization_correct = np.nan
                earliest_camera = None
            else:
                valid_post_drift = {
                    camera: detection
                    for camera, detection in detections.items()
                    if (
                        detection is not None
                        and detection >= drift_start_index
                    )
                }
                system_detection = (
                    min(valid_post_drift.values())
                    if valid_post_drift
                    else None
                )
                detected = system_detection is not None
                detection_delay = (
                    float(system_detection - drift_start_index)
                    if detected
                    else np.nan
                )
                earliest_camera = (
                    report.suspected_camera_by_time[system_detection]
                    if detected
                    else None
                )
                localization_correct = bool(
                    detected and earliest_camera == drift_camera
                )

            summary_rows.append(
                {
                    "scenario": scenario_name,
                    "trial": trial,
                    "drift_camera": drift_camera,
                    "drift_translation_mm": translation_mm,
                    "drift_rotation_deg": rotation_deg,
                    "detected": detected,
                    "detection_delay_frames": detection_delay,
                    "false_alarm": false_alarm,
                    "localization_correct": localization_correct,
                    "earliest_detected_camera": earliest_camera,
                }
            )

            if trial == 0 and scenario_name in {"none", "medium"}:
                for record in report.records:
                    trace_rows.append(
                        {
                            "scenario": scenario_name,
                            **record,
                        }
                    )

    return pd.DataFrame(summary_rows), pd.DataFrame(trace_rows)


def summarize_drift_benchmark(
    trials: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scenario_name, frame in trials.groupby(
        "scenario",
        sort=False,
    ):
        is_no_drift = scenario_name == "none"
        rows.append(
            {
                "scenario": scenario_name,
                "trials": len(frame),
                "drift_translation_mm": float(
                    frame["drift_translation_mm"].iloc[0]
                ),
                "drift_rotation_deg": float(
                    frame["drift_rotation_deg"].iloc[0]
                ),
                "detection_rate": (
                    np.nan
                    if is_no_drift
                    else float(frame["detected"].mean())
                ),
                "median_detection_delay_frames": (
                    np.nan
                    if is_no_drift
                    else float(
                        frame["detection_delay_frames"].median()
                    )
                ),
                "localization_accuracy": (
                    np.nan
                    if is_no_drift
                    else float(
                        frame["localization_correct"].mean()
                    )
                ),
                "false_alarm_rate": float(
                    frame["false_alarm"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_drift_results(
    summary: pd.DataFrame,
    traces: pd.DataFrame,
    output_dir: Path,
    *,
    drift_start_index: int,
) -> None:
    drift_summary = summary[summary["scenario"] != "none"]

    plt.figure(figsize=(8.0, 5.0))
    plt.bar(
        drift_summary["scenario"],
        drift_summary["detection_rate"],
    )
    plt.ylim(0.0, 1.05)
    plt.ylabel("Detection rate")
    plt.xlabel("Injected mount drift")
    plt.title("Calibration Drift Detection Rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase7_detection_rate.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8.0, 5.0))
    plt.bar(
        drift_summary["scenario"],
        drift_summary["median_detection_delay_frames"],
    )
    plt.ylabel("Median detection delay (frames)")
    plt.xlabel("Injected mount drift")
    plt.title("Calibration Drift Detection Delay")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase7_detection_delay.png",
        dpi=180,
    )
    plt.close()

    medium = traces[traces["scenario"] == "medium"]
    plt.figure(figsize=(9.0, 5.3))
    for camera_name, frame in medium.groupby("camera_name"):
        ordered = frame.sort_values("time_index")
        plt.plot(
            ordered["time_index"],
            ordered["health_score"],
            label=camera_name,
        )
    plt.axvline(
        drift_start_index,
        linestyle="--",
        label="drift starts",
    )
    plt.axhline(
        8.0,
        linestyle=":",
        label="critical threshold",
    )
    plt.xlabel("Sequence frame")
    plt.ylabel("Calibration health score")
    plt.title("Online Calibration Health Score — Medium Drift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "phase7_health_score_trace.png",
        dpi=180,
    )
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multi-camera calibration drift monitoring."
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=80)
    parser.add_argument("--calibration-window", type=int, default=20)
    parser.add_argument("--drift-start-index", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trials, traces = run_drift_benchmark(
        trials=args.trials,
        num_steps=args.num_steps,
        calibration_window=args.calibration_window,
        drift_start_index=args.drift_start_index,
    )
    summary = summarize_drift_benchmark(trials)

    trials_path = output_dir / "phase7_drift_trials.csv"
    summary_path = output_dir / "phase7_drift_summary.csv"
    traces_path = output_dir / "phase7_health_traces.csv"

    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    traces.to_csv(traces_path, index=False)
    plot_drift_results(
        summary,
        traces,
        output_dir,
        drift_start_index=args.drift_start_index,
    )

    print("Phase 7: calibration health and drift monitoring")
    print("------------------------------------------------")
    print(f"Trials per scenario: {args.trials}")
    print(f"Frames per sequence: {args.num_steps}")
    print(f"Calibration window: {args.calibration_window}")
    print(f"Drift starts at frame: {args.drift_start_index}")
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
    print(f"Saved: {traces_path}")
    print(
        "Saved: "
        f"{output_dir / 'phase7_detection_rate.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase7_detection_delay.png'}"
    )
    print(
        "Saved: "
        f"{output_dir / 'phase7_health_score_trace.png'}"
    )

    no_drift = summary[summary["scenario"] == "none"].iloc[0]
    medium = summary[summary["scenario"] == "medium"].iloc[0]
    large = summary[summary["scenario"] == "large"].iloc[0]

    if float(no_drift["false_alarm_rate"]) > 0.10:
        raise RuntimeError("false-alarm gate failed")
    if float(medium["detection_rate"]) < 0.80:
        raise RuntimeError("medium-drift detection gate failed")
    if float(large["detection_rate"]) < 0.95:
        raise RuntimeError("large-drift detection gate failed")
    if float(medium["localization_accuracy"]) < 0.85:
        raise RuntimeError("medium-drift localization gate failed")
    if float(large["localization_accuracy"]) < 0.85:
        raise RuntimeError("large-drift localization gate failed")

    print("Phase 7 benchmark gate: PASS")


if __name__ == "__main__":
    main()
