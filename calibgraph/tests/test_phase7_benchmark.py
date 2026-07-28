from calibgraph.experiments.drift_monitoring import (
    run_drift_benchmark,
    summarize_drift_benchmark,
)


def test_phase7_benchmark_smoke():
    trials, traces = run_drift_benchmark(
        trials=2,
        num_steps=60,
        calibration_window=15,
        drift_start_index=35,
    )
    summary = summarize_drift_benchmark(trials)

    assert len(trials) == 8
    assert not traces.empty

    no_drift = summary[summary["scenario"] == "none"].iloc[0]
    large = summary[summary["scenario"] == "large"].iloc[0]

    assert no_drift["false_alarm_rate"] <= 0.5
    assert large["detection_rate"] >= 0.5
