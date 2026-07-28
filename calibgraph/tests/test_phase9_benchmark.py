from calibgraph.experiments.task_space_validation import (
    run_task_space_benchmark,
    summarize_task_space_benchmark,
)


def test_phase9_benchmark_smoke():
    trials = run_task_space_benchmark(
        trials=1,
        calibration_poses=18,
        validation_poses=12,
        time_iterations=2,
    )
    summary = summarize_task_space_benchmark(trials)

    assert not trials.empty
    assert not summary.empty

    fused = summary[summary["source"] == "FUSED_MEDIAN"]
    assert set(fused["scenario"]) == {
        "gaussian",
        "outliers",
        "time_offset",
    }
    assert fused["standard_action_success_rate"].between(
        0.0, 1.0
    ).all()
