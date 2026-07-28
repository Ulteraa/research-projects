from calibgraph.experiments.multicamera_independent import (
    run_multicamera_benchmark,
    summarize_multicamera_trials,
)


def test_phase5_benchmark_smoke():
    trials = run_multicamera_benchmark(
        trials=2,
        num_poses=20,
        method="PARK",
        translation_sigma_mm=0.50,
        rotation_sigma_deg=0.25,
    )
    summary = summarize_multicamera_trials(trials)

    assert len(trials) == 6
    assert len(summary) == 3
    assert (summary["quality"] == "GOOD").all()
    assert summary["translation_error_mm_mean"].max() < 2.0
    assert summary["rotation_error_deg_mean"].max() < 0.5
