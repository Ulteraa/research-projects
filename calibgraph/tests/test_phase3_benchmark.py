from calibgraph.experiments.target_pose_noise import (
    NOISE_REGIMES,
    run_benchmark,
    summarize_trials,
)


def test_small_noise_benchmark_smoke():
    selected_regimes = (NOISE_REGIMES[0], NOISE_REGIMES[-1])
    trials = run_benchmark(
        trials=1,
        num_poses=12,
        noise_regimes=selected_regimes,
    )
    summary = summarize_trials(trials)

    # 2 noise regimes x 1 trial x 5 methods.
    assert len(trials) == 10
    assert len(summary) == 10
    assert trials["success"].all()

    ideal = summary[summary["noise_level"] == 0]
    assert ideal["translation_error_mm_mean"].max() < 1e-3
    assert ideal["rotation_error_deg_mean"].max() < 1e-5
