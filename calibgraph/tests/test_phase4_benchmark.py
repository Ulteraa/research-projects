from calibgraph.experiments.motion_observability import (
    run_motion_benchmark,
    summarize_motion_benchmark,
)


def test_motion_benchmark_smoke():
    trials = run_motion_benchmark(
        trials=1,
        num_poses=12,
        translation_sigma_mm=0.50,
        rotation_sigma_deg=0.25,
    )
    summary = summarize_motion_benchmark(trials)

    # 4 motion regimes × 1 trial × 5 methods.
    assert len(trials) == 20
    assert len(summary) == 20

    diverse = summary[summary["motion_regime"] == "diverse"]
    assert (diverse["failure_rate"] == 0.0).all()
    assert diverse["translation_error_mm_median"].max() < 2.0
    assert diverse["rotation_error_deg_median"].max() < 0.5
