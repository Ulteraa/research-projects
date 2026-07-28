from calibgraph.experiments.time_synchronization import (
    run_time_sync_benchmark,
    summarize_time_sync_benchmark,
)


def test_phase8_benchmark_smoke():
    trials = run_time_sync_benchmark(
        trials=1,
        num_poses=12,
        duration_s=3.0,
        iterations=1,
        offset_levels_ms=(100,),
    )
    summary = summarize_time_sync_benchmark(trials)

    # 1 offset level × 2 methods × 3 cameras.
    assert len(trials) == 6
    assert len(summary) == 2
    assert trials["success"].all()

    severe_baseline = summary[
        (summary["offset_level_ms"] == 100)
        & (summary["method"] == "PARK_ZERO_OFFSET")
    ].iloc[0]
    severe_refined = summary[
        (summary["offset_level_ms"] == 100)
        & (summary["method"] == "TIME_AWARE_PARK")
    ].iloc[0]

    assert (
        severe_refined["translation_error_mm_mean"]
        < severe_baseline["translation_error_mm_mean"]
    )
