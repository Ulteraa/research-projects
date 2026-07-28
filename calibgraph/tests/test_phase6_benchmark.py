from calibgraph.experiments.joint_multicamera import (
    run_joint_benchmark,
    summarize_joint_benchmark,
)


def test_phase6_benchmark_smoke():
    trials = run_joint_benchmark(
        trials=1,
        num_poses=15,
        translation_sigma_mm=0.50,
        rotation_sigma_deg=0.25,
    )
    summary = summarize_joint_benchmark(trials)

    # 2 scenarios × 3 methods × 3 cameras.
    assert len(trials) == 18
    assert len(summary) == 18
    assert trials["success"].all()
