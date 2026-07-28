from calibgraph.evaluation.observability import (
    analyze_motion_observability,
)
from calibgraph.simulation.motion_regimes import (
    generate_motion_regime_dataset,
)


def test_diverse_motion_is_well_observed():
    dataset = generate_motion_regime_dataset(
        motion_regime="diverse",
        num_poses=25,
        seed=7,
    )
    report = analyze_motion_observability(dataset)

    assert report.quality == "GOOD"
    assert report.rotation_design_rank == 8
    assert report.rotation_axis_diversity_ratio > 0.20
    assert report.max_relative_rotation_deg > 20.0


def test_single_axis_motion_is_rejected():
    dataset = generate_motion_regime_dataset(
        motion_regime="single_axis",
        num_poses=25,
        seed=7,
    )
    report = analyze_motion_observability(dataset)

    assert report.quality == "POOR"
    assert report.rotation_design_rank < 8
    assert report.rotation_axis_diversity_ratio < 0.02


def test_translation_only_motion_is_rejected():
    dataset = generate_motion_regime_dataset(
        motion_regime="translation_only",
        num_poses=25,
        seed=7,
    )
    report = analyze_motion_observability(dataset)

    assert report.quality == "POOR"
    assert report.rotation_design_rank == 0
    assert report.max_relative_rotation_deg < 1e-8
