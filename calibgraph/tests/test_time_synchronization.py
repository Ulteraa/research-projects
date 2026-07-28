from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.evaluation.time_sync_metrics import (
    evaluate_time_sync_estimate,
)
from calibgraph.graph.time_offset import solve_time_aware_multicamera
from calibgraph.simulation.time_sync import (
    generate_time_offset_dataset,
)


def test_time_aware_solver_recovers_zero_offsets():
    dataset = generate_time_offset_dataset(
        num_poses=12,
        duration_s=3.0,
        time_offsets_s={
            "upper_arm_camera": 0.0,
            "forearm_camera": 0.0,
            "wrist_camera": 0.0,
        },
        seed=4,
    )
    result = solve_time_aware_multicamera(
        dataset,
        iterations=1,
    )

    assert result.success
    assert max(
        abs(value) for value in result.time_offsets_s.values()
    ) < 0.005


def test_time_aware_solver_improves_misaligned_calibration():
    true_offsets = {
        "upper_arm_camera": 0.015,
        "forearm_camera": 0.050,
        "wrist_camera": -0.040,
    }
    dataset = generate_time_offset_dataset(
        num_poses=14,
        duration_s=3.5,
        time_offsets_s=true_offsets,
        seed=5,
    )
    baseline = solve_independent_multicamera(
        dataset.as_zero_offset_multicamera_dataset(),
        method="PARK",
    )
    baseline_rows = evaluate_time_sync_estimate(
        dataset,
        method="PARK_ZERO_OFFSET",
        camera_extrinsics=baseline.camera_extrinsics,
        time_offsets_s={
            name: 0.0 for name in dataset.camera_names
        },
    )

    result = solve_time_aware_multicamera(
        dataset,
        iterations=2,
    )
    refined_rows = evaluate_time_sync_estimate(
        dataset,
        method=result.method,
        camera_extrinsics=result.camera_extrinsics,
        time_offsets_s=result.time_offsets_s,
    )

    baseline_translation = sum(
        row["translation_error_mm"] for row in baseline_rows
    ) / len(baseline_rows)
    refined_translation = sum(
        row["translation_error_mm"] for row in refined_rows
    ) / len(refined_rows)
    offset_mae = sum(
        row["time_offset_abs_error_ms"] for row in refined_rows
    ) / len(refined_rows)

    assert result.success
    assert refined_translation < 0.35 * baseline_translation
    assert offset_mae < 10.0
