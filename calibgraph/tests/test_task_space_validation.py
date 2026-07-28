from calibgraph.evaluation.task_space import (
    evaluate_multicamera_task_space,
    evaluate_time_sync_task_space,
)
from calibgraph.simulation.articulated_multicamera import (
    generate_articulated_multicamera_dataset,
)
from calibgraph.simulation.time_sync import (
    generate_time_offset_dataset,
)


def test_ground_truth_multicamera_has_zero_task_error():
    dataset = generate_articulated_multicamera_dataset(
        num_poses=12,
        seed=4,
    )
    rows = evaluate_multicamera_task_space(
        dataset,
        method="GROUND_TRUTH",
        camera_extrinsics=dataset.camera_extrinsics_ground_truth,
    )

    assert max(
        float(row["grasp_point_error_mm"])
        for row in rows
    ) < 1e-8
    assert max(
        float(row["approach_normal_error_deg"])
        for row in rows
    ) < 1e-6


def test_ground_truth_offsets_have_zero_time_task_error():
    offsets = {
        "upper_arm_camera": 0.030,
        "forearm_camera": 0.100,
        "wrist_camera": -0.080,
    }
    dataset = generate_time_offset_dataset(
        num_poses=12,
        duration_s=2.0,
        start_time_s=0.5,
        time_offsets_s=offsets,
        observation_translation_sigma_mm=0.0,
        observation_rotation_sigma_deg=0.0,
        seed=5,
    )
    rows = evaluate_time_sync_task_space(
        dataset,
        method="GROUND_TRUTH",
        camera_extrinsics=dataset.camera_extrinsics_ground_truth,
        time_offsets_s=offsets,
    )

    assert max(
        float(row["grasp_point_error_mm"])
        for row in rows
    ) < 1e-8
    assert max(
        float(row["approach_normal_error_deg"])
        for row in rows
    ) < 1e-6
