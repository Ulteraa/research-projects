from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.evaluation.multicamera_metrics import (
    evaluate_independent_multicamera,
)
from calibgraph.geometry.se3 import compose, rotation_error_deg, translation_error
from calibgraph.simulation.articulated_multicamera import (
    generate_articulated_multicamera_dataset,
)


def test_multicamera_transform_equations_are_exact():
    dataset = generate_articulated_multicamera_dataset(
        num_poses=15,
        seed=9,
    )

    for camera_name in dataset.camera_names:
        link_name = dataset.camera_links[camera_name]
        T_L_C = dataset.camera_extrinsics_ground_truth[camera_name]
        for T_B_L, T_C_T in zip(
            dataset.link_poses_by_name[link_name],
            dataset.target_observations_by_camera[camera_name],
            strict=True,
        ):
            reconstructed = compose(T_B_L, T_L_C, T_C_T)
            assert translation_error(
                reconstructed,
                dataset.T_B_T_ground_truth,
            ) < 1e-10
            assert rotation_error_deg(
                reconstructed,
                dataset.T_B_T_ground_truth,
            ) < 1e-8


def test_independent_multicamera_recovers_exact_extrinsics():
    dataset = generate_articulated_multicamera_dataset(
        num_poses=30,
        seed=17,
    )
    result = solve_independent_multicamera(
        dataset,
        method="PARK",
    )
    metrics = evaluate_independent_multicamera(dataset, result)

    assert len(metrics) == 3
    assert max(row["translation_error_mm"] for row in metrics) < 1e-3
    assert max(row["rotation_error_deg"] for row in metrics) < 1e-5
    assert (
        max(
            row["mean_cross_camera_target_disagreement_mm"]
            for row in metrics
        )
        < 1e-3
    )
