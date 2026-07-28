from calibgraph.evaluation.multicamera_metrics import (
    evaluate_multicamera_estimate,
)
from calibgraph.graph.joint_multicamera import solve_joint_multicamera
from calibgraph.simulation.articulated_multicamera import (
    generate_articulated_multicamera_dataset,
)


def test_joint_solver_recovers_exact_multicamera_solution():
    dataset = generate_articulated_multicamera_dataset(
        num_poses=20,
        seed=17,
    )
    result = solve_joint_multicamera(
        dataset,
        loss="linear",
        translation_sigma_m=0.0005,
        rotation_sigma_deg=0.25,
    )
    metrics = evaluate_multicamera_estimate(dataset, result)

    assert result.success
    assert max(row["translation_error_mm"] for row in metrics) < 1e-3
    assert max(row["rotation_error_deg"] for row in metrics) < 1e-5
    assert (
        max(
            row["mean_cross_camera_target_disagreement_mm"]
            for row in metrics
        )
        < 1e-3
    )
