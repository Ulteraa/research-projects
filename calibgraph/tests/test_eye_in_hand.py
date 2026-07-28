import pytest

from calibgraph.baselines.opencv_hand_eye import (
    HAND_EYE_METHODS,
    solve_all_opencv_methods,
)
from calibgraph.evaluation.metrics import evaluate_hand_eye_result
from calibgraph.geometry.se3 import compose, rotation_error_deg, translation_error
from calibgraph.simulation.eye_in_hand import generate_eye_in_hand_dataset


def test_synthetic_transform_equation_is_exact():
    dataset = generate_eye_in_hand_dataset(num_poses=12, seed=11)

    for T_B_G, T_C_T in zip(dataset.T_B_G, dataset.T_C_T, strict=True):
        reconstructed = compose(
            T_B_G,
            dataset.T_G_C_ground_truth,
            T_C_T,
        )
        assert translation_error(
            reconstructed,
            dataset.T_B_T_ground_truth,
        ) < 1e-10
        assert rotation_error_deg(
            reconstructed,
            dataset.T_B_T_ground_truth,
        ) < 1e-8


@pytest.mark.parametrize("method", tuple(HAND_EYE_METHODS))
def test_opencv_solver_recovers_ground_truth(method):
    dataset = generate_eye_in_hand_dataset(num_poses=25, seed=7)
    result = next(
        item
        for item in solve_all_opencv_methods(dataset)
        if item.method == method
    )
    metrics = evaluate_hand_eye_result(dataset, result)

    assert metrics["translation_error_mm"] < 1e-3
    assert metrics["rotation_error_deg"] < 1e-5
