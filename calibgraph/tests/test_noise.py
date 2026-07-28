import numpy as np

from calibgraph.geometry.se3 import (
    rotation_error_deg,
    translation_error,
    validate_transform,
)
from calibgraph.simulation.eye_in_hand import generate_eye_in_hand_dataset
from calibgraph.simulation.noise import add_target_pose_noise


def test_zero_noise_preserves_observations():
    dataset = generate_eye_in_hand_dataset(num_poses=8, seed=3)
    noisy = add_target_pose_noise(
        dataset,
        seed=99,
        translation_sigma_m=0.0,
        rotation_sigma_deg=0.0,
    )

    for original, perturbed in zip(
        dataset.T_C_T,
        noisy.T_C_T,
        strict=True,
    ):
        np.testing.assert_allclose(original, perturbed, atol=1e-12)


def test_noise_is_deterministic_for_fixed_seed():
    dataset = generate_eye_in_hand_dataset(num_poses=8, seed=3)
    first = add_target_pose_noise(
        dataset,
        seed=123,
        translation_sigma_m=0.001,
        rotation_sigma_deg=0.25,
    )
    second = add_target_pose_noise(
        dataset,
        seed=123,
        translation_sigma_m=0.001,
        rotation_sigma_deg=0.25,
    )

    for first_pose, second_pose in zip(
        first.T_C_T,
        second.T_C_T,
        strict=True,
    ):
        np.testing.assert_allclose(first_pose, second_pose, atol=1e-12)


def test_nonzero_noise_changes_valid_transforms():
    dataset = generate_eye_in_hand_dataset(num_poses=8, seed=3)
    noisy = add_target_pose_noise(
        dataset,
        seed=456,
        translation_sigma_m=0.001,
        rotation_sigma_deg=0.5,
    )

    changed = []
    for original, perturbed in zip(
        dataset.T_C_T,
        noisy.T_C_T,
        strict=True,
    ):
        assert validate_transform(perturbed)
        changed.append(
            translation_error(original, perturbed) > 0.0
            or rotation_error_deg(original, perturbed) > 0.0
        )

    assert all(changed)
