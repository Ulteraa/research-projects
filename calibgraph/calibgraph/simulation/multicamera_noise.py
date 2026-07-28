"""Noise and outlier injection for multi-camera target-pose observations."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from calibgraph.simulation.articulated_multicamera import MultiCameraDataset
from calibgraph.simulation.noise import perturb_transform_left


def add_multicamera_target_pose_noise(
    dataset: MultiCameraDataset,
    *,
    seed: int,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
) -> MultiCameraDataset:
    """Perturb every camera observation independently."""
    return add_multicamera_mixed_noise(
        dataset,
        seed=seed,
        translation_sigma_m=translation_sigma_m,
        rotation_sigma_deg=rotation_sigma_deg,
        outlier_probability=0.0,
        outlier_translation_sigma_m=0.0,
        outlier_rotation_sigma_deg=0.0,
    )


def add_multicamera_mixed_noise(
    dataset: MultiCameraDataset,
    *,
    seed: int,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
    outlier_probability: float,
    outlier_translation_sigma_m: float,
    outlier_rotation_sigma_deg: float,
) -> MultiCameraDataset:
    """Add base Gaussian noise plus optional gross outlier perturbations."""
    if not 0.0 <= outlier_probability <= 1.0:
        raise ValueError("outlier_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    observations = {}

    for camera_name, transforms in (
        dataset.target_observations_by_camera.items()
    ):
        camera_observations = []
        for transform in transforms:
            perturbed = perturb_transform_left(
                transform,
                rng,
                translation_sigma_m=translation_sigma_m,
                rotation_sigma_deg=rotation_sigma_deg,
            )
            if rng.random() < outlier_probability:
                perturbed = perturb_transform_left(
                    perturbed,
                    rng,
                    translation_sigma_m=outlier_translation_sigma_m,
                    rotation_sigma_deg=outlier_rotation_sigma_deg,
                )
            camera_observations.append(perturbed)
        observations[camera_name] = tuple(camera_observations)

    return replace(
        dataset,
        target_observations_by_camera=observations,
    )
