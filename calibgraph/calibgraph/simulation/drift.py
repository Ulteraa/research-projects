"""Synthetic mechanical camera-mount drift for articulated robots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, inverse, make_transform
from calibgraph.simulation.articulated_multicamera import (
    CAMERA_NAMES,
    MultiCameraDataset,
    generate_articulated_multicamera_dataset,
)
from calibgraph.simulation.noise import perturb_transform_left

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DriftSequence:
    dataset: MultiCameraDataset
    calibration_window: int
    drift_start_index: int | None
    drift_camera: str | None
    drift_translation_mm: float
    drift_rotation_deg: float


def subset_multicamera_dataset(
    dataset: MultiCameraDataset,
    *,
    start: int,
    stop: int,
) -> MultiCameraDataset:
    """Slice a synchronized multi-camera sequence."""
    if not 0 <= start < stop <= dataset.num_poses:
        raise ValueError(
            f"invalid slice [{start}:{stop}] for {dataset.num_poses} poses"
        )

    return MultiCameraDataset(
        joint_positions_rad=dataset.joint_positions_rad[start:stop].copy(),
        link_poses_by_name={
            name: tuple(poses[start:stop])
            for name, poses in dataset.link_poses_by_name.items()
        },
        target_observations_by_camera={
            name: tuple(poses[start:stop])
            for name, poses in dataset.target_observations_by_camera.items()
        },
        camera_extrinsics_ground_truth={
            name: transform.copy()
            for name, transform in (
                dataset.camera_extrinsics_ground_truth.items()
            )
        },
        camera_links=dict(dataset.camera_links),
        T_B_T_ground_truth=dataset.T_B_T_ground_truth.copy(),
    )


def _sample_fixed_magnitude_drift(
    rng: np.random.Generator,
    *,
    translation_mm: float,
    rotation_deg: float,
) -> FloatArray:
    if translation_mm < 0 or rotation_deg < 0:
        raise ValueError("drift magnitudes must be non-negative")

    translation_direction = rng.normal(size=3)
    translation_norm = np.linalg.norm(translation_direction)
    if translation_norm < 1e-12:
        translation_direction = np.array([1.0, 0.0, 0.0])
    else:
        translation_direction /= translation_norm

    rotation_axis = rng.normal(size=3)
    rotation_norm = np.linalg.norm(rotation_axis)
    if rotation_norm < 1e-12:
        rotation_axis = np.array([0.0, 0.0, 1.0])
    else:
        rotation_axis /= rotation_norm

    translation = translation_direction * (translation_mm / 1000.0)
    rotation = Rotation.from_rotvec(
        rotation_axis * np.deg2rad(rotation_deg)
    ).as_matrix()
    return make_transform(rotation, translation)


def generate_multicamera_drift_sequence(
    *,
    num_steps: int = 80,
    calibration_window: int = 20,
    drift_start_index: int | None = 45,
    drift_camera: str | None = "wrist_camera",
    drift_translation_mm: float = 3.0,
    drift_rotation_deg: float = 0.5,
    observation_translation_sigma_mm: float = 0.25,
    observation_rotation_sigma_deg: float = 0.10,
    seed: int = 7,
) -> DriftSequence:
    """Generate a synchronized validation sequence with optional mount drift.

    The monitor is expected to calibrate on the first ``calibration_window``
    frames. Mechanical drift is applied in the carrying-link frame from
    ``drift_start_index`` onward.
    """
    if num_steps < 10:
        raise ValueError("num_steps must be at least 10")
    if not 3 <= calibration_window < num_steps:
        raise ValueError("calibration_window must be in [3, num_steps)")
    if drift_camera is not None and drift_camera not in CAMERA_NAMES:
        raise ValueError(f"unknown drift camera {drift_camera!r}")
    if drift_start_index is not None:
        if not calibration_window < drift_start_index < num_steps:
            raise ValueError(
                "drift_start_index must be after calibration and before end"
            )
    if drift_camera is None and drift_start_index is not None:
        raise ValueError("drift_start_index requires drift_camera")
    if drift_camera is not None and drift_start_index is None:
        raise ValueError("drift_camera requires drift_start_index")

    rng = np.random.default_rng(seed)
    clean = generate_articulated_multicamera_dataset(
        num_poses=num_steps,
        seed=seed,
    )

    drift_transform = _sample_fixed_magnitude_drift(
        rng,
        translation_mm=drift_translation_mm,
        rotation_deg=drift_rotation_deg,
    )

    observations: dict[str, tuple[FloatArray, ...]] = {}
    for camera_name in clean.camera_names:
        link_name = clean.camera_links[camera_name]
        nominal_mount = clean.camera_extrinsics_ground_truth[camera_name]
        camera_observations: list[FloatArray] = []

        for time_index, T_B_L in enumerate(
            clean.link_poses_by_name[link_name]
        ):
            actual_mount = nominal_mount
            if (
                drift_camera == camera_name
                and drift_start_index is not None
                and time_index >= drift_start_index
            ):
                # Drift is expressed in the carrying-link frame.
                actual_mount = compose(drift_transform, nominal_mount)

            exact_observation = compose(
                inverse(compose(T_B_L, actual_mount)),
                clean.T_B_T_ground_truth,
            )
            noisy_observation = perturb_transform_left(
                exact_observation,
                rng,
                translation_sigma_m=(
                    observation_translation_sigma_mm / 1000.0
                ),
                rotation_sigma_deg=observation_rotation_sigma_deg,
            )
            camera_observations.append(noisy_observation)

        observations[camera_name] = tuple(camera_observations)

    observed = MultiCameraDataset(
        joint_positions_rad=clean.joint_positions_rad.copy(),
        link_poses_by_name={
            name: tuple(poses)
            for name, poses in clean.link_poses_by_name.items()
        },
        target_observations_by_camera=observations,
        camera_extrinsics_ground_truth={
            name: transform.copy()
            for name, transform in (
                clean.camera_extrinsics_ground_truth.items()
            )
        },
        camera_links=dict(clean.camera_links),
        T_B_T_ground_truth=clean.T_B_T_ground_truth.copy(),
    )

    return DriftSequence(
        dataset=observed,
        calibration_window=calibration_window,
        drift_start_index=drift_start_index,
        drift_camera=drift_camera,
        drift_translation_mm=drift_translation_mm,
        drift_rotation_deg=drift_rotation_deg,
    )
