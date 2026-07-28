"""Synthetic multi-camera time synchronization for moving articulated robots.

Each camera observation is timestamped at t, but the image was physically
captured when the robot was at t + delta_i. A zero-offset calibration therefore
pairs the image with the wrong robot pose.

The smooth joint trajectory is analytic in this synthetic phase. In a real
system the same optimizer would interpolate timestamped encoder measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, inverse, make_transform
from calibgraph.simulation.articulated_multicamera import (
    CAMERA_LINKS,
    CAMERA_NAMES,
    MultiCameraDataset,
    forward_kinematics,
)
from calibgraph.simulation.noise import perturb_transform_left

FloatArray = NDArray[np.float64]


def smooth_joint_trajectory(time_s: float | FloatArray) -> FloatArray:
    """Evaluate a persistently excited 4-DoF joint trajectory."""
    t = np.asarray(time_s, dtype=np.float64)
    q0 = 1.35 * np.sin(0.85 * t + 0.20) + 0.25 * np.sin(2.10 * t)
    q1 = 0.95 * np.sin(1.15 * t - 0.40) + 0.18 * np.sin(2.60 * t + 0.3)
    q2 = 1.20 * np.sin(1.45 * t + 0.70) + 0.20 * np.cos(2.20 * t)
    q3 = 1.55 * np.sin(1.95 * t - 0.15) + 0.30 * np.sin(3.10 * t)
    return np.stack([q0, q1, q2, q3], axis=-1)


def _camera_mounts() -> dict[str, FloatArray]:
    return {
        "upper_arm_camera": make_transform(
            Rotation.from_euler(
                "xyz", [8.0, -20.0, 12.0], degrees=True
            ).as_matrix(),
            [0.10, 0.07, 0.05],
        ),
        "forearm_camera": make_transform(
            Rotation.from_euler(
                "xyz", [-12.0, 15.0, -18.0], degrees=True
            ).as_matrix(),
            [0.14, -0.05, 0.035],
        ),
        "wrist_camera": make_transform(
            Rotation.from_euler(
                "xyz", [18.0, -10.0, 28.0], degrees=True
            ).as_matrix(),
            [0.055, 0.025, 0.085],
        ),
    }


def _target_pose() -> FloatArray:
    return make_transform(
        Rotation.from_euler(
            "xyz", [4.0, -8.0, 14.0], degrees=True
        ).as_matrix(),
        [0.82, 0.06, 0.48],
    )


@dataclass(frozen=True)
class TimeOffsetDataset:
    camera_times_s: FloatArray
    target_observations_by_camera: dict[str, tuple[FloatArray, ...]]
    time_offsets_ground_truth_s: dict[str, float]
    camera_extrinsics_ground_truth: dict[str, FloatArray]
    camera_links: dict[str, str]
    T_B_T_ground_truth: FloatArray

    @property
    def num_poses(self) -> int:
        return int(self.camera_times_s.size)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.camera_links)

    def link_pose_at(
        self,
        camera_name: str,
        physical_time_s: float,
    ) -> FloatArray:
        link_name = self.camera_links[camera_name]
        joints = smooth_joint_trajectory(float(physical_time_s))
        return forward_kinematics(joints)[link_name]

    def as_zero_offset_multicamera_dataset(self) -> MultiCameraDataset:
        """Pair observations with robot poses at their uncorrected timestamps."""
        joint_positions = smooth_joint_trajectory(self.camera_times_s)
        links: dict[str, list[FloatArray]] = {
            "upper_arm": [],
            "forearm": [],
            "gripper": [],
        }
        for joints in joint_positions:
            poses = forward_kinematics(joints)
            for name, transform in poses.items():
                links[name].append(transform)

        return MultiCameraDataset(
            joint_positions_rad=np.asarray(joint_positions, dtype=np.float64),
            link_poses_by_name={
                name: tuple(transforms)
                for name, transforms in links.items()
            },
            target_observations_by_camera={
                name: tuple(transforms)
                for name, transforms in (
                    self.target_observations_by_camera.items()
                )
            },
            camera_extrinsics_ground_truth={
                name: transform.copy()
                for name, transform in (
                    self.camera_extrinsics_ground_truth.items()
                )
            },
            camera_links=dict(self.camera_links),
            T_B_T_ground_truth=self.T_B_T_ground_truth.copy(),
        )


def generate_time_offset_dataset(
    *,
    num_poses: int = 50,
    duration_s: float = 7.0,
    start_time_s: float = 0.5,
    time_offsets_s: dict[str, float] | None = None,
    observation_translation_sigma_mm: float = 0.25,
    observation_rotation_sigma_deg: float = 0.10,
    seed: int = 7,
) -> TimeOffsetDataset:
    """Generate moving-camera observations with per-camera clock offsets."""
    if num_poses < 10:
        raise ValueError("num_poses must be at least 10")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if start_time_s < 0:
        raise ValueError("start_time_s must be non-negative")

    offsets = (
        {name: 0.0 for name in CAMERA_NAMES}
        if time_offsets_s is None
        else {name: float(time_offsets_s[name]) for name in CAMERA_NAMES}
    )
    if max(abs(value) for value in offsets.values()) > 0.15:
        raise ValueError("synthetic offsets must be within +/- 150 ms")

    rng = np.random.default_rng(seed)
    times = np.linspace(
        start_time_s,
        start_time_s + duration_s,
        num_poses,
    )
    mounts = _camera_mounts()
    target = _target_pose()

    observations: dict[str, tuple[FloatArray, ...]] = {}
    for camera_name in CAMERA_NAMES:
        link_name = CAMERA_LINKS[camera_name]
        offset = offsets[camera_name]
        mount = mounts[camera_name]
        camera_observations = []

        for timestamp in times:
            physical_time = float(timestamp + offset)
            joints = smooth_joint_trajectory(physical_time)
            T_B_L = forward_kinematics(joints)[link_name]
            exact = compose(
                inverse(compose(T_B_L, mount)),
                target,
            )
            noisy = perturb_transform_left(
                exact,
                rng,
                translation_sigma_m=(
                    observation_translation_sigma_mm / 1000.0
                ),
                rotation_sigma_deg=observation_rotation_sigma_deg,
            )
            camera_observations.append(noisy)

        observations[camera_name] = tuple(camera_observations)

    return TimeOffsetDataset(
        camera_times_s=times,
        target_observations_by_camera=observations,
        time_offsets_ground_truth_s=offsets,
        camera_extrinsics_ground_truth=mounts,
        camera_links=dict(CAMERA_LINKS),
        T_B_T_ground_truth=target,
    )
