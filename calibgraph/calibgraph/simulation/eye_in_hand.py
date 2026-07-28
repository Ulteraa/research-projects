"""Synthetic eye-in-hand calibration data.

Frame notation:
    T_A_B maps coordinates from frame B into frame A.

Frames:
    B: robot base
    G: gripper
    C: camera
    T: calibration target

The fixed unknown is T_G_C (camera to gripper). For each robot pose:

    T_B_T = T_B_G @ T_G_C @ T_C_T

OpenCV names these inputs:
    gripper2base = T_B_G
    target2cam   = T_C_T
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, inverse, make_transform, validate_transform

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EyeInHandDataset:
    """A complete synthetic eye-in-hand calibration sequence."""

    T_B_G: tuple[FloatArray, ...]
    T_C_T: tuple[FloatArray, ...]
    T_G_C_ground_truth: FloatArray
    T_B_T_ground_truth: FloatArray

    def __post_init__(self) -> None:
        if len(self.T_B_G) != len(self.T_C_T):
            raise ValueError("T_B_G and T_C_T must contain the same number of poses")
        if len(self.T_B_G) < 3:
            raise ValueError("hand-eye calibration requires at least 3 poses")

        validate_transform(self.T_G_C_ground_truth)
        validate_transform(self.T_B_T_ground_truth)
        for transform in (*self.T_B_G, *self.T_C_T):
            validate_transform(transform)

    @property
    def num_poses(self) -> int:
        return len(self.T_B_G)


def _sample_rotation(
    rng: np.random.Generator,
    *,
    min_angle_deg: float,
    max_angle_deg: float,
) -> FloatArray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle_deg = rng.uniform(min_angle_deg, max_angle_deg)
    if rng.random() < 0.5:
        angle_deg *= -1.0
    return Rotation.from_rotvec(axis * np.deg2rad(angle_deg)).as_matrix()


def generate_eye_in_hand_dataset(
    *,
    num_poses: int = 25,
    seed: int = 7,
) -> EyeInHandDataset:
    """Generate an exact, noise-free eye-in-hand calibration dataset.

    The trajectory deliberately contains large rotations around diverse axes.
    This avoids the classic motion-degeneracy failure in the first correctness
    benchmark. Later phases will intentionally construct degenerate motion.
    """
    if num_poses < 3:
        raise ValueError("num_poses must be at least 3")

    rng = np.random.default_rng(seed)

    # Fixed camera mounting transform: camera frame C into gripper frame G.
    T_G_C = make_transform(
        Rotation.from_euler("xyz", [12.0, -18.0, 25.0], degrees=True).as_matrix(),
        [0.060, -0.025, 0.110],
    )

    # Fixed calibration target in the robot base frame.
    T_B_T = make_transform(
        Rotation.from_euler("xyz", [3.0, -5.0, 10.0], degrees=True).as_matrix(),
        [0.750, 0.100, 0.450],
    )

    gripper_poses: list[FloatArray] = []
    target_observations: list[FloatArray] = []

    for _ in range(num_poses):
        R_B_G = _sample_rotation(
            rng,
            min_angle_deg=20.0,
            max_angle_deg=125.0,
        )
        t_B_G = np.array(
            [
                rng.uniform(0.25, 0.70),
                rng.uniform(-0.35, 0.35),
                rng.uniform(0.15, 0.75),
            ],
            dtype=np.float64,
        )
        T_B_G = make_transform(R_B_G, t_B_G)

        # From T_B_T = T_B_G T_G_C T_C_T:
        T_C_T = compose(inverse(compose(T_B_G, T_G_C)), T_B_T)

        gripper_poses.append(T_B_G)
        target_observations.append(T_C_T)

    return EyeInHandDataset(
        T_B_G=tuple(gripper_poses),
        T_C_T=tuple(target_observations),
        T_G_C_ground_truth=T_G_C,
        T_B_T_ground_truth=T_B_T,
    )
