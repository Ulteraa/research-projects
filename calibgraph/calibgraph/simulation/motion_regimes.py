"""Synthetic robot-motion regimes for hand-eye observability studies."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, inverse, make_transform
from calibgraph.simulation.eye_in_hand import EyeInHandDataset


MOTION_REGIMES: tuple[str, ...] = (
    "diverse",
    "single_axis",
    "small_rotation",
    "translation_only",
)


def _fixed_calibration_transforms() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed T_G_C and T_B_T used across all motion regimes."""
    T_G_C = make_transform(
        Rotation.from_euler(
            "xyz",
            [12.0, -18.0, 25.0],
            degrees=True,
        ).as_matrix(),
        [0.060, -0.025, 0.110],
    )
    T_B_T = make_transform(
        Rotation.from_euler(
            "xyz",
            [3.0, -5.0, 10.0],
            degrees=True,
        ).as_matrix(),
        [0.750, 0.100, 0.450],
    )
    return T_G_C, T_B_T


def _random_axis(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return axis / norm


def _generate_gripper_poses(
    *,
    motion_regime: str,
    num_poses: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    if motion_regime not in MOTION_REGIMES:
        raise ValueError(
            f"unknown motion regime {motion_regime!r}; "
            f"choose one of {MOTION_REGIMES}"
        )
    if num_poses < 3:
        raise ValueError("num_poses must be at least 3")

    rng = np.random.default_rng(seed)
    poses: list[np.ndarray] = []

    for _ in range(num_poses):
        if motion_regime == "diverse":
            axis = _random_axis(rng)
            angle_deg = rng.uniform(-125.0, 125.0)
            rotation = Rotation.from_rotvec(
                axis * np.deg2rad(angle_deg)
            ).as_matrix()
            translation = np.array(
                [
                    rng.uniform(0.25, 0.70),
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(0.15, 0.75),
                ]
            )

        elif motion_regime == "single_axis":
            angle_deg = rng.uniform(-125.0, 125.0)
            rotation = Rotation.from_euler(
                "z",
                angle_deg,
                degrees=True,
            ).as_matrix()
            translation = np.array(
                [
                    rng.uniform(0.25, 0.70),
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(0.15, 0.75),
                ]
            )

        elif motion_regime == "small_rotation":
            axis = _random_axis(rng)
            angle_deg = rng.uniform(-2.0, 2.0)
            rotation = Rotation.from_rotvec(
                axis * np.deg2rad(angle_deg)
            ).as_matrix()
            translation = np.array(
                [
                    rng.uniform(0.45, 0.55),
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(0.35, 0.45),
                ]
            )

        else:  # translation_only
            rotation = np.eye(3, dtype=np.float64)
            translation = np.array(
                [
                    rng.uniform(0.25, 0.70),
                    rng.uniform(-0.35, 0.35),
                    rng.uniform(0.15, 0.75),
                ]
            )

        poses.append(make_transform(rotation, translation))

    return tuple(poses)


def generate_motion_regime_dataset(
    *,
    motion_regime: str,
    num_poses: int = 25,
    seed: int = 7,
) -> EyeInHandDataset:
    """Generate exact eye-in-hand data for a specified robot motion regime."""
    T_G_C, T_B_T = _fixed_calibration_transforms()
    T_B_G = _generate_gripper_poses(
        motion_regime=motion_regime,
        num_poses=num_poses,
        seed=seed,
    )
    T_C_T = tuple(
        compose(inverse(compose(gripper_pose, T_G_C)), T_B_T)
        for gripper_pose in T_B_G
    )

    return EyeInHandDataset(
        T_B_G=T_B_G,
        T_C_T=T_C_T,
        T_G_C_ground_truth=T_G_C,
        T_B_T_ground_truth=T_B_T,
    )
