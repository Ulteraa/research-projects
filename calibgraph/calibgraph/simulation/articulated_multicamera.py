"""Synthetic articulated robot with cameras mounted on multiple moving links.

Frame convention:
    T_A_B maps coordinates from frame B into frame A.

For camera i mounted on link L_i:

    T_B_T = T_B_L_i(q_t) @ T_L_i_C_i @ T_C_i_T(t)

The unknown extrinsic for camera i is T_L_i_C_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, inverse, make_transform, validate_transform
from calibgraph.simulation.eye_in_hand import EyeInHandDataset

FloatArray = NDArray[np.float64]


CAMERA_NAMES: tuple[str, ...] = (
    "upper_arm_camera",
    "forearm_camera",
    "wrist_camera",
)

CAMERA_LINKS: dict[str, str] = {
    "upper_arm_camera": "upper_arm",
    "forearm_camera": "forearm",
    "wrist_camera": "gripper",
}


@dataclass(frozen=True)
class MultiCameraDataset:
    """Synthetic multi-camera observations from one articulated robot trajectory."""

    joint_positions_rad: FloatArray
    link_poses_by_name: dict[str, tuple[FloatArray, ...]]
    target_observations_by_camera: dict[str, tuple[FloatArray, ...]]
    camera_extrinsics_ground_truth: dict[str, FloatArray]
    camera_links: dict[str, str]
    T_B_T_ground_truth: FloatArray

    def __post_init__(self) -> None:
        if self.joint_positions_rad.ndim != 2 or self.joint_positions_rad.shape[1] != 4:
            raise ValueError("joint_positions_rad must have shape (N, 4)")
        num_poses = self.joint_positions_rad.shape[0]
        if num_poses < 3:
            raise ValueError("at least 3 robot poses are required")

        validate_transform(self.T_B_T_ground_truth)

        for camera_name, link_name in self.camera_links.items():
            if link_name not in self.link_poses_by_name:
                raise ValueError(f"missing link poses for {link_name}")
            if camera_name not in self.target_observations_by_camera:
                raise ValueError(f"missing observations for {camera_name}")
            if camera_name not in self.camera_extrinsics_ground_truth:
                raise ValueError(f"missing ground-truth extrinsic for {camera_name}")

            link_poses = self.link_poses_by_name[link_name]
            observations = self.target_observations_by_camera[camera_name]
            if len(link_poses) != num_poses or len(observations) != num_poses:
                raise ValueError("all camera/link sequences must match trajectory length")

            validate_transform(self.camera_extrinsics_ground_truth[camera_name])
            for transform in (*link_poses, *observations):
                validate_transform(transform)

    @property
    def num_poses(self) -> int:
        return int(self.joint_positions_rad.shape[0])

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self.camera_links)

    def as_hand_eye_dataset(self, camera_name: str) -> EyeInHandDataset:
        """Return one camera/link sequence in the existing AX=XB data format."""
        if camera_name not in self.camera_links:
            raise KeyError(f"unknown camera {camera_name!r}")
        link_name = self.camera_links[camera_name]
        return EyeInHandDataset(
            T_B_G=self.link_poses_by_name[link_name],
            T_C_T=self.target_observations_by_camera[camera_name],
            T_G_C_ground_truth=self.camera_extrinsics_ground_truth[camera_name],
            T_B_T_ground_truth=self.T_B_T_ground_truth,
        )


def _translation(x: float, y: float, z: float) -> FloatArray:
    return make_transform(np.eye(3), [x, y, z])


def _rotation(axis: str, angle_rad: float) -> FloatArray:
    rotation = Rotation.from_euler(axis, angle_rad, degrees=False).as_matrix()
    return make_transform(rotation, [0.0, 0.0, 0.0])


def forward_kinematics(joint_positions_rad: np.ndarray) -> dict[str, FloatArray]:
    """Compute upper-arm, forearm, and gripper poses for a 4-DoF chain.

    Joint order:
        q0: torso yaw
        q1: shoulder pitch
        q2: elbow pitch
        q3: wrist roll
    """
    q = np.asarray(joint_positions_rad, dtype=np.float64).reshape(-1)
    if q.shape != (4,):
        raise ValueError("joint_positions_rad must contain exactly 4 values")

    q0, q1, q2, q3 = q

    T_B_upper = compose(
        _translation(0.0, 0.0, 0.55),
        _rotation("z", q0),
        _rotation("y", q1),
    )
    T_B_forearm = compose(
        T_B_upper,
        _translation(0.34, 0.0, 0.0),
        _rotation("y", q2),
    )
    T_B_gripper = compose(
        T_B_forearm,
        _translation(0.30, 0.0, 0.0),
        _rotation("x", q3),
        _translation(0.12, 0.0, 0.0),
    )
    return {
        "upper_arm": T_B_upper,
        "forearm": T_B_forearm,
        "gripper": T_B_gripper,
    }


def _camera_mounts() -> dict[str, FloatArray]:
    """Ground-truth camera-to-link transforms."""
    return {
        "upper_arm_camera": make_transform(
            Rotation.from_euler("xyz", [8.0, -20.0, 12.0], degrees=True).as_matrix(),
            [0.10, 0.07, 0.05],
        ),
        "forearm_camera": make_transform(
            Rotation.from_euler("xyz", [-12.0, 15.0, -18.0], degrees=True).as_matrix(),
            [0.14, -0.05, 0.035],
        ),
        "wrist_camera": make_transform(
            Rotation.from_euler("xyz", [18.0, -10.0, 28.0], degrees=True).as_matrix(),
            [0.055, 0.025, 0.085],
        ),
    }


def generate_articulated_multicamera_dataset(
    *,
    num_poses: int = 30,
    seed: int = 17,
) -> MultiCameraDataset:
    """Generate exact multi-camera observations for a moving articulated robot."""
    if num_poses < 3:
        raise ValueError("num_poses must be at least 3")

    rng = np.random.default_rng(seed)
    joint_positions = np.column_stack(
        [
            rng.uniform(np.deg2rad(-125.0), np.deg2rad(125.0), size=num_poses),
            rng.uniform(np.deg2rad(-75.0), np.deg2rad(75.0), size=num_poses),
            rng.uniform(np.deg2rad(-105.0), np.deg2rad(105.0), size=num_poses),
            rng.uniform(np.deg2rad(-160.0), np.deg2rad(160.0), size=num_poses),
        ]
    )

    link_poses_lists: dict[str, list[FloatArray]] = {
        "upper_arm": [],
        "forearm": [],
        "gripper": [],
    }
    for joint_state in joint_positions:
        poses = forward_kinematics(joint_state)
        for link_name, transform in poses.items():
            link_poses_lists[link_name].append(transform)

    link_poses = {
        name: tuple(transforms)
        for name, transforms in link_poses_lists.items()
    }

    T_B_T = make_transform(
        Rotation.from_euler("xyz", [4.0, -8.0, 14.0], degrees=True).as_matrix(),
        [0.82, 0.06, 0.48],
    )
    mounts = _camera_mounts()

    observations: dict[str, tuple[FloatArray, ...]] = {}
    for camera_name, link_name in CAMERA_LINKS.items():
        T_L_C = mounts[camera_name]
        observations[camera_name] = tuple(
            compose(inverse(compose(T_B_L, T_L_C)), T_B_T)
            for T_B_L in link_poses[link_name]
        )

    return MultiCameraDataset(
        joint_positions_rad=joint_positions,
        link_poses_by_name=link_poses,
        target_observations_by_camera=observations,
        camera_extrinsics_ground_truth=mounts,
        camera_links=dict(CAMERA_LINKS),
        T_B_T_ground_truth=T_B_T,
    )
