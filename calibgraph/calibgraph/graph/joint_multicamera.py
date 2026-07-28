"""Joint factor-graph-style refinement for articulated multi-camera calibration.

Variables:
    - one fixed camera-to-link extrinsic T_L_C for every camera
    - one shared target pose T_B_T

Factor for camera i at robot pose t:
    T_B_L_i(t) @ T_L_i_C_i @ T_C_i_T(t) == T_B_T

The implementation uses SciPy nonlinear least squares so the project remains
CPU-only and Python-3.13 compatible.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.geometry.lie import (
    average_transforms,
    transform_to_vector,
)
from calibgraph.geometry.se3 import compose
from calibgraph.simulation.articulated_multicamera import MultiCameraDataset

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointMultiCameraResult:
    method: str
    camera_extrinsics: dict[str, FloatArray]
    T_B_T_estimate: FloatArray
    success: bool
    cost: float
    nfev: int
    runtime_ms: float
    optimality: float


def _initial_target_pose(
    dataset: MultiCameraDataset,
    camera_extrinsics: dict[str, FloatArray],
) -> FloatArray:
    target_hypotheses: list[FloatArray] = []
    for camera_name in dataset.camera_names:
        link_name = dataset.camera_links[camera_name]
        X = camera_extrinsics[camera_name]
        for T_B_L, T_C_T in zip(
            dataset.link_poses_by_name[link_name],
            dataset.target_observations_by_camera[camera_name],
            strict=True,
        ):
            target_hypotheses.append(compose(T_B_L, X, T_C_T))
    return average_transforms(target_hypotheses)


def _pack(
    camera_names: tuple[str, ...],
    camera_extrinsics: dict[str, FloatArray],
    T_B_T: FloatArray,
) -> FloatArray:
    blocks = [
        transform_to_vector(camera_extrinsics[name])
        for name in camera_names
    ]
    blocks.append(transform_to_vector(T_B_T))
    return np.concatenate(blocks)


def _vector_to_transform_fast(vector: FloatArray) -> FloatArray:
    value = np.asarray(vector, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(value[3:]).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


def _unpack(
    parameters: FloatArray,
    camera_names: tuple[str, ...],
) -> tuple[dict[str, FloatArray], FloatArray]:
    expected = 6 * (len(camera_names) + 1)
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if values.shape != (expected,):
        raise ValueError(
            f"expected parameter vector of shape ({expected},), "
            f"got {values.shape}"
        )

    camera_extrinsics = {}
    offset = 0
    for name in camera_names:
        camera_extrinsics[name] = _vector_to_transform_fast(
            values[offset : offset + 6]
        )
        offset += 6

    T_B_T = _vector_to_transform_fast(values[offset : offset + 6])
    return camera_extrinsics, T_B_T


def _residual_vector(
    parameters: FloatArray,
    *,
    dataset: MultiCameraDataset,
    translation_sigma_m: float,
    rotation_sigma_rad: float,
) -> FloatArray:
    camera_names = dataset.camera_names
    camera_extrinsics, T_B_T = _unpack(parameters, camera_names)

    R_target = T_B_T[:3, :3]
    t_target = T_B_T[:3, 3]
    R_target_inv = R_target.T

    residual_blocks: list[FloatArray] = []
    for camera_name in camera_names:
        link_name = dataset.camera_links[camera_name]
        X = camera_extrinsics[camera_name]

        link_poses = np.asarray(
            dataset.link_poses_by_name[link_name],
            dtype=np.float64,
        )
        observations = np.asarray(
            dataset.target_observations_by_camera[camera_name],
            dtype=np.float64,
        )
        predicted = link_poses @ X @ observations

        predicted_R = predicted[:, :3, :3]
        predicted_t = predicted[:, :3, 3]
        error_R = R_target_inv[None, :, :] @ predicted_R
        error_t = (
            R_target_inv @ (predicted_t - t_target).T
        ).T

        translation_residual = error_t / translation_sigma_m
        rotation_residual = (
            Rotation.from_matrix(error_R).as_rotvec()
            / rotation_sigma_rad
        )
        residual_blocks.append(
            np.concatenate(
                [translation_residual, rotation_residual],
                axis=1,
            ).reshape(-1)
        )

    return np.concatenate(residual_blocks)


def _jacobian_sparsity(dataset: MultiCameraDataset):
    """Sparse factor-to-variable structure for finite-difference Jacobians."""
    camera_names = dataset.camera_names
    num_factors = len(camera_names) * dataset.num_poses
    num_variables = 6 * (len(camera_names) + 1)
    sparsity = lil_matrix((6 * num_factors, num_variables), dtype=int)

    factor_index = 0
    target_start = 6 * len(camera_names)
    for camera_index, _ in enumerate(camera_names):
        camera_start = 6 * camera_index
        for _pose_index in range(dataset.num_poses):
            row_start = 6 * factor_index
            sparsity[row_start:row_start + 6, camera_start:camera_start + 6] = 1
            sparsity[row_start:row_start + 6, target_start:target_start + 6] = 1
            factor_index += 1
    return sparsity.tocsr()


def solve_joint_multicamera(
    dataset: MultiCameraDataset,
    *,
    initialization_method: str = "PARK",
    loss: str = "linear",
    translation_sigma_m: float = 0.0005,
    rotation_sigma_deg: float = 0.25,
    max_nfev: int = 250,
) -> JointMultiCameraResult:
    """Jointly refine all camera mounts and the shared target pose."""
    if translation_sigma_m <= 0:
        raise ValueError("translation_sigma_m must be positive")
    if rotation_sigma_deg <= 0:
        raise ValueError("rotation_sigma_deg must be positive")
    if loss not in {"linear", "huber", "soft_l1", "cauchy"}:
        raise ValueError(f"unsupported robust loss: {loss}")

    independent = solve_independent_multicamera(
        dataset,
        method=initialization_method,
    )
    initial_extrinsics = {
        name: np.asarray(transform, dtype=np.float64)
        for name, transform in independent.camera_extrinsics.items()
    }
    initial_target = _initial_target_pose(dataset, initial_extrinsics)
    x0 = _pack(dataset.camera_names, initial_extrinsics, initial_target)

    start = perf_counter()
    optimization = least_squares(
        _residual_vector,
        x0,
        kwargs={
            "dataset": dataset,
            "translation_sigma_m": translation_sigma_m,
            "rotation_sigma_rad": np.deg2rad(rotation_sigma_deg),
        },
        method="trf",
        loss=loss,
        f_scale=1.0,
        max_nfev=max_nfev,
        jac_sparsity=_jacobian_sparsity(dataset),
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    runtime_ms = (perf_counter() - start) * 1000.0

    camera_extrinsics, T_B_T = _unpack(
        optimization.x,
        dataset.camera_names,
    )
    return JointMultiCameraResult(
        method=f"JOINT_{loss.upper()}",
        camera_extrinsics=camera_extrinsics,
        T_B_T_estimate=T_B_T,
        success=bool(optimization.success),
        cost=float(optimization.cost),
        nfev=int(optimization.nfev),
        runtime_ms=runtime_ms,
        optimality=float(optimization.optimality),
    )
