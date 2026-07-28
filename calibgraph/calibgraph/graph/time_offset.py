"""Alternating camera-extrinsic and timestamp-offset refinement.

This CPU-efficient solver alternates between:

1. closed-form PARK hand-eye calibration using current time offsets,
2. a shared target-pose estimate,
3. one-dimensional bounded offset optimization for each camera.

The procedure is a coordinate-descent approximation to a full joint graph and
is practical for online/offline calibration tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.geometry.lie import average_transforms
from calibgraph.geometry.se3 import compose, inverse
from calibgraph.simulation.articulated_multicamera import MultiCameraDataset
from calibgraph.simulation.time_sync import (
    TimeOffsetDataset,
    smooth_joint_trajectory,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TimeAwareCalibrationResult:
    method: str
    camera_extrinsics: dict[str, FloatArray]
    T_B_T_estimate: FloatArray
    time_offsets_s: dict[str, float]
    success: bool
    cost: float
    nfev: int
    runtime_ms: float


def _aligned_dataset(
    dataset: TimeOffsetDataset,
    time_offsets_s: dict[str, float],
) -> MultiCameraDataset:
    """Build per-link pose sequences at corrected physical times.

    MultiCameraDataset stores one trajectory per link, while each camera can
    have a different offset. Here each link has exactly one mounted camera, so
    the link sequence is evaluated using that camera's offset.
    """
    joint_positions_zero = np.asarray(
        smooth_joint_trajectory(dataset.camera_times_s),
        dtype=np.float64,
    )

    link_poses: dict[str, tuple[FloatArray, ...]] = {}
    for camera_name in dataset.camera_names:
        link_name = dataset.camera_links[camera_name]
        delta = time_offsets_s[camera_name]
        link_poses[link_name] = tuple(
            dataset.link_pose_at(
                camera_name,
                float(timestamp + delta),
            )
            for timestamp in dataset.camera_times_s
        )

    return MultiCameraDataset(
        joint_positions_rad=joint_positions_zero,
        link_poses_by_name=link_poses,
        target_observations_by_camera={
            name: tuple(observations)
            for name, observations in (
                dataset.target_observations_by_camera.items()
            )
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


def _estimate_target(
    aligned: MultiCameraDataset,
    extrinsics: dict[str, FloatArray],
) -> FloatArray:
    hypotheses = []
    for camera_name in aligned.camera_names:
        link_name = aligned.camera_links[camera_name]
        X = extrinsics[camera_name]
        for T_B_L, T_C_T in zip(
            aligned.link_poses_by_name[link_name],
            aligned.target_observations_by_camera[camera_name],
            strict=True,
        ):
            hypotheses.append(compose(T_B_L, X, T_C_T))
    return average_transforms(hypotheses)


def _huber(value: np.ndarray, delta: float = 3.0) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(
        absolute <= delta,
        0.5 * value**2,
        delta * (absolute - 0.5 * delta),
    )


def _offset_objective(
    offset_s: float,
    *,
    dataset: TimeOffsetDataset,
    camera_name: str,
    extrinsic: FloatArray,
    target_reference: FloatArray,
    translation_sigma_m: float,
    rotation_sigma_rad: float,
) -> float:
    residual_values = []
    target_inverse = inverse(target_reference)

    for timestamp, T_C_T in zip(
        dataset.camera_times_s,
        dataset.target_observations_by_camera[camera_name],
        strict=True,
    ):
        T_B_L = dataset.link_pose_at(
            camera_name,
            float(timestamp + offset_s),
        )
        target_estimate = compose(T_B_L, extrinsic, T_C_T)
        error = compose(target_inverse, target_estimate)
        translation = error[:3, 3] / translation_sigma_m
        rotation = (
            Rotation.from_matrix(error[:3, :3]).as_rotvec()
            / rotation_sigma_rad
        )
        residual_values.extend(translation.tolist())
        residual_values.extend(rotation.tolist())

    residuals = np.asarray(residual_values, dtype=np.float64)
    return float(np.mean(_huber(residuals)))


def solve_time_aware_multicamera(
    dataset: TimeOffsetDataset,
    *,
    translation_sigma_m: float = 0.00025,
    rotation_sigma_deg: float = 0.10,
    max_offset_s: float = 0.15,
    iterations: int = 3,
) -> TimeAwareCalibrationResult:
    """Estimate offsets and recalibrate camera mounts by coordinate descent."""
    if translation_sigma_m <= 0:
        raise ValueError("translation_sigma_m must be positive")
    if rotation_sigma_deg <= 0:
        raise ValueError("rotation_sigma_deg must be positive")
    if max_offset_s <= 0:
        raise ValueError("max_offset_s must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    start = perf_counter()
    offsets = {name: 0.0 for name in dataset.camera_names}
    total_evaluations = 0
    successful = True
    final_cost = np.nan

    for _ in range(iterations):
        aligned = _aligned_dataset(dataset, offsets)
        calibration = solve_independent_multicamera(
            aligned,
            method="PARK",
        )
        extrinsics = {
            name: np.asarray(transform, dtype=np.float64)
            for name, transform in calibration.camera_extrinsics.items()
        }
        target = _estimate_target(aligned, extrinsics)

        updated_offsets = {}
        camera_costs = []
        for camera_name in dataset.camera_names:
            def objective(value: float) -> float:
                return _offset_objective(
                    value,
                    dataset=dataset,
                    camera_name=camera_name,
                    extrinsic=extrinsics[camera_name],
                    target_reference=target,
                    translation_sigma_m=translation_sigma_m,
                    rotation_sigma_rad=np.deg2rad(rotation_sigma_deg),
                )

            result = minimize_scalar(
                objective,
                bounds=(-max_offset_s, max_offset_s),
                method="bounded",
                options={"xatol": 2e-5, "maxiter": 80},
            )
            updated_offsets[camera_name] = float(result.x)
            camera_costs.append(float(result.fun))
            total_evaluations += int(result.nfev)
            successful = successful and bool(result.success)

        offsets = updated_offsets
        final_cost = float(np.mean(camera_costs))

    final_aligned = _aligned_dataset(dataset, offsets)
    final_calibration = solve_independent_multicamera(
        final_aligned,
        method="PARK",
    )
    final_extrinsics = {
        name: np.asarray(transform, dtype=np.float64)
        for name, transform in (
            final_calibration.camera_extrinsics.items()
        )
    }
    final_target = _estimate_target(
        final_aligned,
        final_extrinsics,
    )
    runtime_ms = (perf_counter() - start) * 1000.0

    return TimeAwareCalibrationResult(
        method="TIME_AWARE_PARK",
        camera_extrinsics=final_extrinsics,
        T_B_T_estimate=final_target,
        time_offsets_s=offsets,
        success=successful,
        cost=final_cost,
        nfev=total_evaluations,
        runtime_ms=runtime_ms,
    )
