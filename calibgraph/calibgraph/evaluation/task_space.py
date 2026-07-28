"""Downstream task-space validation for calibration estimates.

The calibration target frame doubles as a rigid parcel/workpiece frame. A
center grasp point, eight surface verification points, and an approach normal
are transformed into the robot base frame.

Metrics:
    - center grasp-point error
    - surface-point RMSE
    - approach-normal angular error
    - precision and standard action acceptance rates

These are simulated geometric acceptance criteria, not claims about a
particular physical gripper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from calibgraph.geometry.se3 import transform_points
from calibgraph.simulation.articulated_multicamera import MultiCameraDataset
from calibgraph.simulation.time_sync import TimeOffsetDataset

FloatArray = NDArray[np.float64]


# A 300 mm x 200 mm planar workpiece centered at the target origin.
TASK_POINTS_T = np.array(
    [
        [0.000, 0.000, 0.000],  # primary grasp / placement point
        [-0.150, -0.100, 0.000],
        [-0.150, 0.100, 0.000],
        [0.150, -0.100, 0.000],
        [0.150, 0.100, 0.000],
        [-0.150, 0.000, 0.000],
        [0.150, 0.000, 0.000],
        [0.000, -0.100, 0.000],
        [0.000, 0.100, 0.000],
    ],
    dtype=np.float64,
)
TASK_NORMAL_T = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class TaskSpaceThresholds:
    precision_position_mm: float = 2.0
    precision_normal_deg: float = 1.0
    standard_position_mm: float = 5.0
    standard_normal_deg: float = 2.0


def _normal_angle_deg(first: FloatArray, second: FloatArray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def _evaluate_prediction(
    *,
    predicted_points_B: FloatArray,
    predicted_normal_B: FloatArray,
    ground_truth_points_B: FloatArray,
    ground_truth_normal_B: FloatArray,
    thresholds: TaskSpaceThresholds,
) -> dict[str, object]:
    point_errors_mm = (
        np.linalg.norm(
            predicted_points_B - ground_truth_points_B,
            axis=1,
        )
        * 1000.0
    )
    grasp_point_error_mm = float(point_errors_mm[0])
    surface_point_rmse_mm = float(
        np.sqrt(np.mean(point_errors_mm**2))
    )
    max_surface_point_error_mm = float(np.max(point_errors_mm))
    normal_error_deg = _normal_angle_deg(
        predicted_normal_B,
        ground_truth_normal_B,
    )

    return {
        "grasp_point_error_mm": grasp_point_error_mm,
        "surface_point_rmse_mm": surface_point_rmse_mm,
        "max_surface_point_error_mm": max_surface_point_error_mm,
        "approach_normal_error_deg": normal_error_deg,
        "precision_action_success": bool(
            grasp_point_error_mm <= thresholds.precision_position_mm
            and normal_error_deg <= thresholds.precision_normal_deg
        ),
        "standard_action_success": bool(
            grasp_point_error_mm <= thresholds.standard_position_mm
            and normal_error_deg <= thresholds.standard_normal_deg
        ),
    }


def _predict_points_and_normal(
    *,
    T_B_L: FloatArray,
    T_L_C: FloatArray,
    T_C_T: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    T_B_T_estimate = T_B_L @ T_L_C @ T_C_T
    points_B = transform_points(T_B_T_estimate, TASK_POINTS_T)
    normal_B = T_B_T_estimate[:3, :3] @ TASK_NORMAL_T
    normal_B /= np.linalg.norm(normal_B)
    return points_B, normal_B


def _fuse_predictions(
    points_by_camera: list[FloatArray],
    normals_by_camera: list[FloatArray],
) -> tuple[FloatArray, FloatArray]:
    # Coordinate-wise median is robust to one disagreeing camera.
    fused_points = np.median(
        np.stack(points_by_camera, axis=0),
        axis=0,
    )
    fused_normal = np.sum(
        np.stack(normals_by_camera, axis=0),
        axis=0,
    )
    fused_normal /= np.linalg.norm(fused_normal)
    return fused_points, fused_normal


def evaluate_multicamera_task_space(
    dataset: MultiCameraDataset,
    *,
    method: str,
    camera_extrinsics: dict[str, FloatArray],
    thresholds: TaskSpaceThresholds = TaskSpaceThresholds(),
) -> list[dict[str, object]]:
    """Evaluate a calibration estimate on a held-out multi-camera trajectory."""
    ground_truth_points_B = transform_points(
        dataset.T_B_T_ground_truth,
        TASK_POINTS_T,
    )
    ground_truth_normal_B = (
        dataset.T_B_T_ground_truth[:3, :3] @ TASK_NORMAL_T
    )

    rows: list[dict[str, object]] = []
    for time_index in range(dataset.num_poses):
        points_by_camera: list[FloatArray] = []
        normals_by_camera: list[FloatArray] = []

        for camera_name in dataset.camera_names:
            link_name = dataset.camera_links[camera_name]
            points_B, normal_B = _predict_points_and_normal(
                T_B_L=dataset.link_poses_by_name[link_name][time_index],
                T_L_C=camera_extrinsics[camera_name],
                T_C_T=dataset.target_observations_by_camera[
                    camera_name
                ][time_index],
            )
            points_by_camera.append(points_B)
            normals_by_camera.append(normal_B)
            rows.append(
                {
                    "method": method,
                    "source": camera_name,
                    "time_index": time_index,
                    **_evaluate_prediction(
                        predicted_points_B=points_B,
                        predicted_normal_B=normal_B,
                        ground_truth_points_B=ground_truth_points_B,
                        ground_truth_normal_B=ground_truth_normal_B,
                        thresholds=thresholds,
                    ),
                }
            )

        fused_points, fused_normal = _fuse_predictions(
            points_by_camera,
            normals_by_camera,
        )
        rows.append(
            {
                "method": method,
                "source": "FUSED_MEDIAN",
                "time_index": time_index,
                **_evaluate_prediction(
                    predicted_points_B=fused_points,
                    predicted_normal_B=fused_normal,
                    ground_truth_points_B=ground_truth_points_B,
                    ground_truth_normal_B=ground_truth_normal_B,
                    thresholds=thresholds,
                ),
            }
        )

    return rows


def evaluate_time_sync_task_space(
    dataset: TimeOffsetDataset,
    *,
    method: str,
    camera_extrinsics: dict[str, FloatArray],
    time_offsets_s: dict[str, float],
    thresholds: TaskSpaceThresholds = TaskSpaceThresholds(),
) -> list[dict[str, object]]:
    """Evaluate extrinsics and offsets on a held-out moving trajectory."""
    ground_truth_points_B = transform_points(
        dataset.T_B_T_ground_truth,
        TASK_POINTS_T,
    )
    ground_truth_normal_B = (
        dataset.T_B_T_ground_truth[:3, :3] @ TASK_NORMAL_T
    )

    rows: list[dict[str, object]] = []
    for time_index, timestamp in enumerate(dataset.camera_times_s):
        points_by_camera: list[FloatArray] = []
        normals_by_camera: list[FloatArray] = []

        for camera_name in dataset.camera_names:
            corrected_time = float(
                timestamp + time_offsets_s[camera_name]
            )
            points_B, normal_B = _predict_points_and_normal(
                T_B_L=dataset.link_pose_at(
                    camera_name,
                    corrected_time,
                ),
                T_L_C=camera_extrinsics[camera_name],
                T_C_T=dataset.target_observations_by_camera[
                    camera_name
                ][time_index],
            )
            points_by_camera.append(points_B)
            normals_by_camera.append(normal_B)
            rows.append(
                {
                    "method": method,
                    "source": camera_name,
                    "time_index": time_index,
                    **_evaluate_prediction(
                        predicted_points_B=points_B,
                        predicted_normal_B=normal_B,
                        ground_truth_points_B=ground_truth_points_B,
                        ground_truth_normal_B=ground_truth_normal_B,
                        thresholds=thresholds,
                    ),
                }
            )

        fused_points, fused_normal = _fuse_predictions(
            points_by_camera,
            normals_by_camera,
        )
        rows.append(
            {
                "method": method,
                "source": "FUSED_MEDIAN",
                "time_index": time_index,
                **_evaluate_prediction(
                    predicted_points_B=fused_points,
                    predicted_normal_B=fused_normal,
                    ground_truth_points_B=ground_truth_points_B,
                    ground_truth_normal_B=ground_truth_normal_B,
                    thresholds=thresholds,
                ),
            }
        )

    return rows
