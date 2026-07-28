"""Evaluation metrics for camera/robot timestamp synchronization."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from calibgraph.geometry.se3 import (
    compose,
    rotation_error_deg,
    translation_error,
)
from calibgraph.simulation.time_sync import TimeOffsetDataset


def evaluate_time_sync_estimate(
    dataset: TimeOffsetDataset,
    *,
    method: str,
    camera_extrinsics: dict[str, object],
    time_offsets_s: dict[str, float],
) -> list[dict[str, object]]:
    target_estimates = {}
    rows = []

    for camera_name in dataset.camera_names:
        estimates = []
        for timestamp, T_C_T in zip(
            dataset.camera_times_s,
            dataset.target_observations_by_camera[camera_name],
            strict=True,
        ):
            T_B_L = dataset.link_pose_at(
                camera_name,
                float(timestamp + time_offsets_s[camera_name]),
            )
            estimates.append(
                compose(
                    T_B_L,
                    camera_extrinsics[camera_name],
                    T_C_T,
                )
            )
        target_estimates[camera_name] = estimates

        target_errors = [
            translation_error(
                estimate,
                dataset.T_B_T_ground_truth,
            )
            * 1000.0
            for estimate in estimates
        ]

        rows.append(
            {
                "method": method,
                "camera_name": camera_name,
                "translation_error_mm": (
                    translation_error(
                        camera_extrinsics[camera_name],
                        dataset.camera_extrinsics_ground_truth[camera_name],
                    )
                    * 1000.0
                ),
                "rotation_error_deg": rotation_error_deg(
                    camera_extrinsics[camera_name],
                    dataset.camera_extrinsics_ground_truth[camera_name],
                ),
                "estimated_time_offset_ms": (
                    time_offsets_s[camera_name] * 1000.0
                ),
                "true_time_offset_ms": (
                    dataset.time_offsets_ground_truth_s[camera_name]
                    * 1000.0
                ),
                "time_offset_abs_error_ms": abs(
                    time_offsets_s[camera_name]
                    - dataset.time_offsets_ground_truth_s[camera_name]
                )
                * 1000.0,
                "mean_target_error_mm": float(
                    np.mean(target_errors)
                ),
            }
        )

    pairwise = []
    for first, second in combinations(dataset.camera_names, 2):
        pairwise.extend(
            translation_error(first_pose, second_pose) * 1000.0
            for first_pose, second_pose in zip(
                target_estimates[first],
                target_estimates[second],
                strict=True,
            )
        )
    mean_disagreement = float(np.mean(pairwise))
    for row in rows:
        row["mean_cross_camera_target_disagreement_mm"] = (
            mean_disagreement
        )
    return rows
