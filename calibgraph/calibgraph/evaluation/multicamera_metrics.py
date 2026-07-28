"""Evaluation of independent and joint multi-camera calibration estimates."""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np

from calibgraph.geometry.se3 import (
    compose,
    rotation_error_deg,
    translation_error,
)
from calibgraph.simulation.articulated_multicamera import MultiCameraDataset


class MultiCameraEstimate(Protocol):
    method: str
    camera_extrinsics: dict[str, object]


def evaluate_multicamera_estimate(
    clean_dataset: MultiCameraDataset,
    result: MultiCameraEstimate,
) -> list[dict[str, object]]:
    """Return per-camera calibration and task-space consistency metrics."""
    rows: list[dict[str, object]] = []
    target_estimates_by_camera: dict[str, list[np.ndarray]] = {}

    for camera_name in clean_dataset.camera_names:
        link_name = clean_dataset.camera_links[camera_name]
        estimate = result.camera_extrinsics[camera_name]
        ground_truth = clean_dataset.camera_extrinsics_ground_truth[camera_name]

        target_estimates = [
            compose(T_B_L, estimate, T_C_T)
            for T_B_L, T_C_T in zip(
                clean_dataset.link_poses_by_name[link_name],
                clean_dataset.target_observations_by_camera[camera_name],
                strict=True,
            )
        ]
        target_estimates_by_camera[camera_name] = target_estimates

        target_translation_errors = [
            translation_error(T_B_T_est, clean_dataset.T_B_T_ground_truth)
            for T_B_T_est in target_estimates
        ]
        target_rotation_errors = [
            rotation_error_deg(T_B_T_est, clean_dataset.T_B_T_ground_truth)
            for T_B_T_est in target_estimates
        ]

        rows.append(
            {
                "method": result.method,
                "camera_name": camera_name,
                "link_name": link_name,
                "translation_error_mm": translation_error(
                    estimate,
                    ground_truth,
                ) * 1000.0,
                "rotation_error_deg": rotation_error_deg(
                    estimate,
                    ground_truth,
                ),
                "mean_target_error_mm": float(
                    np.mean(target_translation_errors) * 1000.0
                ),
                "max_target_error_mm": float(
                    np.max(target_translation_errors) * 1000.0
                ),
                "mean_target_rotation_error_deg": float(
                    np.mean(target_rotation_errors)
                ),
            }
        )

    pairwise_disagreements_mm: list[float] = []
    for first_camera, second_camera in combinations(
        clean_dataset.camera_names, 2
    ):
        for first_target, second_target in zip(
            target_estimates_by_camera[first_camera],
            target_estimates_by_camera[second_camera],
            strict=True,
        ):
            pairwise_disagreements_mm.append(
                translation_error(first_target, second_target) * 1000.0
            )

    global_mean = float(np.mean(pairwise_disagreements_mm))
    global_max = float(np.max(pairwise_disagreements_mm))
    for row in rows:
        row["mean_cross_camera_target_disagreement_mm"] = global_mean
        row["max_cross_camera_target_disagreement_mm"] = global_max

    return rows


# Backward-compatible name used by Phase 5.
evaluate_independent_multicamera = evaluate_multicamera_estimate
