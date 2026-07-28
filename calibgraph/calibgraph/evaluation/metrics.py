"""Evaluation metrics for calibration estimates."""

from __future__ import annotations

from typing import Any

import numpy as np

from calibgraph.baselines.opencv_hand_eye import HandEyeResult
from calibgraph.geometry.se3 import compose, rotation_error_deg, translation_error
from calibgraph.simulation.eye_in_hand import EyeInHandDataset


def evaluate_hand_eye_result(
    dataset: EyeInHandDataset,
    result: HandEyeResult,
) -> dict[str, Any]:
    """Evaluate extrinsic error and reconstructed target consistency."""
    translation_error_m = translation_error(
        result.T_G_C_estimate,
        dataset.T_G_C_ground_truth,
    )
    rotation_error = rotation_error_deg(
        result.T_G_C_estimate,
        dataset.T_G_C_ground_truth,
    )

    reconstructed_targets = [
        compose(T_B_G, result.T_G_C_estimate, T_C_T)
        for T_B_G, T_C_T in zip(dataset.T_B_G, dataset.T_C_T, strict=True)
    ]

    target_translation_errors_m = [
        translation_error(T_B_T_est, dataset.T_B_T_ground_truth)
        for T_B_T_est in reconstructed_targets
    ]
    target_rotation_errors_deg = [
        rotation_error_deg(T_B_T_est, dataset.T_B_T_ground_truth)
        for T_B_T_est in reconstructed_targets
    ]

    return {
        "method": result.method,
        "translation_error_mm": translation_error_m * 1000.0,
        "rotation_error_deg": rotation_error,
        "mean_target_error_mm": float(np.mean(target_translation_errors_m) * 1000.0),
        "max_target_error_mm": float(np.max(target_translation_errors_m) * 1000.0),
        "mean_target_rotation_error_deg": float(
            np.mean(target_rotation_errors_deg)
        ),
        "max_target_rotation_error_deg": float(
            np.max(target_rotation_errors_deg)
        ),
    }
