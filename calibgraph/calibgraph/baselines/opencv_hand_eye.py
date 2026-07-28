"""OpenCV hand-eye calibration baselines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from calibgraph.geometry.se3 import make_transform
from calibgraph.simulation.eye_in_hand import EyeInHandDataset

FloatArray = NDArray[np.float64]

HAND_EYE_METHODS: dict[str, int] = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class HandEyeResult:
    method: str
    T_G_C_estimate: FloatArray


def solve_opencv_hand_eye(
    dataset: EyeInHandDataset,
    *,
    method: str,
) -> HandEyeResult:
    """Estimate T_G_C using one OpenCV AX=XB implementation."""
    normalized_method = method.upper()
    if normalized_method not in HAND_EYE_METHODS:
        available = ", ".join(HAND_EYE_METHODS)
        raise ValueError(f"unknown method {method!r}; choose one of: {available}")

    R_gripper2base = [T[:3, :3] for T in dataset.T_B_G]
    t_gripper2base = [T[:3, 3].reshape(3, 1) for T in dataset.T_B_G]
    R_target2cam = [T[:3, :3] for T in dataset.T_C_T]
    t_target2cam = [T[:3, 3].reshape(3, 1) for T in dataset.T_C_T]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=HAND_EYE_METHODS[normalized_method],
    )

    return HandEyeResult(
        method=normalized_method,
        T_G_C_estimate=make_transform(
            R_cam2gripper,
            np.asarray(t_cam2gripper, dtype=np.float64).reshape(3),
        ),
    )


def solve_all_opencv_methods(
    dataset: EyeInHandDataset,
) -> tuple[HandEyeResult, ...]:
    """Run all five classical OpenCV hand-eye solvers."""
    return tuple(
        solve_opencv_hand_eye(dataset, method=method)
        for method in HAND_EYE_METHODS
    )
