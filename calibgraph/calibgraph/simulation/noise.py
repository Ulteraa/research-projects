"""Noise models for synthetic calibration experiments.

The current phase perturbs an already estimated target pose T_C_T. This models
pose-space uncertainty after target detection and PnP. A later phase will inject
image-corner noise before PnP for a more sensor-level benchmark.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, make_transform, validate_transform
from calibgraph.simulation.eye_in_hand import EyeInHandDataset

FloatArray = NDArray[np.float64]


def sample_isotropic_pose_noise(
    rng: np.random.Generator,
    *,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
) -> FloatArray:
    """Sample a zero-mean SE(3) perturbation.

    Translation noise is independent Gaussian noise per Cartesian axis.
    Rotation noise uses a random isotropic axis and a Gaussian angle whose
    standard deviation is ``rotation_sigma_deg``.
    """
    if translation_sigma_m < 0:
        raise ValueError("translation_sigma_m must be non-negative")
    if rotation_sigma_deg < 0:
        raise ValueError("rotation_sigma_deg must be non-negative")

    translation = rng.normal(0.0, translation_sigma_m, size=3)

    if rotation_sigma_deg == 0:
        rotation = np.eye(3, dtype=np.float64)
    else:
        axis = rng.normal(size=3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / axis_norm

        angle_rad = np.deg2rad(rng.normal(0.0, rotation_sigma_deg))
        rotation = Rotation.from_rotvec(axis * angle_rad).as_matrix()

    return make_transform(rotation, translation)


def perturb_transform_left(
    transform: FloatArray,
    rng: np.random.Generator,
    *,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
) -> FloatArray:
    """Apply left-multiplicative noise: T_noisy = delta_T @ T.

    For T_C_T observations, this expresses the perturbation in the camera frame,
    which is a reasonable first pose-space approximation to PnP uncertainty.
    """
    validate_transform(transform)
    delta = sample_isotropic_pose_noise(
        rng,
        translation_sigma_m=translation_sigma_m,
        rotation_sigma_deg=rotation_sigma_deg,
    )
    return compose(delta, transform)


def add_target_pose_noise(
    dataset: EyeInHandDataset,
    *,
    seed: int,
    translation_sigma_m: float,
    rotation_sigma_deg: float,
) -> EyeInHandDataset:
    """Return a dataset whose observed T_C_T poses are independently perturbed."""
    rng = np.random.default_rng(seed)
    noisy_observations = tuple(
        perturb_transform_left(
            transform,
            rng,
            translation_sigma_m=translation_sigma_m,
            rotation_sigma_deg=rotation_sigma_deg,
        )
        for transform in dataset.T_C_T
    )
    return replace(dataset, T_C_T=noisy_observations)
