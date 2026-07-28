"""Minimal SE(3) parameterization utilities for nonlinear optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import make_transform, validate_transform

FloatArray = NDArray[np.float64]


def transform_to_vector(transform: ArrayLike) -> FloatArray:
    """Convert SE(3) transform to [tx, ty, tz, rx, ry, rz]."""
    T = np.asarray(transform, dtype=np.float64)
    validate_transform(T)
    return np.concatenate(
        [
            T[:3, 3],
            Rotation.from_matrix(T[:3, :3]).as_rotvec(),
        ]
    )


def vector_to_transform(vector: ArrayLike) -> FloatArray:
    """Convert [tx, ty, tz, rx, ry, rz] to an SE(3) transform."""
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (6,):
        raise ValueError(f"expected a 6-vector, got {value.shape}")
    return make_transform(
        Rotation.from_rotvec(value[3:]).as_matrix(),
        value[:3],
    )


def average_transforms(transforms: list[FloatArray]) -> FloatArray:
    """Compute a simple chordal/mean initialization for multiple transforms."""
    if not transforms:
        raise ValueError("at least one transform is required")
    for transform in transforms:
        validate_transform(transform)

    translations = np.asarray([T[:3, 3] for T in transforms])
    rotations = Rotation.from_matrix(
        np.asarray([T[:3, :3] for T in transforms])
    )
    return make_transform(
        rotations.mean().as_matrix(),
        np.mean(translations, axis=0),
    )
