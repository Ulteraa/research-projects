"""Motion observability diagnostics for eye-in-hand calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from calibgraph.geometry.se3 import compose, inverse
from calibgraph.simulation.eye_in_hand import EyeInHandDataset


@dataclass(frozen=True)
class MotionObservability:
    quality: str
    recommendation: str
    num_poses: int
    num_relative_motions: int
    max_relative_rotation_deg: float
    mean_relative_rotation_deg: float
    translation_baseline_mm: float
    rotation_axis_diversity_ratio: float
    rotation_design_rank: int
    expected_rotation_design_rank: int
    smallest_informative_singular_value: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _relative_motion_pairs(
    dataset: EyeInHandDataset,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct consecutive A, B pairs satisfying A X = X B."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(dataset.num_poses - 1):
        next_index = index + 1

        # From H_g_i X H_c_i = H_g_j X H_c_j:
        # A = inv(H_g_j) H_g_i
        # B = H_c_j inv(H_c_i)
        A = compose(
            inverse(dataset.T_B_G[next_index]),
            dataset.T_B_G[index],
        )
        B = compose(
            dataset.T_C_T[next_index],
            inverse(dataset.T_C_T[index]),
        )
        pairs.append((A, B))
    return pairs


def analyze_motion_observability(
    dataset: EyeInHandDataset,
) -> MotionObservability:
    """Analyze whether robot motion sufficiently constrains hand-eye rotation."""
    pairs = _relative_motion_pairs(dataset)

    rotation_vectors = np.asarray(
        [
            Rotation.from_matrix(A[:3, :3]).as_rotvec()
            for A, _ in pairs
        ],
        dtype=np.float64,
    )
    rotation_angles_deg = np.rad2deg(
        np.linalg.norm(rotation_vectors, axis=1)
    )

    axis_singular_values = np.linalg.svd(
        rotation_vectors,
        compute_uv=False,
    )
    if axis_singular_values[0] > 1e-12:
        axis_diversity_ratio = float(
            axis_singular_values[-1] / axis_singular_values[0]
        )
    else:
        axis_diversity_ratio = 0.0

    design_blocks: list[np.ndarray] = []
    for A, B in pairs:
        R_A = A[:3, :3]
        R_B = B[:3, :3]
        design_blocks.append(
            np.kron(np.eye(3), R_A)
            - np.kron(R_B.T, np.eye(3))
        )
    design_matrix = np.vstack(design_blocks)
    design_singular_values = np.linalg.svd(
        design_matrix,
        compute_uv=False,
    )

    absolute_tolerance = 1e-10
    relative_tolerance = (
        1e-8 * design_singular_values[0]
        if design_singular_values[0] > absolute_tolerance
        else absolute_tolerance
    )
    tolerance = max(absolute_tolerance, relative_tolerance)
    rank = int(np.count_nonzero(design_singular_values > tolerance))

    expected_rank = 8
    if rank > 0:
        smallest_informative = float(
            design_singular_values[rank - 1]
        )
    else:
        smallest_informative = 0.0

    translations = np.asarray(
        [transform[:3, 3] for transform in dataset.T_B_G]
    )
    pairwise_distances = np.linalg.norm(
        translations[:, None, :] - translations[None, :, :],
        axis=2,
    )
    translation_baseline_mm = float(
        np.max(pairwise_distances) * 1000.0
    )

    max_rotation = float(np.max(rotation_angles_deg))
    mean_rotation = float(np.mean(rotation_angles_deg))

    if (
        rank < expected_rank
        or max_rotation < 5.0
        or axis_diversity_ratio < 0.02
    ):
        quality = "POOR"
        if rank < expected_rank or axis_diversity_ratio < 0.02:
            recommendation = (
                "Add large rotations about at least two additional axes; "
                "the current motion does not uniquely constrain rotation."
            )
        else:
            recommendation = (
                "Increase rotational excitation; the current angular motion "
                "is too small relative to expected measurement noise."
            )
    elif (
        max_rotation < 20.0
        or axis_diversity_ratio < 0.10
        or smallest_informative < 0.10
    ):
        quality = "WEAK"
        recommendation = (
            "Collect wider, multi-axis wrist rotations and increase pose-space "
            "coverage before accepting the calibration."
        )
    else:
        quality = "GOOD"
        recommendation = (
            "Motion has broad multi-axis rotational excitation; proceed with "
            "solver comparison and residual validation."
        )

    return MotionObservability(
        quality=quality,
        recommendation=recommendation,
        num_poses=dataset.num_poses,
        num_relative_motions=len(pairs),
        max_relative_rotation_deg=max_rotation,
        mean_relative_rotation_deg=mean_rotation,
        translation_baseline_mm=translation_baseline_mm,
        rotation_axis_diversity_ratio=axis_diversity_ratio,
        rotation_design_rank=rank,
        expected_rotation_design_rank=expected_rank,
        smallest_informative_singular_value=smallest_informative,
    )
