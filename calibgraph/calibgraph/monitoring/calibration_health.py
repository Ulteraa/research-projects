"""Target-based online calibration health monitoring.

Detection and localization are deliberately separated:

- Detection uses the original validated absolute and cross-camera residuals.
- Localization uses a leave-one-camera-out isolation score.

For camera i:

    isolation(i)
      = mean distance(camera i, peers)
        - mean distance among the peers themselves

If one camera drifts while the other two remain consistent, the drifting
camera receives the largest isolation score.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from calibgraph.baselines.multicamera_independent import (
    solve_independent_multicamera,
)
from calibgraph.geometry.lie import average_transforms
from calibgraph.geometry.se3 import (
    compose,
    rotation_error_deg,
    translation_error,
)
from calibgraph.simulation.articulated_multicamera import MultiCameraDataset
from calibgraph.simulation.drift import subset_multicamera_dataset

FloatArray = NDArray[np.float64]


def _robust_center_scale(
    values: list[float],
    *,
    minimum_scale: float,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    scale = max(1.4826 * mad, minimum_scale)
    return median, scale


def _positive_robust_z(
    value: float,
    center_scale: tuple[float, float],
) -> float:
    center, scale = center_scale
    return max(0.0, (value - center) / scale)


def _pairwise_metrics(
    target_estimates: dict[str, FloatArray],
) -> tuple[
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
]:
    translation_distances: dict[tuple[str, str], float] = {}
    rotation_distances: dict[tuple[str, str], float] = {}

    for first, second in combinations(target_estimates, 2):
        key = tuple(sorted((first, second)))
        translation_distances[key] = (
            translation_error(
                target_estimates[first],
                target_estimates[second],
            )
            * 1000.0
        )
        rotation_distances[key] = rotation_error_deg(
            target_estimates[first],
            target_estimates[second],
        )

    return translation_distances, rotation_distances


def _distance(
    distances: dict[tuple[str, str], float],
    first: str,
    second: str,
) -> float:
    return distances[tuple(sorted((first, second)))]


def _camera_isolation_residual(
    *,
    camera_name: str,
    camera_names: tuple[str, ...],
    distances: dict[tuple[str, str], float],
) -> float:
    peers = tuple(
        name for name in camera_names if name != camera_name
    )
    distances_to_camera = [
        _distance(distances, camera_name, peer)
        for peer in peers
    ]
    peer_pair_distances = [
        _distance(distances, first, second)
        for first, second in combinations(peers, 2)
    ]
    peer_consistency = (
        float(np.mean(peer_pair_distances))
        if peer_pair_distances
        else 0.0
    )
    return max(
        0.0,
        float(np.mean(distances_to_camera)) - peer_consistency,
    )


@dataclass(frozen=True)
class CalibrationHealthReport:
    records: tuple[dict[str, object], ...]
    first_detection_by_camera: dict[str, int | None]
    suspected_camera_by_time: dict[int, str]


@dataclass(frozen=True)
class CalibrationHealthMonitor:
    camera_extrinsics: dict[str, FloatArray]
    target_reference: FloatArray
    baseline_stats: dict[str, dict[str, tuple[float, float]]]
    calibration_window: int
    warning_threshold: float
    critical_threshold: float
    persistence: int

    def evaluate(
        self,
        dataset: MultiCameraDataset,
    ) -> CalibrationHealthReport:
        camera_names = dataset.camera_names
        records: list[dict[str, object]] = []
        critical_streak = {name: 0 for name in camera_names}
        first_detection = {name: None for name in camera_names}
        suspected_camera_by_time: dict[int, str] = {}

        for time_index in range(dataset.num_poses):
            target_estimates = {}
            for camera_name in camera_names:
                link_name = dataset.camera_links[camera_name]
                target_estimates[camera_name] = compose(
                    dataset.link_poses_by_name[link_name][time_index],
                    self.camera_extrinsics[camera_name],
                    dataset.target_observations_by_camera[camera_name][
                        time_index
                    ],
                )

            (
                pairwise_translation_mm,
                pairwise_rotation_deg,
            ) = _pairwise_metrics(target_estimates)

            frame_records: list[dict[str, object]] = []

            for camera_name in camera_names:
                target_estimate = target_estimates[camera_name]
                translation_residual_mm = (
                    translation_error(
                        target_estimate,
                        self.target_reference,
                    )
                    * 1000.0
                )
                rotation_residual_deg = rotation_error_deg(
                    target_estimate,
                    self.target_reference,
                )

                peer_distances_mm = [
                    _distance(
                        pairwise_translation_mm,
                        camera_name,
                        peer_name,
                    )
                    for peer_name in camera_names
                    if peer_name != camera_name
                ]
                cross_camera_residual_mm = float(
                    np.mean(peer_distances_mm)
                )

                isolation_translation_mm = (
                    _camera_isolation_residual(
                        camera_name=camera_name,
                        camera_names=camera_names,
                        distances=pairwise_translation_mm,
                    )
                )
                isolation_rotation_deg = (
                    _camera_isolation_residual(
                        camera_name=camera_name,
                        camera_names=camera_names,
                        distances=pairwise_rotation_deg,
                    )
                )

                stats = self.baseline_stats[camera_name]
                translation_z = _positive_robust_z(
                    translation_residual_mm,
                    stats["translation_mm"],
                )
                rotation_z = _positive_robust_z(
                    rotation_residual_deg,
                    stats["rotation_deg"],
                )
                cross_camera_z = _positive_robust_z(
                    cross_camera_residual_mm,
                    stats["cross_camera_mm"],
                )
                isolation_translation_z = _positive_robust_z(
                    isolation_translation_mm,
                    stats["isolation_translation_mm"],
                )
                isolation_rotation_z = _positive_robust_z(
                    isolation_rotation_deg,
                    stats["isolation_rotation_deg"],
                )

                # Preserve the original validated detection score.
                health_score = float(
                    np.sqrt(
                        translation_z**2
                        + rotation_z**2
                        + cross_camera_z**2
                    )
                )

                # Use isolation only to identify the likely faulty camera.
                isolation_score = float(
                    np.sqrt(
                        isolation_translation_z**2
                        + isolation_rotation_z**2
                    )
                )

                if time_index < self.calibration_window:
                    state = "BASELINE"
                    critical_streak[camera_name] = 0
                elif health_score >= self.critical_threshold:
                    critical_streak[camera_name] += 1
                    if (
                        critical_streak[camera_name]
                        >= self.persistence
                    ):
                        state = "RECALIBRATION_REQUIRED"
                        if first_detection[camera_name] is None:
                            first_detection[camera_name] = (
                                time_index
                                - self.persistence
                                + 1
                            )
                    else:
                        state = "DEGRADED"
                elif health_score >= self.warning_threshold:
                    state = "DEGRADED"
                    critical_streak[camera_name] = 0
                else:
                    state = "HEALTHY"
                    critical_streak[camera_name] = 0

                frame_records.append(
                    {
                        "time_index": time_index,
                        "camera_name": camera_name,
                        "translation_residual_mm": (
                            translation_residual_mm
                        ),
                        "rotation_residual_deg": (
                            rotation_residual_deg
                        ),
                        "cross_camera_residual_mm": (
                            cross_camera_residual_mm
                        ),
                        "isolation_translation_mm": (
                            isolation_translation_mm
                        ),
                        "isolation_rotation_deg": (
                            isolation_rotation_deg
                        ),
                        "translation_z": translation_z,
                        "rotation_z": rotation_z,
                        "cross_camera_z": cross_camera_z,
                        "isolation_translation_z": (
                            isolation_translation_z
                        ),
                        "isolation_rotation_z": (
                            isolation_rotation_z
                        ),
                        "health_score": health_score,
                        "isolation_score": isolation_score,
                        "state": state,
                    }
                )

            suspected_camera_by_time[time_index] = str(
                max(
                    frame_records,
                    key=lambda record: float(
                        record["isolation_score"]
                    ),
                )["camera_name"]
            )
            records.extend(frame_records)

        return CalibrationHealthReport(
            records=tuple(records),
            first_detection_by_camera=first_detection,
            suspected_camera_by_time=suspected_camera_by_time,
        )


def fit_calibration_health_monitor(
    dataset: MultiCameraDataset,
    *,
    calibration_window: int,
    calibration_method: str = "PARK",
    warning_threshold: float = 4.0,
    critical_threshold: float = 8.0,
    persistence: int = 3,
) -> CalibrationHealthMonitor:
    """Fit nominal calibration and robust residual thresholds."""
    if not 3 <= calibration_window < dataset.num_poses:
        raise ValueError("invalid calibration_window")
    if warning_threshold <= 0:
        raise ValueError("warning_threshold must be positive")
    if critical_threshold <= warning_threshold:
        raise ValueError(
            "critical_threshold must exceed warning_threshold"
        )
    if persistence < 1:
        raise ValueError("persistence must be positive")

    calibration_data = subset_multicamera_dataset(
        dataset,
        start=0,
        stop=calibration_window,
    )
    calibration = solve_independent_multicamera(
        calibration_data,
        method=calibration_method,
    )
    extrinsics = {
        name: np.asarray(transform, dtype=np.float64)
        for name, transform in calibration.camera_extrinsics.items()
    }

    target_hypotheses: list[FloatArray] = []
    for camera_name in calibration_data.camera_names:
        link_name = calibration_data.camera_links[camera_name]
        for T_B_L, T_C_T in zip(
            calibration_data.link_poses_by_name[link_name],
            calibration_data.target_observations_by_camera[camera_name],
            strict=True,
        ):
            target_hypotheses.append(
                compose(
                    T_B_L,
                    extrinsics[camera_name],
                    T_C_T,
                )
            )
    target_reference = average_transforms(target_hypotheses)

    metric_names = (
        "translation_mm",
        "rotation_deg",
        "cross_camera_mm",
        "isolation_translation_mm",
        "isolation_rotation_deg",
    )
    raw_metrics: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metric_names}
        for name in calibration_data.camera_names
    }

    for time_index in range(calibration_window):
        target_estimates = {}
        for camera_name in calibration_data.camera_names:
            link_name = calibration_data.camera_links[camera_name]
            target_estimates[camera_name] = compose(
                calibration_data.link_poses_by_name[link_name][
                    time_index
                ],
                extrinsics[camera_name],
                calibration_data.target_observations_by_camera[
                    camera_name
                ][time_index],
            )

        (
            pairwise_translation_mm,
            pairwise_rotation_deg,
        ) = _pairwise_metrics(target_estimates)

        for camera_name in calibration_data.camera_names:
            estimate = target_estimates[camera_name]
            raw_metrics[camera_name]["translation_mm"].append(
                translation_error(
                    estimate,
                    target_reference,
                )
                * 1000.0
            )
            raw_metrics[camera_name]["rotation_deg"].append(
                rotation_error_deg(estimate, target_reference)
            )
            raw_metrics[camera_name]["cross_camera_mm"].append(
                float(
                    np.mean(
                        [
                            _distance(
                                pairwise_translation_mm,
                                camera_name,
                                peer,
                            )
                            for peer in calibration_data.camera_names
                            if peer != camera_name
                        ]
                    )
                )
            )
            raw_metrics[camera_name][
                "isolation_translation_mm"
            ].append(
                _camera_isolation_residual(
                    camera_name=camera_name,
                    camera_names=calibration_data.camera_names,
                    distances=pairwise_translation_mm,
                )
            )
            raw_metrics[camera_name][
                "isolation_rotation_deg"
            ].append(
                _camera_isolation_residual(
                    camera_name=camera_name,
                    camera_names=calibration_data.camera_names,
                    distances=pairwise_rotation_deg,
                )
            )

    minimum_scales = {
        "translation_mm": 0.08,
        "rotation_deg": 0.015,
        "cross_camera_mm": 0.10,
        "isolation_translation_mm": 0.08,
        "isolation_rotation_deg": 0.015,
    }
    baseline_stats = {}
    for camera_name, metrics in raw_metrics.items():
        baseline_stats[camera_name] = {
            metric_name: _robust_center_scale(
                values,
                minimum_scale=minimum_scales[metric_name],
            )
            for metric_name, values in metrics.items()
        }

    return CalibrationHealthMonitor(
        camera_extrinsics=extrinsics,
        target_reference=target_reference,
        baseline_stats=baseline_stats,
        calibration_window=calibration_window,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
        persistence=persistence,
    )
