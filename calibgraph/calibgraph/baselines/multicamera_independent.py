"""Independent AX=XB calibration for cameras mounted on arbitrary robot links."""

from __future__ import annotations

from dataclasses import dataclass

from calibgraph.baselines.opencv_hand_eye import solve_opencv_hand_eye
from calibgraph.simulation.articulated_multicamera import MultiCameraDataset


@dataclass(frozen=True)
class IndependentMultiCameraResult:
    method: str
    camera_extrinsics: dict[str, object]


def solve_independent_multicamera(
    dataset: MultiCameraDataset,
    *,
    method: str = "PARK",
) -> IndependentMultiCameraResult:
    """Calibrate each camera independently using its carrying-link trajectory."""
    estimates = {}
    for camera_name in dataset.camera_names:
        result = solve_opencv_hand_eye(
            dataset.as_hand_eye_dataset(camera_name),
            method=method,
        )
        estimates[camera_name] = result.T_G_C_estimate

    return IndependentMultiCameraResult(
        method=method.upper(),
        camera_extrinsics=estimates,
    )
