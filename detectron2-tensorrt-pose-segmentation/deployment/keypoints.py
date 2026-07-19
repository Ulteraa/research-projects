"""Keypoint heatmap decoding utilities with no TensorRT dependency."""

from __future__ import annotations

import numpy as np


def decode_keypoint_heatmaps(heatmaps: np.ndarray, box_xyxy) -> np.ndarray:
    """Decode K heatmaps into ``[x, y, score]`` coordinates inside a box.

    Args:
        heatmaps: Array shaped ``[K, H, W]`` or ``[H, W]`` for one keypoint.
        box_xyxy: Detection box in original-image ``[xmin, ymin, xmax, ymax]``
            coordinates.

    Returns:
        Float32 array shaped ``[K, 3]``. The score is the maximum raw heatmap
        value; consumers may calibrate or threshold it for their dataset.
    """
    heatmaps = np.asarray(heatmaps)
    if heatmaps.ndim == 2:
        heatmaps = heatmaps[None, ...]
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected [K,H,W] or [H,W], got shape {heatmaps.shape}")

    num_keypoints, height, width = heatmaps.shape
    if height == 0 or width == 0:
        raise ValueError("Heatmap height and width must be non-zero")

    xmin, ymin, xmax, ymax = np.asarray(box_xyxy, dtype=np.float32)
    flat = heatmaps.reshape(num_keypoints, -1)
    maxima = flat.argmax(axis=1)
    rows, cols = np.divmod(maxima, width)

    # The half-pixel offset places a prediction at the center of its heatmap bin.
    xs = xmin + (cols.astype(np.float32) + 0.5) * (xmax - xmin) / width
    ys = ymin + (rows.astype(np.float32) + 0.5) * (ymax - ymin) / height
    scores = flat[np.arange(num_keypoints), maxima].astype(np.float32)
    return np.column_stack((xs, ys, scores)).astype(np.float32, copy=False)
