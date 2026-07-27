from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
import numpy as np


REPO = Path(
    "/workspace/interior-slam/third_party/MASt3R-SLAM"
)

ESTIMATED = (
    REPO
    / "logs/tum_fr1_desk/rgbd_dataset_freiburg1_desk.txt"
)

GROUND_TRUTH = (
    REPO
    / "datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt"
)

MAX_TIME_DIFFERENCE_S = 0.02


def read_positions(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) < 4:
                continue

            timestamp = float(fields[0])
            position = np.asarray(
                fields[1:4],
                dtype=np.float64,
            )

            records.append((timestamp, position))

    records.sort(key=lambda item: item[0])
    return records


def associate(reference, estimated):
    reference_times = [
        item[0] for item in reference
    ]

    used_reference_indices = set()
    pairs = []

    for estimated_time, estimated_position in estimated:
        insertion = bisect_left(
            reference_times,
            estimated_time,
        )

        candidate_indices = []

        if insertion < len(reference):
            candidate_indices.append(insertion)

        if insertion > 0:
            candidate_indices.append(insertion - 1)

        candidate_indices.sort(
            key=lambda index: abs(
                reference[index][0] - estimated_time
            )
        )

        for reference_index in candidate_indices:
            if reference_index in used_reference_indices:
                continue

            time_difference = abs(
                reference[reference_index][0]
                - estimated_time
            )

            if time_difference > MAX_TIME_DIFFERENCE_S:
                continue

            pairs.append(
                (
                    estimated_position,
                    reference[reference_index][1],
                    time_difference,
                )
            )

            used_reference_indices.add(reference_index)
            break

    return pairs


def umeyama_align(
    source: np.ndarray,
    target: np.ndarray,
    correct_scale: bool,
):
    """
    Estimate target ~= scale * rotation @ source + translation.
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = (
        target_centered.T @ source_centered
    ) / len(source)

    u, singular_values, vt = np.linalg.svd(
        covariance
    )

    correction = np.eye(3)

    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt

    if correct_scale:
        source_variance = np.mean(
            np.sum(source_centered ** 2, axis=1)
        )

        scale = float(
            np.trace(
                np.diag(singular_values)
                @ correction
            )
            / source_variance
        )
    else:
        scale = 1.0

    translation = (
        target_mean
        - scale * rotation @ source_mean
    )

    aligned = (
        scale * (source @ rotation.T)
        + translation
    )

    errors = np.linalg.norm(
        aligned - target,
        axis=1,
    )

    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "errors": errors,
        "rmse": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "maximum": float(errors.max()),
    }


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0

    return float(
        np.linalg.norm(
            np.diff(points, axis=0),
            axis=1,
        ).sum()
    )


reference = read_positions(GROUND_TRUTH)
estimated = read_positions(ESTIMATED)

pairs = associate(reference, estimated)

if len(pairs) < 3:
    raise RuntimeError(
        f"Only {len(pairs)} trajectory associations."
    )

estimated_positions = np.stack(
    [pair[0] for pair in pairs]
)

reference_positions = np.stack(
    [pair[1] for pair in pairs]
)

time_differences = np.asarray(
    [pair[2] for pair in pairs]
)

se3 = umeyama_align(
    estimated_positions,
    reference_positions,
    correct_scale=False,
)

sim3 = umeyama_align(
    estimated_positions,
    reference_positions,
    correct_scale=True,
)

estimated_length = path_length(
    estimated_positions
)

reference_length = path_length(
    reference_positions
)

print("========== DESK SCALE DIAGNOSTIC ==========")
print("Associated poses:", len(pairs))
print(
    "Maximum timestamp difference:",
    round(float(time_differences.max()), 6),
    "s",
)

print()
print("Matched estimated path length:")
print(round(estimated_length, 6), "m")

print("Matched ground-truth path length:")
print(round(reference_length, 6), "m")

print(
    "Ground-truth / estimated path ratio:",
    round(
        reference_length
        / max(estimated_length, 1e-12),
        6,
    ),
)

print()
print("Scale applied to estimated translations:")
print(round(sim3["scale"], 8))

print(
    "Scale deviation from 1:",
    round(abs(sim3["scale"] - 1.0) * 100, 3),
    "%",
)

print()
print("SE3 RMSE:", round(se3["rmse"], 6), "m")
print("SE3 mean:", round(se3["mean"], 6), "m")
print("SE3 median:", round(se3["median"], 6), "m")
print("SE3 max:", round(se3["maximum"], 6), "m")

print()
print("Sim3 RMSE:", round(sim3["rmse"], 6), "m")
print("Sim3 mean:", round(sim3["mean"], 6), "m")
print("Sim3 median:", round(sim3["median"], 6), "m")
print("Sim3 max:", round(sim3["maximum"], 6), "m")

print()
print("DESK_SCALE_DIAGNOSTIC_OK")
