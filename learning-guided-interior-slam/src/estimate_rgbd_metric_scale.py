from __future__ import annotations

import bisect
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


ROOT = Path("/workspace/interior-slam")
REPO = ROOT / "third_party/MASt3R-SLAM"

DATASET = (
    REPO
    / "datasets/tum/rgbd_dataset_freiburg1_desk"
)

ESTIMATED_TRAJECTORY = (
    REPO
    / "logs/tum_fr1_desk/"
    "rgbd_dataset_freiburg1_desk.txt"
)

OUTPUT_DIR = (
    ROOT
    / "results/tum_fr1_desk_baseline/"
    "rgbd_scale_anchor"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "rgbd_metric_scale.json"

MAX_RGB_DIFFERENCE_S = 0.020
MAX_DEPTH_DIFFERENCE_S = 0.030

DEPTH_SCALE = 5000.0
DEPTH_TRUNC_M = 5.0

PAIR_GAPS = (1, 2)

MIN_ESTIMATED_BASELINE_M = 0.04
MAX_ESTIMATED_BASELINE_M = 1.50

MIN_RGBD_BASELINE_M = 0.04
MAX_RGBD_BASELINE_M = 2.00

MAX_ROTATION_DISAGREEMENT_DEG = 15.0
MIN_TRANSLATION_DIRECTION_COSINE = 0.75

MIN_SCALE_RATIO = 0.50
MAX_SCALE_RATIO = 2.00

MIN_ACCEPTED_PAIRS = 3


def read_file_index(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) < 2:
                continue

            records.append(
                (float(fields[0]), fields[1])
            )

    records.sort(key=lambda item: item[0])
    return records


def read_trajectory(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) < 8:
                continue

            values = np.asarray(
                [float(value) for value in fields[1:8]],
                dtype=np.float64,
            )

            records.append(
                (float(fields[0]), values)
            )

    records.sort(key=lambda item: item[0])
    return records


def nearest_record(
    records,
    timestamps,
    query_time,
    maximum_difference,
):
    insertion = bisect.bisect_left(
        timestamps,
        query_time,
    )

    candidates = []

    if insertion < len(records):
        candidates.append(insertion)

    if insertion > 0:
        candidates.append(insertion - 1)

    if not candidates:
        return None

    index = min(
        candidates,
        key=lambda candidate: abs(
            records[candidate][0] - query_time
        ),
    )

    difference = abs(
        records[index][0] - query_time
    )

    if difference > maximum_difference:
        return None

    return records[index], difference


def tum_pose_matrix(values: np.ndarray):
    pose = np.eye(4, dtype=np.float64)

    pose[:3, :3] = Rotation.from_quat(
        values[3:7]
    ).as_matrix()

    pose[:3, 3] = values[:3]

    return pose


def rotation_difference_degrees(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
):
    relative = (
        first_rotation
        @ second_rotation.T
    )

    return float(
        np.degrees(
            Rotation.from_matrix(relative).magnitude()
        )
    )


def load_rgbd(rgb_path: Path, depth_path: Path):
    color = o3d.io.read_image(str(rgb_path))
    depth = o3d.io.read_image(str(depth_path))

    return (
        o3d.geometry.RGBDImage
        .create_from_color_and_depth(
            color,
            depth,
            depth_scale=DEPTH_SCALE,
            depth_trunc=DEPTH_TRUNC_M,
            convert_rgb_to_intensity=False,
        )
    )


def bootstrap_median_interval(
    values: np.ndarray,
    sample_count: int = 5000,
):
    rng = np.random.default_rng(42)

    estimates = np.empty(
        sample_count,
        dtype=np.float64,
    )

    for index in range(sample_count):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        estimates[index] = np.median(sample)

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


rgb_records = read_file_index(
    DATASET / "rgb.txt"
)

depth_records = read_file_index(
    DATASET / "depth.txt"
)

estimated_records = read_trajectory(
    ESTIMATED_TRAJECTORY
)

rgb_timestamps = [
    record[0] for record in rgb_records
]

depth_timestamps = [
    record[0] for record in depth_records
]

frames = []

for timestamp, pose_values in estimated_records:
    rgb_match = nearest_record(
        rgb_records,
        rgb_timestamps,
        timestamp,
        MAX_RGB_DIFFERENCE_S,
    )

    depth_match = nearest_record(
        depth_records,
        depth_timestamps,
        timestamp,
        MAX_DEPTH_DIFFERENCE_S,
    )

    if rgb_match is None or depth_match is None:
        continue

    rgb_record, rgb_difference = rgb_match
    depth_record, depth_difference = depth_match

    frames.append(
        {
            "timestamp": timestamp,
            "rgb_path": DATASET / rgb_record[1],
            "depth_path": DATASET / depth_record[1],
            "pose_wc": tum_pose_matrix(pose_values),
            "rgb_difference_s": rgb_difference,
            "depth_difference_s": depth_difference,
        }
    )

if len(frames) < 4:
    raise RuntimeError(
        f"Only {len(frames)} associated frames."
    )

print("Associated MASt3R keyframes:", len(frames))

intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640,
    480,
    525.0,
    525.0,
    319.5,
    239.5,
)

jacobian = (
    o3d.pipelines.odometry
    .RGBDOdometryJacobianFromHybridTerm()
)

odometry_option = (
    o3d.pipelines.odometry.OdometryOption()
)

rgbd_frames = []

for index, frame in enumerate(frames):
    rgbd_frames.append(
        load_rgbd(
            frame["rgb_path"],
            frame["depth_path"],
        )
    )

    print(
        f"Loaded RGB-D frame "
        f"{index + 1}/{len(frames)}"
    )


pair_results = []
accepted_vectors = []

for gap in PAIR_GAPS:
    for source_index in range(
        len(frames) - gap
    ):
        target_index = source_index + gap

        source_pose_wc = frames[
            source_index
        ]["pose_wc"]

        target_pose_wc = frames[
            target_index
        ]["pose_wc"]

        # Maps source-camera coordinates into
        # target-camera coordinates.
        estimated_source_to_target = (
            np.linalg.inv(target_pose_wc)
            @ source_pose_wc
        )

        estimated_translation = (
            estimated_source_to_target[:3, 3]
        )

        estimated_baseline = float(
            np.linalg.norm(
                estimated_translation
            )
        )

        record = {
            "source_index": source_index,
            "target_index": target_index,
            "gap": gap,
            "source_timestamp": frames[
                source_index
            ]["timestamp"],
            "target_timestamp": frames[
                target_index
            ]["timestamp"],
            "estimated_baseline_m": (
                estimated_baseline
            ),
            "odometry_success": False,
            "accepted_initially": False,
        }

        if not (
            MIN_ESTIMATED_BASELINE_M
            <= estimated_baseline
            <= MAX_ESTIMATED_BASELINE_M
        ):
            record["rejection_reason"] = (
                "estimated baseline outside range"
            )

            pair_results.append(record)
            continue

        success, rgbd_transform, information = (
            o3d.pipelines.odometry
            .compute_rgbd_odometry(
                rgbd_frames[source_index],
                rgbd_frames[target_index],
                intrinsic,
                estimated_source_to_target,
                jacobian,
                odometry_option,
            )
        )

        record["odometry_success"] = bool(success)

        if not success:
            record["rejection_reason"] = (
                "RGB-D odometry failed"
            )

            pair_results.append(record)
            continue

        rgbd_translation = (
            np.asarray(
                rgbd_transform,
                dtype=np.float64,
            )[:3, 3]
        )

        rgbd_baseline = float(
            np.linalg.norm(rgbd_translation)
        )

        rotation_error = (
            rotation_difference_degrees(
                np.asarray(
                    rgbd_transform
                )[:3, :3],
                estimated_source_to_target[
                    :3, :3
                ],
            )
        )

        denominator = (
            estimated_baseline
            * max(rgbd_baseline, 1e-12)
        )

        direction_cosine = float(
            np.dot(
                estimated_translation,
                rgbd_translation,
            )
            / denominator
        )

        projection_scale = float(
            np.dot(
                estimated_translation,
                rgbd_translation,
            )
            / max(
                np.dot(
                    estimated_translation,
                    estimated_translation,
                ),
                1e-12,
            )
        )

        norm_scale = float(
            rgbd_baseline
            / max(estimated_baseline, 1e-12)
        )

        record.update(
            {
                "rgbd_baseline_m": rgbd_baseline,
                "rotation_disagreement_deg": (
                    rotation_error
                ),
                "translation_direction_cosine": (
                    direction_cosine
                ),
                "projection_scale": (
                    projection_scale
                ),
                "norm_scale": norm_scale,
                "information_trace": float(
                    np.trace(information)
                ),
            }
        )

        rejection_reasons = []

        if not (
            MIN_RGBD_BASELINE_M
            <= rgbd_baseline
            <= MAX_RGBD_BASELINE_M
        ):
            rejection_reasons.append(
                "RGB-D baseline outside range"
            )

        if (
            rotation_error
            > MAX_ROTATION_DISAGREEMENT_DEG
        ):
            rejection_reasons.append(
                "rotation disagreement too large"
            )

        if (
            direction_cosine
            < MIN_TRANSLATION_DIRECTION_COSINE
        ):
            rejection_reasons.append(
                "translation direction disagreement"
            )

        if not (
            MIN_SCALE_RATIO
            <= projection_scale
            <= MAX_SCALE_RATIO
        ):
            rejection_reasons.append(
                "scale ratio outside range"
            )

        if rejection_reasons:
            record["rejection_reason"] = "; ".join(
                rejection_reasons
            )
        else:
            record["accepted_initially"] = True

            accepted_vectors.append(
                {
                    "pair_result_index": (
                        len(pair_results)
                    ),
                    "estimated_translation": (
                        estimated_translation.copy()
                    ),
                    "rgbd_translation": (
                        rgbd_translation.copy()
                    ),
                    "projection_scale": (
                        projection_scale
                    ),
                }
            )

        pair_results.append(record)

        print(
            f"pair {source_index:02d}"
            f"->{target_index:02d} "
            f"gap={gap} "
            f"success={success} "
            f"rot={rotation_error:.2f} deg "
            f"cos={direction_cosine:.3f} "
            f"scale={projection_scale:.4f} "
            f"accepted="
            f"{record['accepted_initially']}"
        )


if len(accepted_vectors) < MIN_ACCEPTED_PAIRS:
    summary = {
        "associated_keyframes": len(frames),
        "pair_results": pair_results,
        "error": (
            "Insufficient accepted RGB-D motion pairs"
        ),
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    raise RuntimeError(
        f"Only {len(accepted_vectors)} accepted "
        "RGB-D motion pairs. Diagnostics saved to "
        f"{OUTPUT_JSON}"
    )


initial_scales = np.asarray(
    [
        pair["projection_scale"]
        for pair in accepted_vectors
    ],
    dtype=np.float64,
)

initial_median = float(
    np.median(initial_scales)
)

mad = float(
    np.median(
        np.abs(
            initial_scales - initial_median
        )
    )
)

robust_sigma = 1.4826 * mad

outlier_threshold = max(
    0.08,
    3.0 * robust_sigma,
)

robust_inlier_mask = (
    np.abs(
        initial_scales - initial_median
    )
    <= outlier_threshold
)

robust_vectors = [
    vector
    for vector, keep
    in zip(
        accepted_vectors,
        robust_inlier_mask,
    )
    if keep
]

for vector, keep in zip(
    accepted_vectors,
    robust_inlier_mask,
):
    pair_results[
        vector["pair_result_index"]
    ]["accepted_robustly"] = bool(keep)


if len(robust_vectors) < MIN_ACCEPTED_PAIRS:
    raise RuntimeError(
        "Too few pairs survived robust "
        "scale filtering."
    )


robust_scales = np.asarray(
    [
        vector["projection_scale"]
        for vector in robust_vectors
    ],
    dtype=np.float64,
)

metric_scale = float(
    np.median(robust_scales)
)

scale_ci_low, scale_ci_high = (
    bootstrap_median_interval(
        robust_scales
    )
)

estimated_translation_stack = np.stack(
    [
        vector["estimated_translation"]
        for vector in robust_vectors
    ]
)

rgbd_translation_stack = np.stack(
    [
        vector["rgbd_translation"]
        for vector in robust_vectors
    ]
)

least_squares_scale = float(
    np.sum(
        estimated_translation_stack
        * rgbd_translation_stack
    )
    / np.sum(
        estimated_translation_stack ** 2
    )
)

post_scale_residuals = np.linalg.norm(
    rgbd_translation_stack
    - metric_scale
    * estimated_translation_stack,
    axis=1,
)

summary = {
    "method": (
        "Robust global translation-scale estimate "
        "from metric RGB-D odometry between nearby "
        "MASt3R keyframes. No ground-truth poses "
        "are used."
    ),
    "associated_keyframes": len(frames),
    "tested_pair_count": len(pair_results),
    "initially_accepted_pair_count": (
        len(accepted_vectors)
    ),
    "robust_inlier_pair_count": (
        len(robust_vectors)
    ),
    "metric_scale_median": metric_scale,
    "metric_scale_least_squares": (
        least_squares_scale
    ),
    "bootstrap_95_percent_interval": [
        scale_ci_low,
        scale_ci_high,
    ],
    "initial_scale_median": initial_median,
    "initial_scale_mad": mad,
    "robust_outlier_threshold": (
        outlier_threshold
    ),
    "post_scale_translation_residual": {
        "mean_m": float(
            np.mean(post_scale_residuals)
        ),
        "median_m": float(
            np.median(post_scale_residuals)
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    post_scale_residuals ** 2
                )
            )
        ),
        "maximum_m": float(
            np.max(post_scale_residuals)
        ),
    },
    "pair_results": pair_results,
}

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, indent=2)


print()
print("========== RGB-D METRIC SCALE ==========")
print(
    "Associated keyframes:",
    len(frames),
)
print(
    "Tested pairs:",
    len(pair_results),
)
print(
    "Initially accepted pairs:",
    len(accepted_vectors),
)
print(
    "Robust inlier pairs:",
    len(robust_vectors),
)
print(
    "Metric scale, robust median:",
    round(metric_scale, 8),
)
print(
    "Metric scale, least squares:",
    round(least_squares_scale, 8),
)
print(
    "Bootstrap 95% interval:",
    round(scale_ci_low, 8),
    "to",
    round(scale_ci_high, 8),
)
print(
    "Post-scale translation RMSE:",
    round(
        float(
            np.sqrt(
                np.mean(
                    post_scale_residuals ** 2
                )
            )
        ),
        6,
    ),
    "m",
)
print()
print("Ground-truth-derived reference: 1.11322895")
print(
    "Absolute scale difference:",
    round(
        abs(metric_scale - 1.113228945276162),
        8,
    ),
)
print()
print("Saved:", OUTPUT_JSON)
print("RGBD_METRIC_SCALE_OK")
