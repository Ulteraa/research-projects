from __future__ import annotations

import bisect
import gc
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path("/workspace/interior-slam")
REPO = ROOT / "third_party/MASt3R-SLAM"

DATASET = REPO / "datasets/tum/rgbd_dataset_freiburg1_desk"

ESTIMATED_TRAJECTORY = (
    REPO / "logs/tum_fr1_desk/rgbd_dataset_freiburg1_desk.txt"
)

OUTPUT_DIR = ROOT / "results/tum_fr1_desk_baseline/tsdf_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGBD_SCALE_SUMMARY = (
    ROOT
    / "results/tum_fr1_desk_baseline/"
    "rgbd_scale_anchor/rgbd_metric_scale.json"
)

with RGBD_SCALE_SUMMARY.open(
    "r",
    encoding="utf-8",
) as handle:
    rgbd_scale_summary = json.load(handle)

RGBD_METRIC_SCALE = float(
    rgbd_scale_summary["metric_scale_median"]
)

MAX_RGB_DIFF_S = 0.020
MAX_DEPTH_DIFF_S = 0.030
MAX_GT_DIFF_S = 0.020

VOXEL_LENGTH_M = 0.02
SDF_TRUNC_M = 0.06
DEPTH_TRUNC_M = 5.0
DEPTH_SCALE = 5000.0

SURFACE_SAMPLE_COUNT = 120_000


def read_file_index(path: Path):
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()
            records.append((float(fields[0]), fields[1]))

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

            records.append(
                (
                    float(fields[0]),
                    np.asarray(
                        [float(value) for value in fields[1:8]],
                        dtype=np.float64,
                    ),
                )
            )

    records.sort(key=lambda item: item[0])
    return records


def nearest_record(
    records,
    timestamps,
    query_time,
    maximum_difference,
):
    insertion = bisect.bisect_left(timestamps, query_time)
    candidates = []

    if insertion < len(records):
        candidates.append(insertion)

    if insertion > 0:
        candidates.append(insertion - 1)

    if not candidates:
        return None

    index = min(
        candidates,
        key=lambda i: abs(records[i][0] - query_time),
    )

    record = records[index]
    difference = abs(record[0] - query_time)

    if difference > maximum_difference:
        return None

    return record, difference


def tum_pose_matrix(values: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)

    transform[:3, :3] = Rotation.from_quat(
        values[3:7]
    ).as_matrix()

    transform[:3, 3] = values[:3]
    return transform


def umeyama(
    source: np.ndarray,
    target: np.ndarray,
    correct_scale: bool,
):
    """
    Estimate:
        target ~= scale * R @ source + t
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_zero = source - source_mean
    target_zero = target - target_mean

    covariance = (
        target_zero.T @ source_zero
    ) / len(source)

    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3)

    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt

    if correct_scale:
        source_variance = np.mean(
            np.sum(source_zero ** 2, axis=1)
        )

        scale = float(
            np.trace(
                np.diag(singular_values) @ correction
            ) / source_variance
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

    errors = np.linalg.norm(aligned - target, axis=1)

    return {
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "rmse_m": float(
            np.sqrt(np.mean(errors ** 2))
        ),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "maximum_m": float(np.max(errors)),
    }


def align_pose(
    pose_wc: np.ndarray,
    alignment: dict,
) -> np.ndarray:
    """
    Apply a global SE(3)/Sim(3) alignment to a camera pose.

    Rotation:
        R_aligned = R_global @ R_wc

    Translation:
        t_aligned = scale * R_global @ t_wc + t_global
    """
    output = np.eye(4, dtype=np.float64)

    global_rotation = alignment["rotation"]
    scale = alignment["scale"]
    translation = alignment["translation"]

    output[:3, :3] = (
        global_rotation @ pose_wc[:3, :3]
    )

    output[:3, 3] = (
        scale * global_rotation @ pose_wc[:3, 3]
        + translation
    )

    return output


rgb_records = read_file_index(DATASET / "rgb.txt")
depth_records = read_file_index(DATASET / "depth.txt")
gt_records = read_trajectory(DATASET / "groundtruth.txt")
estimated_records = read_trajectory(ESTIMATED_TRAJECTORY)

rgb_times = [item[0] for item in rgb_records]
depth_times = [item[0] for item in depth_records]
gt_times = [item[0] for item in gt_records]

associations = []

for timestamp, estimated_values in estimated_records:
    rgb_match = nearest_record(
        rgb_records,
        rgb_times,
        timestamp,
        MAX_RGB_DIFF_S,
    )

    depth_match = nearest_record(
        depth_records,
        depth_times,
        timestamp,
        MAX_DEPTH_DIFF_S,
    )

    gt_match = nearest_record(
        gt_records,
        gt_times,
        timestamp,
        MAX_GT_DIFF_S,
    )

    if (
        rgb_match is None
        or depth_match is None
        or gt_match is None
    ):
        continue

    rgb_record, rgb_difference = rgb_match
    depth_record, depth_difference = depth_match
    gt_record, gt_difference = gt_match

    associations.append(
        {
            "timestamp": timestamp,
            "rgb_path": DATASET / rgb_record[1],
            "depth_path": DATASET / depth_record[1],
            "estimated_pose_wc": tum_pose_matrix(
                estimated_values
            ),
            "groundtruth_pose_wc": tum_pose_matrix(
                gt_record[1]
            ),
            "rgb_difference_s": rgb_difference,
            "depth_difference_s": depth_difference,
            "gt_difference_s": gt_difference,
        }
    )

if len(associations) < 5:
    raise RuntimeError(
        f"Only {len(associations)} valid associations."
    )

estimated_centers = np.stack(
    [
        item["estimated_pose_wc"][:3, 3]
        for item in associations
    ]
)

groundtruth_centers = np.stack(
    [
        item["groundtruth_pose_wc"][:3, 3]
        for item in associations
    ]
)

se3_alignment = umeyama(
    estimated_centers,
    groundtruth_centers,
    correct_scale=False,
)

sim3_alignment = umeyama(
    estimated_centers,
    groundtruth_centers,
    correct_scale=True,
)

# Apply the independently estimated RGB-D metric scale.
# Ground truth is used only for the final rigid evaluation
# alignment, not for estimating this scale.
rgbd_scaled_centers = (
    RGBD_METRIC_SCALE * estimated_centers
)

rgbd_scale_se3_alignment = umeyama(
    rgbd_scaled_centers,
    groundtruth_centers,
    correct_scale=False,
)

print("Associations:", len(associations))
print("SE3 RMSE:", se3_alignment["rmse_m"])
print("Sim3 RMSE:", sim3_alignment["rmse_m"])
print("Sim3 scale:", sim3_alignment["scale"])
print("RGB-D metric scale:", RGBD_METRIC_SCALE)
print(
    "RGB-D scaled trajectory SE3 RMSE:",
    rgbd_scale_se3_alignment["rmse_m"],
)


intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640,
    480,
    525.0,
    525.0,
    319.5,
    239.5,
)


def clean_mesh(mesh):
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def integrate_variant(name: str):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_LENGTH_M,
        sdf_trunc=SDF_TRUNC_M,
        color_type=(
            o3d.pipelines.integration
            .TSDFVolumeColorType.RGB8
        ),
        volume_unit_resolution=16,
        depth_sampling_stride=1,
    )

    for index, item in enumerate(associations):
        color = o3d.io.read_image(
            str(item["rgb_path"])
        )

        depth = o3d.io.read_image(
            str(item["depth_path"])
        )

        rgbd = (
            o3d.geometry.RGBDImage
            .create_from_color_and_depth(
                color,
                depth,
                depth_scale=DEPTH_SCALE,
                depth_trunc=DEPTH_TRUNC_M,
                convert_rgb_to_intensity=False,
            )
        )

        if name == "estimated_se3":
            pose_wc = align_pose(
                item["estimated_pose_wc"],
                se3_alignment,
            )

        elif name == "estimated_sim3":
            pose_wc = align_pose(
                item["estimated_pose_wc"],
                sim3_alignment,
            )

        elif name == "estimated_rgbd_scale":
            scaled_pose_wc = (
                item["estimated_pose_wc"].copy()
            )

            scaled_pose_wc[:3, 3] *= (
                RGBD_METRIC_SCALE
            )

            pose_wc = align_pose(
                scaled_pose_wc,
                rgbd_scale_se3_alignment,
            )

        elif name == "groundtruth":
            pose_wc = item["groundtruth_pose_wc"]

        else:
            raise ValueError(name)

        volume.integrate(
            rgbd,
            intrinsic,
            np.linalg.inv(pose_wc),
        )

        if (
            (index + 1) % 5 == 0
            or index == len(associations) - 1
        ):
            print(
                f"[{name}] integrated "
                f"{index + 1}/{len(associations)}"
            )

    mesh = clean_mesh(
        volume.extract_triangle_mesh()
    )

    points = volume.extract_point_cloud()

    mesh_path = OUTPUT_DIR / f"{name}_mesh.ply"
    points_path = OUTPUT_DIR / f"{name}_points.ply"

    o3d.io.write_triangle_mesh(
        str(mesh_path),
        mesh,
        write_ascii=False,
        compressed=False,
        write_vertex_normals=True,
        write_vertex_colors=True,
    )

    o3d.io.write_point_cloud(
        str(points_path),
        points,
        write_ascii=False,
        compressed=False,
    )

    vertices = np.asarray(mesh.vertices)

    result = {
        "name": name,
        "integrated_frames": len(associations),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "pointcloud_points": int(len(points.points)),
        "minimum_xyz": vertices.min(axis=0).tolist(),
        "maximum_xyz": vertices.max(axis=0).tolist(),
        "extent_xyz": (
            vertices.max(axis=0)
            - vertices.min(axis=0)
        ).tolist(),
        "mesh_path": str(mesh_path),
        "pointcloud_path": str(points_path),
    }

    del points
    del volume
    gc.collect()

    return mesh, result


meshes = {}
variant_statistics = {}

for variant in (
    "estimated_se3",
    "estimated_sim3",
    "estimated_rgbd_scale",
    "groundtruth",
):
    mesh, statistics = integrate_variant(variant)
    meshes[variant] = mesh
    variant_statistics[variant] = statistics


def symmetric_surface_distance(
    first_mesh,
    second_mesh,
):
    first_cloud = first_mesh.sample_points_uniformly(
        SURFACE_SAMPLE_COUNT
    )

    second_cloud = second_mesh.sample_points_uniformly(
        SURFACE_SAMPLE_COUNT
    )

    first_points = np.asarray(first_cloud.points)
    second_points = np.asarray(second_cloud.points)

    second_tree = cKDTree(second_points)
    first_tree = cKDTree(first_points)

    first_to_second = second_tree.query(
        first_points,
        k=1,
        workers=-1,
    )[0]

    second_to_first = first_tree.query(
        second_points,
        k=1,
        workers=-1,
    )[0]

    combined = np.concatenate(
        [first_to_second, second_to_first]
    )

    return {
        "mean_m": float(np.mean(combined)),
        "median_m": float(np.median(combined)),
        "rmse_m": float(
            np.sqrt(np.mean(combined ** 2))
        ),
        "p90_m": float(np.percentile(combined, 90)),
        "p95_m": float(np.percentile(combined, 95)),
        "coverage_below_2cm": float(
            np.mean(combined < 0.02)
        ),
        "coverage_below_5cm": float(
            np.mean(combined < 0.05)
        ),
        "coverage_below_10cm": float(
            np.mean(combined < 0.10)
        ),
    }


se3_vs_gt = symmetric_surface_distance(
    meshes["estimated_se3"],
    meshes["groundtruth"],
)

sim3_vs_gt = symmetric_surface_distance(
    meshes["estimated_sim3"],
    meshes["groundtruth"],
)

rgbd_scale_vs_gt = symmetric_surface_distance(
    meshes["estimated_rgbd_scale"],
    meshes["groundtruth"],
)

summary = {
    "dataset": "rgbd_dataset_freiburg1_desk",
    "association_count": len(associations),
    "trajectory_alignment": {
        "se3": {
            "scale": 1.0,
            "rmse_m": se3_alignment["rmse_m"],
            "mean_m": se3_alignment["mean_m"],
            "median_m": se3_alignment["median_m"],
            "maximum_m": se3_alignment["maximum_m"],
        },
        "sim3": {
            "scale": sim3_alignment["scale"],
            "rmse_m": sim3_alignment["rmse_m"],
            "mean_m": sim3_alignment["mean_m"],
            "median_m": sim3_alignment["median_m"],
            "maximum_m": sim3_alignment["maximum_m"],
        },
    },
    "rgbd_metric_scale_evaluation": {
        "estimated_scale": RGBD_METRIC_SCALE,
        "groundtruth_derived_reference_scale": (
            sim3_alignment["scale"]
        ),
        "absolute_scale_difference": abs(
            RGBD_METRIC_SCALE
            - sim3_alignment["scale"]
        ),
        "se3_rmse_after_fixed_rgbd_scale_m": (
            rgbd_scale_se3_alignment["rmse_m"]
        ),
        "note": (
            "Ground truth is used only for rigid evaluation "
            "alignment after fixing the independently estimated "
            "RGB-D scale."
        ),
    },
    "variants": variant_statistics,
    "surface_comparisons": {
        "estimated_se3_vs_groundtruth": se3_vs_gt,
        "estimated_sim3_vs_groundtruth": sim3_vs_gt,
        "estimated_rgbd_scale_vs_groundtruth": (
            rgbd_scale_vs_gt
        ),
    },
    "interpretation": (
        "Both comparisons use the same RGB-D observations. "
        "They measure the influence of camera-pose alignment "
        "and scale, not absolute surface error against an "
        "independent scanned model."
    ),
}

summary_path = OUTPUT_DIR / "desk_tsdf_ablation.json"

with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print()
print("========== DESK TSDF ABLATION ==========")
print("Scale correction:", sim3_alignment["scale"])

print()
print("SE3 estimated mesh versus GT mesh:")
print(json.dumps(se3_vs_gt, indent=2))

print()
print("Sim3 estimated mesh versus GT mesh:")
print(json.dumps(sim3_vs_gt, indent=2))

print()
print("RGB-D-scale estimated mesh versus GT mesh:")
print(json.dumps(rgbd_scale_vs_gt, indent=2))

print()
print("Saved:", summary_path)
print("DESK_TSDF_ABLATION_OK")
