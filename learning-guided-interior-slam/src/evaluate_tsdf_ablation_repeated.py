from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(
    "/workspace/interior-slam/results/"
    "tum_fr1_desk_baseline/tsdf_ablation"
)

MESH_PATHS = {
    "estimated_se3": ROOT / "estimated_se3_mesh.ply",
    "estimated_sim3": ROOT / "estimated_sim3_mesh.ply",
    "estimated_rgbd_scale": (
        ROOT / "estimated_rgbd_scale_mesh.ply"
    ),
    "groundtruth": ROOT / "groundtruth_mesh.ply",
}

OUTPUT_PATH = ROOT / "desk_tsdf_ablation_repeated.json"

SEEDS = [0, 1, 2, 3, 4]
SAMPLE_COUNT = 100_000


def load_mesh(path: Path):
    mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh.is_empty():
        raise RuntimeError(f"Empty mesh: {path}")

    if len(mesh.triangles) == 0:
        raise RuntimeError(
            f"Mesh has no triangles: {path}"
        )

    print(
        path.name,
        "vertices=", len(mesh.vertices),
        "triangles=", len(mesh.triangles),
    )

    return mesh


def sample_surface(
    mesh: o3d.geometry.TriangleMesh,
    seed: int,
):
    o3d.utility.random.seed(seed)

    cloud = mesh.sample_points_uniformly(
        number_of_points=SAMPLE_COUNT
    )

    points = np.asarray(
        cloud.points,
        dtype=np.float64,
    )

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    return points


def symmetric_metrics(
    source_points: np.ndarray,
    target_points: np.ndarray,
):
    target_tree = cKDTree(target_points)
    source_tree = cKDTree(source_points)

    source_to_target = target_tree.query(
        source_points,
        k=1,
        workers=-1,
    )[0]

    target_to_source = source_tree.query(
        target_points,
        k=1,
        workers=-1,
    )[0]

    distances = np.concatenate(
        [source_to_target, target_to_source]
    )

    return {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(
            np.sqrt(np.mean(distances ** 2))
        ),
        "p90_m": float(
            np.percentile(distances, 90)
        ),
        "p95_m": float(
            np.percentile(distances, 95)
        ),
        "coverage_below_2cm": float(
            np.mean(distances < 0.02)
        ),
        "coverage_below_5cm": float(
            np.mean(distances < 0.05)
        ),
        "coverage_below_10cm": float(
            np.mean(distances < 0.10)
        ),
    }


meshes = {
    name: load_mesh(path)
    for name, path in MESH_PATHS.items()
}

variant_names = [
    "estimated_se3",
    "estimated_sim3",
    "estimated_rgbd_scale",
]

runs = {
    name: []
    for name in variant_names
}

for seed in SEEDS:
    print()
    print("Seed:", seed)

    groundtruth_points = sample_surface(
        meshes["groundtruth"],
        seed=10_000 + seed,
    )

    for variant_index, variant_name in enumerate(
        variant_names
    ):
        variant_points = sample_surface(
            meshes[variant_name],
            seed=(
                20_000
                + variant_index * 1_000
                + seed
            ),
        )

        metrics = symmetric_metrics(
            variant_points,
            groundtruth_points,
        )

        metrics["seed"] = seed
        runs[variant_name].append(metrics)

        print(
            variant_name,
            "RMSE=",
            round(metrics["rmse_m"] * 100, 3),
            "cm",
            "within 2 cm=",
            round(
                metrics["coverage_below_2cm"]
                * 100,
                2,
            ),
            "%",
        )


metric_names = [
    "mean_m",
    "median_m",
    "rmse_m",
    "p90_m",
    "p95_m",
    "coverage_below_2cm",
    "coverage_below_5cm",
    "coverage_below_10cm",
]

aggregate = {}

for variant_name, variant_runs in runs.items():
    aggregate[variant_name] = {}

    for metric_name in metric_names:
        values = np.asarray(
            [
                run[metric_name]
                for run in variant_runs
            ],
            dtype=np.float64,
        )

        aggregate[variant_name][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(
                np.std(values, ddof=1)
            ),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }


se3_rmse = aggregate[
    "estimated_se3"
]["rmse_m"]["mean"]

rgbd_rmse = aggregate[
    "estimated_rgbd_scale"
]["rmse_m"]["mean"]

sim3_rmse = aggregate[
    "estimated_sim3"
]["rmse_m"]["mean"]

comparison = {
    "rgbd_rmse_reduction_vs_se3_percent": float(
        100.0 * (se3_rmse - rgbd_rmse)
        / se3_rmse
    ),
    "rgbd_rmse_difference_vs_sim3_percent": float(
        100.0 * (rgbd_rmse - sim3_rmse)
        / sim3_rmse
    ),
}

summary = {
    "sample_count_per_mesh": SAMPLE_COUNT,
    "seeds": SEEDS,
    "runs": runs,
    "aggregate": aggregate,
    "comparison": comparison,
    "interpretation": (
        "Repeated deterministic surface sampling "
        "measures evaluation stability. It does not "
        "provide independent geometric ground truth "
        "because all variants fuse the same RGB-D data."
    ),
}

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, indent=2)


print()
print("========== REPEATED TSDF EVALUATION ==========")

for variant_name in variant_names:
    rmse = aggregate[
        variant_name
    ]["rmse_m"]

    coverage = aggregate[
        variant_name
    ]["coverage_below_2cm"]

    print()
    print(variant_name)
    print(
        "RMSE:",
        round(rmse["mean"] * 100, 3),
        "+/-",
        round(rmse["std"] * 100, 3),
        "cm",
    )
    print(
        "Within 2 cm:",
        round(coverage["mean"] * 100, 2),
        "+/-",
        round(coverage["std"] * 100, 2),
        "%",
    )

print()
print(
    "RGB-D RMSE reduction versus SE3:",
    round(
        comparison[
            "rgbd_rmse_reduction_vs_se3_percent"
        ],
        2,
    ),
    "%",
)

print(
    "RGB-D RMSE difference versus Sim3:",
    round(
        comparison[
            "rgbd_rmse_difference_vs_sim3_percent"
        ],
        2,
    ),
    "%",
)

print()
print("Saved:", OUTPUT_PATH)
print("REPEATED_TSDF_EVAL_OK")
