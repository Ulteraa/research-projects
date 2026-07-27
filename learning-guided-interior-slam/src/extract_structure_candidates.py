from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


ROOT = Path(
    "/workspace/interior-slam/results/"
    "tum_fr1_room_baseline/tsdf/final"
)

INPUT_MESH = (
    ROOT / "portfolio/tum_fr1_room_portfolio_crop.ply"
)

OUTPUT_DIR = ROOT / "structure_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANGLE_TOLERANCE_DEG = 18.0
HEIGHT_BIN_M = 0.025
PLANE_SEED_BAND_M = 0.10
PLANE_INLIER_THRESHOLD_M = 0.035
WALL_VERTICAL_MARGIN_M = 0.12
GRID_RESOLUTION_M = 0.10


def fit_plane_svd(
    points: np.ndarray,
    inlier_threshold: float,
    iterations: int = 3,
):
    if len(points) < 100:
        raise RuntimeError(
            f"Insufficient points for plane fitting: {len(points)}"
        )

    active = np.ones(len(points), dtype=bool)

    for _ in range(iterations):
        current = points[active]
        center = current.mean(axis=0)

        _, _, vt = np.linalg.svd(
            current - center,
            full_matrices=False,
        )

        normal = vt[-1]

        # Make the normal point approximately upward.
        if normal[2] < 0:
            normal = -normal

        normal = normal / np.linalg.norm(normal)
        offset = -float(normal @ center)

        residuals = np.abs(points @ normal + offset)
        active = residuals <= inlier_threshold

        if active.sum() < 100:
            raise RuntimeError(
                "Plane fitting rejected too many points."
            )

    final_points = points[active]
    center = final_points.mean(axis=0)

    _, _, vt = np.linalg.svd(
        final_points - center,
        full_matrices=False,
    )

    normal = vt[-1]

    if normal[2] < 0:
        normal = -normal

    normal = normal / np.linalg.norm(normal)
    offset = -float(normal @ center)

    residuals = np.abs(final_points @ normal + offset)

    return {
        "normal": normal,
        "offset": offset,
        "inlier_mask": active,
        "rmse": float(
            np.sqrt(np.mean(residuals ** 2))
        ),
        "median_residual": float(np.median(residuals)),
        "maximum_residual": float(np.max(residuals)),
    }


def xy_coverage_area(
    points: np.ndarray,
    resolution: float,
) -> float:
    if len(points) == 0:
        return 0.0

    cells = np.floor(
        points[:, :2] / resolution
    ).astype(np.int64)

    unique_cells = np.unique(cells, axis=0)

    return float(
        len(unique_cells) * resolution * resolution
    )


def save_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray | None = None,
):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    if len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(
            np.clip(colors, 0.0, 1.0)
        )

    if normals is not None and len(normals) == len(points):
        cloud.normals = o3d.utility.Vector3dVector(normals)

    success = o3d.io.write_point_cloud(
        str(path),
        cloud,
        write_ascii=False,
        compressed=False,
    )

    if not success:
        raise RuntimeError(f"Failed to save: {path}")

    print("Saved:", path)


print("Reading:", INPUT_MESH)

mesh = o3d.io.read_triangle_mesh(str(INPUT_MESH))

if mesh.is_empty():
    raise RuntimeError(f"Empty mesh: {INPUT_MESH}")

mesh.compute_vertex_normals()

points = np.asarray(mesh.vertices).astype(np.float64)
normals = np.asarray(mesh.vertex_normals).astype(np.float64)
colors = np.asarray(mesh.vertex_colors).astype(np.float64)

if len(colors) != len(points):
    colors = np.full((len(points), 3), 0.65)

normal_lengths = np.linalg.norm(normals, axis=1)

valid = (
    np.isfinite(points).all(axis=1)
    & np.isfinite(normals).all(axis=1)
    & (normal_lengths > 1e-6)
)

points = points[valid]
normals = normals[valid]
colors = colors[valid]

normals = normals / np.linalg.norm(
    normals,
    axis=1,
    keepdims=True,
)

minimum = points.min(axis=0)
maximum = points.max(axis=0)
extent = maximum - minimum

angle_rad = math.radians(ANGLE_TOLERANCE_DEG)

horizontal_mask = (
    np.abs(normals[:, 2]) >= math.cos(angle_rad)
)

vertical_mask = (
    np.abs(normals[:, 2]) <= math.sin(angle_rad)
)

horizontal_points = points[horizontal_mask]

if len(horizontal_points) < 500:
    raise RuntimeError(
        "Not enough horizontal surface candidates."
    )

# Build a height histogram using approximately horizontal surfaces.
z_edges = np.arange(
    minimum[2] - HEIGHT_BIN_M,
    maximum[2] + 2.0 * HEIGHT_BIN_M,
    HEIGHT_BIN_M,
)

height_histogram, z_edges = np.histogram(
    horizontal_points[:, 2],
    bins=z_edges,
)

z_centers = 0.5 * (
    z_edges[:-1] + z_edges[1:]
)

# Smooth the histogram to suppress isolated mesh layers.
kernel = np.ones(7, dtype=np.float64) / 7.0

smoothed_histogram = np.convolve(
    height_histogram.astype(np.float64),
    kernel,
    mode="same",
)

# Restrict floor search to the lowest 15% of scene height.
floor_search = (
    z_centers
    <= minimum[2] + 0.15 * extent[2]
)

# Restrict ceiling search to the highest 15%.
# This prevents tables and shelves from being classified as ceiling.
ceiling_search = (
    z_centers
    >= maximum[2] - 0.15 * extent[2]
)

if not floor_search.any() or not ceiling_search.any():
    raise RuntimeError("Invalid floor or ceiling search interval.")

floor_seed_z = z_centers[floor_search][
    np.argmax(smoothed_histogram[floor_search])
]

ceiling_seed_z = z_centers[ceiling_search][
    np.argmax(smoothed_histogram[ceiling_search])
]

floor_seed_mask = (
    horizontal_mask
    & (
        np.abs(points[:, 2] - floor_seed_z)
        <= PLANE_SEED_BAND_M
    )
)

ceiling_seed_mask = (
    horizontal_mask
    & (
        np.abs(points[:, 2] - ceiling_seed_z)
        <= PLANE_SEED_BAND_M
    )
)

floor_seed_points = points[floor_seed_mask]
ceiling_seed_points = points[ceiling_seed_mask]

floor_fit = fit_plane_svd(
    floor_seed_points,
    PLANE_INLIER_THRESHOLD_M,
)

ceiling_fit = fit_plane_svd(
    ceiling_seed_points,
    PLANE_INLIER_THRESHOLD_M,
)

floor_seed_indices = np.flatnonzero(floor_seed_mask)
ceiling_seed_indices = np.flatnonzero(ceiling_seed_mask)

floor_indices = floor_seed_indices[
    floor_fit["inlier_mask"]
]

ceiling_indices = ceiling_seed_indices[
    ceiling_fit["inlier_mask"]
]

floor_points = points[floor_indices]
ceiling_points = points[ceiling_indices]

floor_z = float(np.median(floor_points[:, 2]))
ceiling_z = float(np.median(ceiling_points[:, 2]))
room_height = ceiling_z - floor_z

if room_height <= 1.5:
    raise RuntimeError(
        f"Implausible room height: {room_height:.3f} m"
    )

wall_mask = (
    vertical_mask
    & (points[:, 2] >= floor_z + WALL_VERTICAL_MARGIN_M)
    & (points[:, 2] <= ceiling_z - WALL_VERTICAL_MARGIN_M)
)

wall_indices = np.flatnonzero(wall_mask)
wall_points = points[wall_indices]

floor_angle = math.degrees(
    math.acos(
        np.clip(
            abs(floor_fit["normal"][2]),
            0.0,
            1.0,
        )
    )
)

ceiling_angle = math.degrees(
    math.acos(
        np.clip(
            abs(ceiling_fit["normal"][2]),
            0.0,
            1.0,
        )
    )
)

save_cloud(
    OUTPUT_DIR / "floor_candidates.ply",
    floor_points,
    colors[floor_indices],
    normals[floor_indices],
)

save_cloud(
    OUTPUT_DIR / "ceiling_candidates.ply",
    ceiling_points,
    colors[ceiling_indices],
    normals[ceiling_indices],
)

save_cloud(
    OUTPUT_DIR / "wall_candidates.ply",
    wall_points,
    colors[wall_indices],
    normals[wall_indices],
)

# Create a color-coded cloud for quick visual inspection.
label_colors = np.full(
    (len(points), 3),
    [0.35, 0.35, 0.35],
    dtype=np.float64,
)

# Floor: blue
label_colors[floor_indices] = [0.15, 0.45, 1.00]

# Ceiling: yellow
label_colors[ceiling_indices] = [1.00, 0.75, 0.10]

# Wall candidates: red
label_colors[wall_indices] = [0.95, 0.20, 0.20]

save_cloud(
    OUTPUT_DIR / "structure_labels.ply",
    points,
    label_colors,
    normals,
)

# Height histogram.
fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

ax.plot(
    z_centers,
    smoothed_histogram,
    linewidth=1.8,
    label="Smoothed horizontal-surface support",
)

ax.axvline(
    floor_z,
    linestyle="--",
    linewidth=2,
    label=f"Floor: {floor_z:.3f} m",
)

ax.axvline(
    ceiling_z,
    linestyle="--",
    linewidth=2,
    label=f"Ceiling: {ceiling_z:.3f} m",
)

ax.set_xlabel("Z height (m)")
ax.set_ylabel("Horizontal-surface support")
ax.set_title("Automatic floor and ceiling estimation")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "height_histogram.png")
plt.close(fig)

# Top-down wall-candidate plot.
rng = np.random.default_rng(42)

wall_plot_count = min(150_000, len(wall_points))
wall_plot_indices = rng.choice(
    len(wall_points),
    wall_plot_count,
    replace=False,
)

fig, ax = plt.subplots(figsize=(9, 10), dpi=160)

ax.scatter(
    wall_points[wall_plot_indices, 0],
    wall_points[wall_plot_indices, 1],
    s=0.25,
    linewidths=0,
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Vertical wall candidates: top-down view")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "wall_candidates_xy.png")
plt.close(fig)

# Side view showing floor, ceiling and wall candidates.
fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

wall_side_count = min(100_000, len(wall_points))
wall_side_indices = rng.choice(
    len(wall_points),
    wall_side_count,
    replace=False,
)

ax.scatter(
    wall_points[wall_side_indices, 0],
    wall_points[wall_side_indices, 2],
    s=0.20,
    label="Wall candidates",
)

ax.scatter(
    floor_points[:, 0],
    floor_points[:, 2],
    s=0.35,
    label="Floor",
)

ax.scatter(
    ceiling_points[:, 0],
    ceiling_points[:, 2],
    s=0.35,
    label="Ceiling",
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Z (m)")
ax.set_title("Structural candidates: XZ view")
ax.legend(markerscale=8)
ax.grid(alpha=0.15)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "structure_xz.png")
plt.close(fig)

summary = {
    "input_mesh": str(INPUT_MESH),
    "input_vertices": int(len(points)),
    "scene_minimum_xyz": minimum.tolist(),
    "scene_maximum_xyz": maximum.tolist(),
    "scene_extent_xyz": extent.tolist(),
    "parameters": {
        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
        "height_bin_m": HEIGHT_BIN_M,
        "plane_seed_band_m": PLANE_SEED_BAND_M,
        "plane_inlier_threshold_m": PLANE_INLIER_THRESHOLD_M,
        "wall_vertical_margin_m": WALL_VERTICAL_MARGIN_M,
        "grid_resolution_m": GRID_RESOLUTION_M,
    },
    "floor": {
        "seed_z_m": float(floor_seed_z),
        "estimated_z_m": floor_z,
        "normal": floor_fit["normal"].tolist(),
        "offset": floor_fit["offset"],
        "normal_angle_from_gravity_deg": floor_angle,
        "support_points": int(len(floor_points)),
        "support_area_m2": xy_coverage_area(
            floor_points,
            GRID_RESOLUTION_M,
        ),
        "plane_rmse_m": floor_fit["rmse"],
        "median_residual_m": floor_fit["median_residual"],
        "maximum_residual_m": floor_fit["maximum_residual"],
    },
    "ceiling": {
        "seed_z_m": float(ceiling_seed_z),
        "estimated_z_m": ceiling_z,
        "normal": ceiling_fit["normal"].tolist(),
        "offset": ceiling_fit["offset"],
        "normal_angle_from_gravity_deg": ceiling_angle,
        "support_points": int(len(ceiling_points)),
        "support_area_m2": xy_coverage_area(
            ceiling_points,
            GRID_RESOLUTION_M,
        ),
        "plane_rmse_m": ceiling_fit["rmse"],
        "median_residual_m": ceiling_fit["median_residual"],
        "maximum_residual_m": ceiling_fit["maximum_residual"],
    },
    "room_height_m": room_height,
    "wall_candidate_points": int(len(wall_points)),
    "horizontal_candidate_points": int(horizontal_mask.sum()),
    "vertical_candidate_points": int(vertical_mask.sum()),
    "outputs": {
        "floor_candidates": str(
            OUTPUT_DIR / "floor_candidates.ply"
        ),
        "ceiling_candidates": str(
            OUTPUT_DIR / "ceiling_candidates.ply"
        ),
        "wall_candidates": str(
            OUTPUT_DIR / "wall_candidates.ply"
        ),
        "structure_labels": str(
            OUTPUT_DIR / "structure_labels.ply"
        ),
    },
}

summary_path = OUTPUT_DIR / "structure_summary.json"

with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print()
print("========== STRUCTURE SUMMARY ==========")
print(json.dumps(summary, indent=2))
print()
print("Saved:", summary_path)
print("STRUCTURE_CANDIDATES_OK")
