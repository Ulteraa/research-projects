from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(
    "/workspace/interior-slam/results/tum_fr1_room_baseline/"
    "tsdf/final/structure_baseline"
)

INPUT_CLOUD = ROOT / "wall_candidates.ply"
STRUCTURE_SUMMARY = ROOT / "structure_summary.json"

OUTPUT_DIR = ROOT / "manhattan_walls"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COORDINATE_BIN_M = 0.04
PEAK_MIN_DISTANCE_M = 0.18
PLANE_SUPPORT_BAND_M = 0.08
MIN_WALL_SEPARATION_M = 2.0
MIN_VERTICAL_SPAN_M = 1.15
MIN_TANGENT_SPAN_M = 0.65
MIN_SUPPORT_POINTS = 180
NORMAL_ALIGNMENT_DEG = 24.0


with STRUCTURE_SUMMARY.open("r", encoding="utf-8") as handle:
    structure = json.load(handle)

floor_z = float(structure["floor"]["estimated_z_m"])
ceiling_z = float(structure["ceiling"]["estimated_z_m"])
room_height = ceiling_z - floor_z

print("Reading:", INPUT_CLOUD)

cloud = o3d.io.read_point_cloud(str(INPUT_CLOUD))

if cloud.is_empty():
    raise RuntimeError(f"Empty point cloud: {INPUT_CLOUD}")

points = np.asarray(cloud.points).astype(np.float64)
normals = np.asarray(cloud.normals).astype(np.float64)

if len(normals) != len(points):
    raise RuntimeError("Wall-candidate cloud does not contain normals.")

valid = (
    np.isfinite(points).all(axis=1)
    & np.isfinite(normals).all(axis=1)
)

points = points[valid]
normals = normals[valid]

normal_lengths = np.linalg.norm(normals, axis=1)
valid_normals = normal_lengths > 1e-8

points = points[valid_normals]
normals = normals[valid_normals]
normals /= np.linalg.norm(normals, axis=1, keepdims=True)

# Ignore uncertain regions very close to the floor and ceiling.
height_mask = (
    (points[:, 2] >= floor_z + 0.15)
    & (points[:, 2] <= ceiling_z - 0.15)
)

points = points[height_mask]
normals = normals[height_mask]

if len(points) < 1000:
    raise RuntimeError("Too few wall candidates after height filtering.")


# ------------------------------------------------------------
# 1. Estimate the Manhattan orientation.
#
# Wall normals are sign-ambiguous and occur in two orthogonal families.
# Multiplying their angles by four makes all Manhattan-equivalent
# directions coincide before computing a circular average.
# ------------------------------------------------------------

normal_angles = np.arctan2(normals[:, 1], normals[:, 0])

normalized_height = np.clip(
    (points[:, 2] - floor_z) / max(room_height, 1e-6),
    0.0,
    1.0,
)

# Give slightly more weight to tall structure than low furniture.
weights = 0.5 + normalized_height

cosine_sum = np.sum(weights * np.cos(4.0 * normal_angles))
sine_sum = np.sum(weights * np.sin(4.0 * normal_angles))

manhattan_angle = 0.25 * math.atan2(sine_sum, cosine_sum)
manhattan_angle %= math.pi / 2.0

concentration = float(
    math.hypot(cosine_sum, sine_sum) / np.sum(weights)
)

cos_angle = math.cos(manhattan_angle)
sin_angle = math.sin(manhattan_angle)

# Rotate the scene by -manhattan_angle.
rotation_to_manhattan = np.array(
    [
        [cos_angle, sin_angle],
        [-sin_angle, cos_angle],
    ],
    dtype=np.float64,
)

rotation_to_world = np.array(
    [
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle],
    ],
    dtype=np.float64,
)

points_xy_m = points[:, :2] @ rotation_to_manhattan.T
normals_xy_m = normals[:, :2] @ rotation_to_manhattan.T

alignment_threshold = math.cos(
    math.radians(NORMAL_ALIGNMENT_DEG)
)

x_family_mask = (
    np.abs(normals_xy_m[:, 0]) >= alignment_threshold
)

y_family_mask = (
    np.abs(normals_xy_m[:, 1]) >= alignment_threshold
)

x_family_points = np.column_stack(
    [points_xy_m[x_family_mask], points[x_family_mask, 2]]
)

y_family_points = np.column_stack(
    [points_xy_m[y_family_mask], points[y_family_mask, 2]]
)

print("Manhattan angle (deg):", math.degrees(manhattan_angle))
print("Manhattan concentration:", concentration)
print("X-normal family points:", len(x_family_points))
print("Y-normal family points:", len(y_family_points))


def quantile_span(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0

    lower, upper = np.percentile(values, [5.0, 95.0])
    return float(upper - lower)


def build_coordinate_evidence(
    family_points: np.ndarray,
    coordinate_axis: int,
    tangent_axis: int,
):
    coordinate = family_points[:, coordinate_axis]
    tangent = family_points[:, tangent_axis]
    height = family_points[:, 2]

    minimum = math.floor(
        coordinate.min() / COORDINATE_BIN_M
    ) * COORDINATE_BIN_M

    maximum = math.ceil(
        coordinate.max() / COORDINATE_BIN_M
    ) * COORDINATE_BIN_M

    edges = np.arange(
        minimum,
        maximum + 2.0 * COORDINATE_BIN_M,
        COORDINATE_BIN_M,
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_ids = np.digitize(coordinate, edges) - 1

    counts = np.zeros(len(centers), dtype=np.int64)
    vertical_spans = np.zeros(len(centers), dtype=np.float64)
    tangent_spans = np.zeros(len(centers), dtype=np.float64)
    raw_scores = np.zeros(len(centers), dtype=np.float64)

    for bin_index in range(len(centers)):
        mask = bin_ids == bin_index
        count = int(mask.sum())

        if count < 20:
            continue

        counts[bin_index] = count
        vertical_span = quantile_span(height[mask])
        tangent_span = quantile_span(tangent[mask])

        vertical_spans[bin_index] = vertical_span
        tangent_spans[bin_index] = tangent_span

        # Walls generally persist both vertically and laterally.
        # Furniture tends to have lower vertical or lateral persistence.
        raw_scores[bin_index] = (
            math.sqrt(count)
            * max(vertical_span, 0.0)
            * math.sqrt(max(tangent_span, 0.0))
        )

    smooth_scores = gaussian_filter1d(
        raw_scores,
        sigma=1.4,
    )

    minimum_peak_distance = max(
        1,
        int(PEAK_MIN_DISTANCE_M / COORDINATE_BIN_M),
    )

    maximum_score = float(smooth_scores.max())

    if maximum_score <= 0:
        raise RuntimeError("No valid coordinate evidence found.")

    peaks, _ = find_peaks(
        smooth_scores,
        distance=minimum_peak_distance,
        prominence=0.06 * maximum_score,
        height=0.08 * maximum_score,
    )

    # Fallback in case peak detection is too restrictive.
    if len(peaks) < 2:
        selected = []

        for index in np.argsort(smooth_scores)[::-1]:
            if smooth_scores[index] <= 0:
                continue

            if all(
                abs(centers[index] - centers[other])
                >= PEAK_MIN_DISTANCE_M
                for other in selected
            ):
                selected.append(int(index))

            if len(selected) >= 8:
                break

        peaks = np.asarray(selected, dtype=np.int64)

    candidates = []

    for peak in peaks:
        center = centers[peak]

        local_mask = (
            np.abs(coordinate - center)
            <= PLANE_SUPPORT_BAND_M
        )

        if local_mask.sum() == 0:
            continue

        refined_coordinate = float(
            np.median(coordinate[local_mask])
        )

        residuals = np.abs(
            coordinate[local_mask] - refined_coordinate
        )

        candidate = {
            "coordinate_m": refined_coordinate,
            "histogram_center_m": float(center),
            "score": float(smooth_scores[peak]),
            "support_points": int(local_mask.sum()),
            "vertical_span_m": quantile_span(height[local_mask]),
            "tangent_span_m": quantile_span(tangent[local_mask]),
            "residual_rmse_m": float(
                np.sqrt(np.mean(residuals ** 2))
            ),
            "residual_median_m": float(np.median(residuals)),
        }

        candidates.append(candidate)

    candidates.sort(key=lambda item: item["coordinate_m"])

    return {
        "centers": centers,
        "raw_scores": raw_scores,
        "smooth_scores": smooth_scores,
        "candidates": candidates,
    }


x_evidence = build_coordinate_evidence(
    x_family_points,
    coordinate_axis=0,
    tangent_axis=1,
)

y_evidence = build_coordinate_evidence(
    y_family_points,
    coordinate_axis=1,
    tangent_axis=0,
)


def choose_boundary_pair(candidates):
    if len(candidates) < 2:
        raise RuntimeError("Fewer than two wall-plane candidates.")

    qualified = [
        candidate
        for candidate in candidates
        if (
            candidate["vertical_span_m"] >= MIN_VERTICAL_SPAN_M
            and candidate["tangent_span_m"] >= MIN_TANGENT_SPAN_M
            and candidate["support_points"] >= MIN_SUPPORT_POINTS
        )
    ]

    if len(qualified) < 2:
        print(
            "Warning: relaxing persistence thresholds because only",
            len(qualified),
            "candidates qualified.",
        )
        qualified = candidates

    maximum_score = max(
        candidate["score"] for candidate in qualified
    )

    full_range = max(
        qualified[-1]["coordinate_m"]
        - qualified[0]["coordinate_m"],
        1e-6,
    )

    best_pair = None
    best_objective = -np.inf

    for first, second in combinations(qualified, 2):
        separation = (
            second["coordinate_m"]
            - first["coordinate_m"]
        )

        if separation < MIN_WALL_SEPARATION_M:
            continue

        normalized_score = (
            first["score"] + second["score"]
        ) / max(2.0 * maximum_score, 1e-6)

        normalized_vertical_support = (
            first["vertical_span_m"]
            + second["vertical_span_m"]
        ) / max(2.0 * room_height, 1e-6)

        normalized_separation = separation / full_range

        objective = (
            1.0 * normalized_score
            + 0.75 * normalized_separation
            + 0.50 * normalized_vertical_support
        )

        if objective > best_objective:
            best_objective = objective
            best_pair = (first, second)

    if best_pair is None:
        qualified.sort(key=lambda item: item["coordinate_m"])
        best_pair = (qualified[0], qualified[-1])

    return best_pair, float(best_objective)


(x_min_wall, x_max_wall), x_objective = choose_boundary_pair(
    x_evidence["candidates"]
)

(y_min_wall, y_max_wall), y_objective = choose_boundary_pair(
    y_evidence["candidates"]
)

x_min = float(x_min_wall["coordinate_m"])
x_max = float(x_max_wall["coordinate_m"])
y_min = float(y_min_wall["coordinate_m"])
y_max = float(y_max_wall["coordinate_m"])

room_width = x_max - x_min
room_length = y_max - y_min

if room_width <= 2.0 or room_length <= 2.0:
    raise RuntimeError(
        f"Implausible room dimensions: "
        f"{room_width:.3f} × {room_length:.3f} m"
    )

print("Selected Manhattan bounds:")
print("  X:", x_min, x_max)
print("  Y:", y_min, y_max)
print("Dimensions:", room_width, room_length, room_height)


def plot_evidence(evidence, selected_pair, axis_label, output):
    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)

    ax.plot(
        evidence["centers"],
        evidence["smooth_scores"],
        linewidth=1.8,
        label="Vertical-persistence wall evidence",
    )

    for candidate in evidence["candidates"]:
        ax.axvline(
            candidate["coordinate_m"],
            linewidth=0.8,
            alpha=0.25,
        )

    ax.axvline(
        selected_pair[0]["coordinate_m"],
        linestyle="--",
        linewidth=2.4,
        label=f"Selected minimum {axis_label} wall",
    )

    ax.axvline(
        selected_pair[1]["coordinate_m"],
        linestyle="--",
        linewidth=2.4,
        label=f"Selected maximum {axis_label} wall",
    )

    ax.set_xlabel(f"{axis_label} coordinate in Manhattan frame (m)")
    ax.set_ylabel("Wall evidence")
    ax.set_title(
        f"{axis_label}-normal wall-plane evidence"
    )
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


plot_evidence(
    x_evidence,
    (x_min_wall, x_max_wall),
    "X",
    OUTPUT_DIR / "x_wall_evidence.png",
)

plot_evidence(
    y_evidence,
    (y_min_wall, y_max_wall),
    "Y",
    OUTPUT_DIR / "y_wall_evidence.png",
)


# ------------------------------------------------------------
# Floor-plan rectangle in the Manhattan frame.
# ------------------------------------------------------------

rectangle_m = np.array(
    [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ],
    dtype=np.float64,
)

rectangle_world = rectangle_m @ rotation_to_world.T

rng = np.random.default_rng(42)
plot_count = min(100_000, len(points_xy_m))
plot_indices = rng.choice(
    len(points_xy_m),
    plot_count,
    replace=False,
)

fig, ax = plt.subplots(figsize=(9, 10), dpi=170)

ax.scatter(
    points_xy_m[plot_indices, 0],
    points_xy_m[plot_indices, 1],
    s=0.2,
    linewidths=0,
    alpha=0.35,
    label="Vertical candidates",
)

ax.plot(
    rectangle_m[:, 0],
    rectangle_m[:, 1],
    linewidth=3,
    label="Selected room envelope",
)

ax.set_xlabel("Manhattan X (m)")
ax.set_ylabel("Manhattan Y (m)")
ax.set_title("Rectangular floor-plan baseline")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "floorplan_rectangle_manhattan.png"
)
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 10), dpi=170)

ax.scatter(
    points[plot_indices, 0],
    points[plot_indices, 1],
    s=0.2,
    linewidths=0,
    alpha=0.35,
    label="Vertical candidates",
)

ax.plot(
    rectangle_world[:, 0],
    rectangle_world[:, 1],
    linewidth=3,
    label="Selected room envelope",
)

ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_title("Room-envelope baseline in reconstruction frame")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "floorplan_rectangle_world.png"
)
plt.close(fig)


# ------------------------------------------------------------
# Export a simple closed room-envelope mesh.
#
# This is a geometric baseline, not a claim that doors and openings
# have already been recovered.
# ------------------------------------------------------------

vertices_m = np.array(
    [
        [x_min, y_min, floor_z],
        [x_max, y_min, floor_z],
        [x_max, y_max, floor_z],
        [x_min, y_max, floor_z],
        [x_min, y_min, ceiling_z],
        [x_max, y_min, ceiling_z],
        [x_max, y_max, ceiling_z],
        [x_min, y_max, ceiling_z],
    ],
    dtype=np.float64,
)

vertices_world = vertices_m.copy()
vertices_world[:, :2] = (
    vertices_m[:, :2] @ rotation_to_world.T
)

triangles = np.array(
    [
        # Floor
        [0, 2, 1],
        [0, 3, 2],

        # Ceiling
        [4, 5, 6],
        [4, 6, 7],

        # Four walls
        [0, 1, 5],
        [0, 5, 4],

        [1, 2, 6],
        [1, 6, 5],

        [2, 3, 7],
        [2, 7, 6],

        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)

envelope_mesh = o3d.geometry.TriangleMesh()
envelope_mesh.vertices = o3d.utility.Vector3dVector(
    vertices_world
)
envelope_mesh.triangles = o3d.utility.Vector3iVector(
    triangles
)

envelope_mesh.compute_vertex_normals()
envelope_mesh.paint_uniform_color([0.72, 0.78, 0.86])

envelope_path = (
    OUTPUT_DIR / "room_envelope_baseline.ply"
)

success = o3d.io.write_triangle_mesh(
    str(envelope_path),
    envelope_mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
)

if not success:
    raise RuntimeError(
        f"Failed to save envelope mesh: {envelope_path}"
    )


def world_normal_from_manhattan(
    normal_m: np.ndarray,
) -> list[float]:
    normal_xy = normal_m[:2] @ rotation_to_world.T

    return [
        float(normal_xy[0]),
        float(normal_xy[1]),
        float(normal_m[2]),
    ]


summary = {
    "input_cloud": str(INPUT_CLOUD),
    "candidate_points_after_height_filter": int(len(points)),
    "floor_z_m": floor_z,
    "ceiling_z_m": ceiling_z,
    "room_height_m": room_height,
    "manhattan": {
        "angle_rad": float(manhattan_angle),
        "angle_deg": float(math.degrees(manhattan_angle)),
        "concentration": concentration,
        "x_family_points": int(len(x_family_points)),
        "y_family_points": int(len(y_family_points)),
    },
    "selected_bounds_manhattan": {
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
    },
    "room_dimensions_m": {
        "width_x": room_width,
        "length_y": room_length,
        "height_z": room_height,
        "floor_area": room_width * room_length,
    },
    "selection_objective": {
        "x_pair": x_objective,
        "y_pair": y_objective,
    },
    "selected_walls": {
        "x_min": x_min_wall,
        "x_max": x_max_wall,
        "y_min": y_min_wall,
        "y_max": y_max_wall,
    },
    "floorplan_corners_manhattan": (
        rectangle_m[:-1].tolist()
    ),
    "floorplan_corners_world": (
        rectangle_world[:-1].tolist()
    ),
    "all_x_candidates": x_evidence["candidates"],
    "all_y_candidates": y_evidence["candidates"],
    "outputs": {
        "x_wall_evidence": str(
            OUTPUT_DIR / "x_wall_evidence.png"
        ),
        "y_wall_evidence": str(
            OUTPUT_DIR / "y_wall_evidence.png"
        ),
        "floorplan_manhattan": str(
            OUTPUT_DIR / "floorplan_rectangle_manhattan.png"
        ),
        "floorplan_world": str(
            OUTPUT_DIR / "floorplan_rectangle_world.png"
        ),
        "room_envelope_mesh": str(envelope_path),
    },
}

summary_path = OUTPUT_DIR / "manhattan_wall_summary.json"

with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print()
print("========== MANHATTAN WALL SUMMARY ==========")
print(json.dumps(summary, indent=2))
print()
print("Saved:", summary_path)
print("MANHATTAN_WALLS_OK")
