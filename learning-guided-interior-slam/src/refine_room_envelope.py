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
    "/workspace/interior-slam/results/tum_fr1_room_baseline/"
    "tsdf/final/structure_baseline"
)

INPUT_SUMMARY = (
    ROOT / "manhattan_walls/manhattan_wall_summary.json"
)
INPUT_CLOUD = ROOT / "wall_candidates.ply"

OUTPUT_DIR = ROOT / "refined_envelope"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUPPORT_POINTS = 1000
MIN_VERTICAL_SPAN_M = 1.70
MIN_TANGENT_SPAN_M = 3.00


with INPUT_SUMMARY.open("r", encoding="utf-8") as handle:
    data = json.load(handle)

cloud = o3d.io.read_point_cloud(str(INPUT_CLOUD))

if cloud.is_empty():
    raise RuntimeError(f"Empty cloud: {INPUT_CLOUD}")

points = np.asarray(cloud.points, dtype=np.float64)

valid = np.isfinite(points).all(axis=1)
points = points[valid]

floor_z = float(data["floor_z_m"])
ceiling_z = float(data["ceiling_z_m"])
room_height = ceiling_z - floor_z

angle = float(data["manhattan"]["angle_rad"])
cos_angle = math.cos(angle)
sin_angle = math.sin(angle)

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

points_m = points[:, :2] @ rotation_to_manhattan.T


def qualify(candidate: dict) -> bool:
    return (
        candidate["support_points"] >= MIN_SUPPORT_POINTS
        and candidate["vertical_span_m"] >= MIN_VERTICAL_SPAN_M
        and candidate["tangent_span_m"] >= MIN_TANGENT_SPAN_M
    )


def classify_candidates(candidates: list[dict]):
    qualified = [
        candidate
        for candidate in candidates
        if qualify(candidate)
    ]

    if len(qualified) < 2:
        raise RuntimeError(
            f"Only {len(qualified)} persistent candidates qualified."
        )

    qualified.sort(
        key=lambda candidate: candidate["coordinate_m"]
    )

    minimum = qualified[0]
    maximum = qualified[-1]
    interior = qualified[1:-1]

    rejected = [
        candidate
        for candidate in candidates
        if not qualify(candidate)
    ]

    return minimum, maximum, interior, rejected


x_min_wall, x_max_wall, x_interior, x_rejected = (
    classify_candidates(data["all_x_candidates"])
)

y_min_wall, y_max_wall, y_interior, y_rejected = (
    classify_candidates(data["all_y_candidates"])
)

x_min = float(x_min_wall["coordinate_m"])
x_max = float(x_max_wall["coordinate_m"])
y_min = float(y_min_wall["coordinate_m"])
y_max = float(y_max_wall["coordinate_m"])

width = x_max - x_min
length = y_max - y_min
floor_area = width * length

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


def draw_internal_planes(ax):
    for candidate in x_interior:
        coordinate = candidate["coordinate_m"]

        ax.plot(
            [coordinate, coordinate],
            [y_min, y_max],
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label="Persistent internal plane",
        )

    for candidate in y_interior:
        coordinate = candidate["coordinate_m"]

        ax.plot(
            [x_min, x_max],
            [coordinate, coordinate],
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label="Persistent internal plane",
        )


rng = np.random.default_rng(42)
count = min(100_000, len(points_m))
indices = rng.choice(len(points_m), count, replace=False)

fig, ax = plt.subplots(figsize=(9, 10), dpi=170)

ax.scatter(
    points_m[indices, 0],
    points_m[indices, 1],
    s=0.2,
    linewidths=0,
    alpha=0.30,
    label="Vertical candidates",
)

ax.plot(
    rectangle_m[:, 0],
    rectangle_m[:, 1],
    linewidth=3,
    label="Refined perimeter envelope",
)

draw_internal_planes(ax)

ax.set_xlabel("Manhattan X (m)")
ax.set_ylabel("Manhattan Y (m)")
ax.set_title("Refined room envelope and internal planes")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "refined_floorplan_manhattan.png")
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 10), dpi=170)

ax.scatter(
    points[indices, 0],
    points[indices, 1],
    s=0.2,
    linewidths=0,
    alpha=0.30,
    label="Vertical candidates",
)

ax.plot(
    rectangle_world[:, 0],
    rectangle_world[:, 1],
    linewidth=3,
    label="Refined perimeter envelope",
)

ax.set_xlabel("World X (m)")
ax.set_ylabel("World Y (m)")
ax.set_title("Refined envelope in reconstruction frame")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "refined_floorplan_world.png")
plt.close(fig)


# Closed room-envelope baseline.
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
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.int32,
)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices_world)
mesh.triangles = o3d.utility.Vector3iVector(triangles)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.72, 0.78, 0.86])

mesh_path = OUTPUT_DIR / "refined_room_envelope.ply"

if not o3d.io.write_triangle_mesh(
    str(mesh_path),
    mesh,
    write_ascii=False,
    compressed=False,
    write_vertex_normals=True,
    write_vertex_colors=True,
):
    raise RuntimeError(f"Failed to save {mesh_path}")


summary = {
    "method": (
        "Outermost persistent Manhattan planes selected as perimeter; "
        "remaining persistent planes classified as internal structure."
    ),
    "thresholds": {
        "minimum_support_points": MIN_SUPPORT_POINTS,
        "minimum_vertical_span_m": MIN_VERTICAL_SPAN_M,
        "minimum_tangent_span_m": MIN_TANGENT_SPAN_M,
    },
    "bounds_manhattan": {
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
    },
    "dimensions_m": {
        "width": width,
        "length": length,
        "height": room_height,
        "floor_area_m2": floor_area,
    },
    "perimeter_walls": {
        "x_min": x_min_wall,
        "x_max": x_max_wall,
        "y_min": y_min_wall,
        "y_max": y_max_wall,
    },
    "internal_persistent_planes": {
        "x_planes": x_interior,
        "y_planes": y_interior,
    },
    "rejected_candidates": {
        "x_planes": x_rejected,
        "y_planes": y_rejected,
    },
    "corners_manhattan": rectangle_m[:-1].tolist(),
    "corners_world": rectangle_world[:-1].tolist(),
    "outputs": {
        "manhattan_plot": str(
            OUTPUT_DIR / "refined_floorplan_manhattan.png"
        ),
        "world_plot": str(
            OUTPUT_DIR / "refined_floorplan_world.png"
        ),
        "envelope_mesh": str(mesh_path),
    },
}

summary_path = OUTPUT_DIR / "refined_envelope_summary.json"

with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print(json.dumps(summary, indent=2))
print("REFINED_ENVELOPE_OK")
