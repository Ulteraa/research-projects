from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing


ROOT = Path(
    "/workspace/interior-slam/results/tum_fr1_room_baseline/"
    "tsdf/final/structure_baseline"
)

WALL_CLOUD = ROOT / "wall_candidates.ply"

MANHATTAN_SUMMARY = (
    ROOT / "manhattan_walls/manhattan_wall_summary.json"
)

REFINED_SUMMARY = (
    ROOT / "refined_envelope/refined_envelope_summary.json"
)

OUTPUT_DIR = ROOT / "opening_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WALL_DISTANCE_M = 0.10
TANGENT_RESOLUTION_M = 0.05
HEIGHT_RESOLUTION_M = 0.05
MIN_POINTS_PER_CELL = 2

MIN_GAP_WIDTH_M = 0.35
MAX_GAP_WIDTH_M = 2.20
EDGE_MARGIN_M = 0.15
LOW_COVERAGE_THRESHOLD = 0.10


with MANHATTAN_SUMMARY.open("r", encoding="utf-8") as handle:
    manhattan_data = json.load(handle)

with REFINED_SUMMARY.open("r", encoding="utf-8") as handle:
    refined_data = json.load(handle)

cloud = o3d.io.read_point_cloud(str(WALL_CLOUD))

if cloud.is_empty():
    raise RuntimeError(f"Empty cloud: {WALL_CLOUD}")

points = np.asarray(cloud.points, dtype=np.float64)

valid = np.isfinite(points).all(axis=1)
points = points[valid]

floor_z = float(manhattan_data["floor_z_m"])
ceiling_z = float(manhattan_data["ceiling_z_m"])
room_height = ceiling_z - floor_z

angle = float(manhattan_data["manhattan"]["angle_rad"])

rotation_to_manhattan = np.array(
    [
        [math.cos(angle), math.sin(angle)],
        [-math.sin(angle), math.cos(angle)],
    ],
    dtype=np.float64,
)

points_xy_m = points[:, :2] @ rotation_to_manhattan.T

x_min = float(refined_data["bounds_manhattan"]["x_min_m"])
x_max = float(refined_data["bounds_manhattan"]["x_max_m"])
y_min = float(refined_data["bounds_manhattan"]["y_min_m"])
y_max = float(refined_data["bounds_manhattan"]["y_max_m"])


WALLS = {
    "x_min": {
        "normal_axis": 0,
        "plane": x_min,
        "tangent_axis": 1,
        "tangent_min": y_min,
        "tangent_max": y_max,
    },
    "x_max": {
        "normal_axis": 0,
        "plane": x_max,
        "tangent_axis": 1,
        "tangent_min": y_min,
        "tangent_max": y_max,
    },
    "y_min": {
        "normal_axis": 1,
        "plane": y_min,
        "tangent_axis": 0,
        "tangent_min": x_min,
        "tangent_max": x_max,
    },
    "y_max": {
        "normal_axis": 1,
        "plane": y_max,
        "tangent_axis": 0,
        "tangent_min": x_min,
        "tangent_max": x_max,
    },
}


def contiguous_runs(mask: np.ndarray):
    runs = []
    start = None

    for index, value in enumerate(mask):
        if value and start is None:
            start = index

        if start is not None and (
            not value or index == len(mask) - 1
        ):
            end = index if not value else index + 1
            runs.append((start, end))
            start = None

    return runs


def mean_band_profile(
    occupancy: np.ndarray,
    z_centers: np.ndarray,
    minimum_height: float,
    maximum_height: float,
):
    mask = (
        (z_centers >= floor_z + minimum_height)
        & (z_centers < floor_z + maximum_height)
    )

    if not mask.any():
        return np.zeros(occupancy.shape[1])

    return occupancy[mask].mean(axis=0)


def classify_gap(low, middle, high):
    low_mean = float(np.mean(low))
    middle_mean = float(np.mean(middle))
    high_mean = float(np.mean(high))

    if (
        low_mean < 0.10
        and middle_mean < 0.10
        and high_mean >= 0.08
    ):
        label = "door_like"

    elif (
        low_mean >= 0.08
        and middle_mean < 0.10
        and high_mean >= 0.08
    ):
        label = "window_like"

    else:
        label = "unobserved_or_occluded"

    return label, low_mean, middle_mean, high_mean


def analyze_wall(name: str, definition: dict):
    normal_coordinate = points_xy_m[:, definition["normal_axis"]]
    tangent_coordinate = points_xy_m[:, definition["tangent_axis"]]

    wall_mask = (
        (np.abs(normal_coordinate - definition["plane"])
         <= WALL_DISTANCE_M)
        & (tangent_coordinate >= definition["tangent_min"])
        & (tangent_coordinate <= definition["tangent_max"])
        & (points[:, 2] >= floor_z)
        & (points[:, 2] <= ceiling_z)
    )

    wall_tangent = tangent_coordinate[wall_mask]
    wall_height = points[wall_mask, 2]

    tangent_edges = np.arange(
        definition["tangent_min"],
        definition["tangent_max"] + TANGENT_RESOLUTION_M,
        TANGENT_RESOLUTION_M,
    )

    z_edges = np.arange(
        floor_z,
        ceiling_z + HEIGHT_RESOLUTION_M,
        HEIGHT_RESOLUTION_M,
    )

    histogram, z_edges, tangent_edges = np.histogram2d(
        wall_height,
        wall_tangent,
        bins=[z_edges, tangent_edges],
    )

    raw_occupancy = histogram >= MIN_POINTS_PER_CELL

    occupancy = binary_closing(
        raw_occupancy,
        structure=np.ones((2, 2), dtype=bool),
    )

    tangent_centers = 0.5 * (
        tangent_edges[:-1] + tangent_edges[1:]
    )

    z_centers = 0.5 * (
        z_edges[:-1] + z_edges[1:]
    )

    low_profile = mean_band_profile(
        occupancy,
        z_centers,
        0.10,
        0.90,
    )

    middle_profile = mean_band_profile(
        occupancy,
        z_centers,
        0.90,
        2.05,
    )

    high_profile = mean_band_profile(
        occupancy,
        z_centers,
        2.05,
        room_height - 0.05,
    )

    combined_profile = occupancy.mean(axis=0)

    gap_mask = (
        combined_profile < LOW_COVERAGE_THRESHOLD
    )

    edge_valid = (
        (tangent_centers
         >= definition["tangent_min"] + EDGE_MARGIN_M)
        & (tangent_centers
           <= definition["tangent_max"] - EDGE_MARGIN_M)
    )

    gap_mask &= edge_valid

    candidates = []

    for start_index, end_index in contiguous_runs(gap_mask):
        start = float(tangent_edges[start_index])
        end = float(tangent_edges[end_index])
        width = end - start

        if not (
            MIN_GAP_WIDTH_M <= width <= MAX_GAP_WIDTH_M
        ):
            continue

        label, low_mean, middle_mean, high_mean = classify_gap(
            low_profile[start_index:end_index],
            middle_profile[start_index:end_index],
            high_profile[start_index:end_index],
        )

        candidates.append(
            {
                "start_m": start,
                "end_m": end,
                "width_m": width,
                "classification": label,
                "low_band_coverage": low_mean,
                "middle_band_coverage": middle_mean,
                "high_band_coverage": high_mean,
            }
        )

    coverage_ratio = float(occupancy.mean())

    # Occupancy heatmap.
    fig, ax = plt.subplots(figsize=(12, 6), dpi=170)

    ax.imshow(
        occupancy.astype(np.float32),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[
            tangent_edges[0],
            tangent_edges[-1],
            z_edges[0],
            z_edges[-1],
        ],
        vmin=0,
        vmax=1,
    )

    for candidate in candidates:
        ax.add_patch(
            Rectangle(
                (
                    candidate["start_m"],
                    floor_z,
                ),
                candidate["width_m"],
                room_height,
                fill=False,
                linewidth=2,
            )
        )

        ax.text(
            0.5 * (
                candidate["start_m"]
                + candidate["end_m"]
            ),
            floor_z + 0.10,
            candidate["classification"],
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Position along wall (m)")
    ax.set_ylabel("Height Z (m)")
    ax.set_title(
        f"{name}: wall occupancy, coverage={coverage_ratio:.3f}"
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_occupancy.png")
    plt.close(fig)

    # Coverage profiles.
    fig, ax = plt.subplots(figsize=(12, 5), dpi=170)

    ax.plot(tangent_centers, low_profile, label="Low: 0.1–0.9 m")
    ax.plot(tangent_centers, middle_profile, label="Middle: 0.9–2.05 m")
    ax.plot(tangent_centers, high_profile, label="High: above 2.05 m")
    ax.plot(
        tangent_centers,
        combined_profile,
        linewidth=2,
        label="Full-height coverage",
    )

    ax.axhline(
        LOW_COVERAGE_THRESHOLD,
        linestyle="--",
        linewidth=1.5,
        label="Gap threshold",
    )

    for candidate in candidates:
        ax.axvspan(
            candidate["start_m"],
            candidate["end_m"],
            alpha=0.15,
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Position along wall (m)")
    ax.set_ylabel("Occupied-height fraction")
    ax.set_title(f"{name}: wall coverage profiles")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_profiles.png")
    plt.close(fig)

    return {
        "plane_coordinate_m": float(definition["plane"]),
        "support_points": int(wall_mask.sum()),
        "wall_length_m": float(
            definition["tangent_max"]
            - definition["tangent_min"]
        ),
        "coverage_ratio": coverage_ratio,
        "candidate_gaps": candidates,
    }


results = {}

for wall_name, wall_definition in WALLS.items():
    print("Analyzing:", wall_name)
    results[wall_name] = analyze_wall(
        wall_name,
        wall_definition,
    )


# Floor-plan visualization of all candidate gaps.
rng = np.random.default_rng(42)
plot_count = min(100_000, len(points_xy_m))
plot_indices = rng.choice(
    len(points_xy_m),
    plot_count,
    replace=False,
)

fig, ax = plt.subplots(figsize=(10, 9), dpi=180)

ax.scatter(
    points_xy_m[plot_indices, 0],
    points_xy_m[plot_indices, 1],
    s=0.18,
    linewidths=0,
    alpha=0.25,
)

rectangle = np.array(
    [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ]
)

ax.plot(
    rectangle[:, 0],
    rectangle[:, 1],
    linewidth=2.5,
    label="Refined room envelope",
)

for wall_name, wall_result in results.items():
    for candidate in wall_result["candidate_gaps"]:
        start = candidate["start_m"]
        end = candidate["end_m"]

        if wall_name == "x_min":
            ax.plot([x_min, x_min], [start, end], linewidth=6)

        elif wall_name == "x_max":
            ax.plot([x_max, x_max], [start, end], linewidth=6)

        elif wall_name == "y_min":
            ax.plot([start, end], [y_min, y_min], linewidth=6)

        elif wall_name == "y_max":
            ax.plot([start, end], [y_max, y_max], linewidth=6)

ax.set_xlabel("Manhattan X (m)")
ax.set_ylabel("Manhattan Y (m)")
ax.set_title("Wall-coverage gap candidates")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "opening_candidates_floorplan.png")
plt.close(fig)


summary = {
    "method": (
        "Per-wall metric occupancy grids with low/middle/high "
        "height-band coverage. Gaps are hypotheses, not confirmed openings."
    ),
    "parameters": {
        "wall_distance_m": WALL_DISTANCE_M,
        "tangent_resolution_m": TANGENT_RESOLUTION_M,
        "height_resolution_m": HEIGHT_RESOLUTION_M,
        "minimum_points_per_cell": MIN_POINTS_PER_CELL,
        "minimum_gap_width_m": MIN_GAP_WIDTH_M,
        "maximum_gap_width_m": MAX_GAP_WIDTH_M,
        "low_coverage_threshold": LOW_COVERAGE_THRESHOLD,
    },
    "walls": results,
}

summary_path = OUTPUT_DIR / "wall_opening_summary.json"

with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print()
print(json.dumps(summary, indent=2))
print()
print("Saved:", summary_path)
print("WALL_OPENING_ANALYSIS_OK")
