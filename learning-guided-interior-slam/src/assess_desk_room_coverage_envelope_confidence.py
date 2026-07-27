from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


ROOT = Path(
    "/workspace/interior-slam/results/"
    "tum_fr1_desk_baseline"
)

MESH_PATH = (
    ROOT
    / "tsdf_ablation/estimated_rgbd_scale_mesh.ply"
)

OUTPUT_DIR = ROOT / "structure_coverage_envelope_confidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "desk_structure_coverage.json"
OUTPUT_TOPDOWN = OUTPUT_DIR / "desk_structure_coverage.png"
OUTPUT_HISTOGRAM = OUTPUT_DIR / "desk_plane_histograms.png"


# ------------------------------------------------------------
# Conservative thresholds
# ------------------------------------------------------------

HORIZONTAL_NORMAL_Z = 0.85
VERTICAL_NORMAL_Z = 0.35

FLOOR_Z_TOLERANCE_M = 0.05
PLANE_TOLERANCE_M = 0.08

COORDINATE_BIN_M = 0.05
TANGENT_BIN_M = 0.15
HEIGHT_BIN_M = 0.15

MIN_FLOOR_SUPPORT_POINTS = 300
MIN_FLOOR_AREA_M2 = 1.0

MIN_CEILING_SUPPORT_POINTS = 40
MIN_CEILING_AREA_M2 = 0.20
MIN_ROOM_HEIGHT_M = 2.0
MAX_ROOM_HEIGHT_M = 3.5

MIN_PERSISTENT_PLANE_SUPPORT = 250
MIN_PERSISTENT_VERTICAL_SPAN_M = 1.50
MIN_PERSISTENT_TANGENT_SPAN_M = 1.20
MIN_PERSISTENT_CELLS = 25

MIN_OPPOSING_PLANE_SEPARATION_M = 2.0

MIN_WALL_SUPPORT_POINTS = 400
MIN_WALL_VERTICAL_SPAN_M = 1.60
MIN_WALL_TANGENT_COVERAGE = 0.35
MIN_WALL_VERTICAL_COVERAGE = 0.35
MIN_WALL_RELATIVE_TANGENT_SPAN = 0.50

MIN_FLOOR_ENVELOPE_COVERAGE = 0.15


def occupied_area_xy(
    points: np.ndarray,
    resolution: float = 0.10,
) -> float:
    if len(points) == 0:
        return 0.0

    cells = np.floor(
        points[:, :2] / resolution
    ).astype(np.int64)

    return float(
        len(np.unique(cells, axis=0))
        * resolution
        * resolution
    )


def robust_span(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0

    low, high = np.percentile(
        values,
        [5.0, 95.0],
    )

    return float(high - low)


def padded_peaks(
    histogram: np.ndarray,
    minimum_prominence: float,
    minimum_distance_bins: int,
):
    padded = np.pad(
        histogram,
        (1, 1),
        mode="constant",
        constant_values=0.0,
    )

    peaks, properties = find_peaks(
        padded,
        prominence=minimum_prominence,
        distance=minimum_distance_bins,
    )

    peaks = peaks - 1

    valid = (
        (peaks >= 0)
        & (peaks < len(histogram))
    )

    return peaks[valid], {
        key: np.asarray(value)[valid]
        for key, value in properties.items()
    }


mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))

if mesh.is_empty():
    raise RuntimeError(f"Empty mesh: {MESH_PATH}")

mesh.compute_vertex_normals()

points = np.asarray(
    mesh.vertices,
    dtype=np.float64,
)

normals = np.asarray(
    mesh.vertex_normals,
    dtype=np.float64,
)

finite = (
    np.isfinite(points).all(axis=1)
    & np.isfinite(normals).all(axis=1)
)

points = points[finite]
normals = normals[finite]

normal_lengths = np.linalg.norm(
    normals,
    axis=1,
)

valid_normals = normal_lengths > 1e-8
points = points[valid_normals]
normals = normals[valid_normals]
normals /= normal_lengths[valid_normals, None]

print("Mesh vertices:", len(points))


# ------------------------------------------------------------
# Horizontal-plane candidates: floor and ceiling
# ------------------------------------------------------------

horizontal_mask = (
    np.abs(normals[:, 2])
    >= HORIZONTAL_NORMAL_Z
)

horizontal_points = points[horizontal_mask]

if len(horizontal_points) < 100:
    raise RuntimeError(
        "Insufficient horizontal surface evidence."
    )

z_min = float(np.percentile(horizontal_points[:, 2], 0.5))
z_max = float(np.percentile(horizontal_points[:, 2], 99.5))

z_edges = np.arange(
    z_min,
    z_max + 0.03,
    0.03,
)

z_histogram, z_edges = np.histogram(
    horizontal_points[:, 2],
    bins=z_edges,
)

z_smooth = gaussian_filter1d(
    z_histogram.astype(np.float64),
    sigma=1.0,
)

z_peaks, _ = padded_peaks(
    z_smooth,
    minimum_prominence=max(
        5.0,
        0.02 * float(z_smooth.max()),
    ),
    minimum_distance_bins=3,
)

horizontal_candidates = []

for peak in z_peaks:
    z_center = float(
        0.5 * (
            z_edges[peak]
            + z_edges[peak + 1]
        )
    )

    support_mask = (
        np.abs(
            horizontal_points[:, 2]
            - z_center
        )
        <= FLOOR_Z_TOLERANCE_M
    )

    support_points = horizontal_points[
        support_mask
    ]

    if len(support_points) == 0:
        continue

    horizontal_candidates.append(
        {
            "z_m": float(
                np.median(
                    support_points[:, 2]
                )
            ),
            "support_points": int(
                len(support_points)
            ),
            "support_area_m2": occupied_area_xy(
                support_points
            ),
        }
    )

horizontal_candidates.sort(
    key=lambda candidate: candidate["z_m"]
)

floor_candidates = [
    candidate
    for candidate in horizontal_candidates
    if (
        candidate["support_points"]
        >= MIN_FLOOR_SUPPORT_POINTS
        and candidate["support_area_m2"]
        >= MIN_FLOOR_AREA_M2
    )
]

floor = (
    floor_candidates[0]
    if floor_candidates
    else None
)

ceiling = None

if floor is not None:
    valid_ceiling_candidates = [
        candidate
        for candidate in horizontal_candidates
        if (
            MIN_ROOM_HEIGHT_M
            <= candidate["z_m"] - floor["z_m"]
            <= MAX_ROOM_HEIGHT_M
            and candidate["support_points"]
            >= MIN_CEILING_SUPPORT_POINTS
            and candidate["support_area_m2"]
            >= MIN_CEILING_AREA_M2
        )
    ]

    if valid_ceiling_candidates:
        ceiling = max(
            valid_ceiling_candidates,
            key=lambda candidate: candidate["z_m"],
        )

if floor is None:
    raise RuntimeError(
        "No reliable floor candidate was found."
    )

floor_z = float(floor["z_m"])

analysis_top_z = (
    float(ceiling["z_m"])
    if ceiling is not None
    else floor_z + 2.50
)

print("Floor z:", floor_z)
print("Floor support area:", floor["support_area_m2"])
print("Ceiling found:", ceiling is not None)


# ------------------------------------------------------------
# Vertical evidence and Manhattan orientation
# ------------------------------------------------------------

vertical_mask = (
    (np.abs(normals[:, 2]) <= VERTICAL_NORMAL_Z)
    & (points[:, 2] >= floor_z + 0.10)
    & (points[:, 2] <= analysis_top_z + 0.20)
)

vertical_points = points[vertical_mask]
vertical_normals = normals[vertical_mask]

if len(vertical_points) < 500:
    raise RuntimeError(
        "Insufficient vertical surface evidence."
    )

normal_xy = vertical_normals[:, :2]
normal_xy_lengths = np.linalg.norm(
    normal_xy,
    axis=1,
)

valid_xy_normals = normal_xy_lengths > 1e-8
vertical_points = vertical_points[valid_xy_normals]
normal_xy = normal_xy[valid_xy_normals]
normal_xy /= normal_xy_lengths[valid_xy_normals, None]

normal_angles = np.arctan2(
    normal_xy[:, 1],
    normal_xy[:, 0],
)

fourth_cosine = float(
    np.mean(np.cos(4.0 * normal_angles))
)

fourth_sine = float(
    np.mean(np.sin(4.0 * normal_angles))
)

manhattan_concentration = float(
    math.hypot(
        fourth_cosine,
        fourth_sine,
    )
)

manhattan_angle = 0.25 * math.atan2(
    fourth_sine,
    fourth_cosine,
)

rotation_to_manhattan = np.array(
    [
        [
            math.cos(manhattan_angle),
            math.sin(manhattan_angle),
        ],
        [
            -math.sin(manhattan_angle),
            math.cos(manhattan_angle),
        ],
    ],
    dtype=np.float64,
)

points_m = vertical_points.copy()
points_m[:, :2] = (
    vertical_points[:, :2]
    @ rotation_to_manhattan.T
)

normals_m_xy = (
    normal_xy
    @ rotation_to_manhattan.T
)

print(
    "Manhattan angle:",
    math.degrees(manhattan_angle),
    "degrees",
)

print(
    "Manhattan concentration:",
    manhattan_concentration,
)


# ------------------------------------------------------------
# Persistent vertical-plane extraction
# ------------------------------------------------------------

def extract_axis_planes(axis: int):
    tangent_axis = 1 - axis

    orientation_mask = (
        np.abs(normals_m_xy[:, axis])
        >= math.cos(math.radians(25.0))
    )

    axis_points = points_m[orientation_mask]

    if len(axis_points) < 100:
        return [], axis_points, None

    coordinates = axis_points[:, axis]

    coordinate_min = float(
        np.percentile(coordinates, 0.5)
    )

    coordinate_max = float(
        np.percentile(coordinates, 99.5)
    )

    edges = np.arange(
        coordinate_min,
        coordinate_max + COORDINATE_BIN_M,
        COORDINATE_BIN_M,
    )

    histogram, edges = np.histogram(
        coordinates,
        bins=edges,
    )

    smooth = gaussian_filter1d(
        histogram.astype(np.float64),
        sigma=1.2,
    )

    peaks, _ = padded_peaks(
        smooth,
        minimum_prominence=max(
            10.0,
            0.025 * float(smooth.max()),
        ),
        minimum_distance_bins=max(
            1,
            int(0.20 / COORDINATE_BIN_M),
        ),
    )

    candidates = []

    for peak in peaks:
        coordinate = float(
            0.5 * (
                edges[peak]
                + edges[peak + 1]
            )
        )

        support_mask = (
            np.abs(
                axis_points[:, axis]
                - coordinate
            )
            <= PLANE_TOLERANCE_M
        )

        support = axis_points[support_mask]

        if len(support) == 0:
            continue

        tangent_values = support[:, tangent_axis]
        z_values = support[:, 2]

        tangent_span = robust_span(
            tangent_values
        )

        vertical_span = robust_span(z_values)

        residuals = (
            support[:, axis] - coordinate
        )

        cells = np.floor(
            np.column_stack(
                [
                    tangent_values
                    / TANGENT_BIN_M,
                    z_values
                    / HEIGHT_BIN_M,
                ]
            )
        ).astype(np.int64)

        occupied_cells = int(
            len(np.unique(cells, axis=0))
        )

        persistent = bool(
            len(support)
            >= MIN_PERSISTENT_PLANE_SUPPORT
            and vertical_span
            >= MIN_PERSISTENT_VERTICAL_SPAN_M
            and tangent_span
            >= MIN_PERSISTENT_TANGENT_SPAN_M
            and occupied_cells
            >= MIN_PERSISTENT_CELLS
        )

        candidates.append(
            {
                "coordinate_m": coordinate,
                "support_points": int(
                    len(support)
                ),
                "vertical_span_m": vertical_span,
                "tangent_span_m": tangent_span,
                "occupied_cells": occupied_cells,
                "residual_rmse_m": float(
                    np.sqrt(
                        np.mean(residuals ** 2)
                    )
                ),
                "persistent": persistent,
            }
        )

    candidates.sort(
        key=lambda candidate:
        candidate["coordinate_m"]
    )

    return candidates, axis_points, {
        "edges": edges,
        "histogram": histogram,
        "smooth": smooth,
    }


x_candidates, x_axis_points, x_hist = (
    extract_axis_planes(axis=0)
)

y_candidates, y_axis_points, y_hist = (
    extract_axis_planes(axis=1)
)


def select_outer_pair(candidates):
    persistent = [
        candidate
        for candidate in candidates
        if candidate["persistent"]
    ]

    if len(persistent) < 2:
        return None

    lower = persistent[0]
    upper = persistent[-1]

    if (
        upper["coordinate_m"]
        - lower["coordinate_m"]
        < MIN_OPPOSING_PLANE_SEPARATION_M
    ):
        return None

    return lower, upper


x_pair = select_outer_pair(x_candidates)
y_pair = select_outer_pair(y_candidates)


# ------------------------------------------------------------
# Selected wall coverage
# ------------------------------------------------------------

def score_wall(
    axis_points: np.ndarray,
    axis: int,
    coordinate: float,
    tangent_min: float,
    tangent_max: float,
):
    tangent_axis = 1 - axis

    support_mask = (
        np.abs(
            axis_points[:, axis]
            - coordinate
        )
        <= PLANE_TOLERANCE_M
    )

    support = axis_points[support_mask]

    if len(support) == 0:
        return {
            "support_points": 0,
            "tangent_span_m": 0.0,
            "vertical_span_m": 0.0,
            "tangent_coverage": 0.0,
            "vertical_coverage": 0.0,
            "passes": False,
        }

    tangent = support[:, tangent_axis]
    height = support[:, 2]

    inside = (
        (tangent >= tangent_min)
        & (tangent <= tangent_max)
        & (height >= floor_z + 0.10)
        & (height <= analysis_top_z)
    )

    support = support[inside]
    tangent = support[:, tangent_axis]
    height = support[:, 2]

    side_length = tangent_max - tangent_min

    tangent_edges = np.arange(
        tangent_min,
        tangent_max + TANGENT_BIN_M,
        TANGENT_BIN_M,
    )

    tangent_observed = []

    for index in range(len(tangent_edges) - 1):
        mask = (
            (tangent >= tangent_edges[index])
            & (tangent < tangent_edges[index + 1])
        )

        bin_heights = height[mask]

        tangent_observed.append(
            len(bin_heights) >= 15
            and robust_span(bin_heights) >= 0.80
        )

    tangent_coverage = float(
        np.mean(tangent_observed)
        if tangent_observed
        else 0.0
    )

    height_edges = np.arange(
        floor_z + 0.10,
        analysis_top_z + HEIGHT_BIN_M,
        HEIGHT_BIN_M,
    )

    height_observed = []

    for index in range(len(height_edges) - 1):
        mask = (
            (height >= height_edges[index])
            & (height < height_edges[index + 1])
        )

        bin_tangent = tangent[mask]

        height_observed.append(
            len(bin_tangent) >= 15
            and robust_span(bin_tangent) >= 0.80
        )

    vertical_coverage = float(
        np.mean(height_observed)
        if height_observed
        else 0.0
    )

    tangent_span = robust_span(tangent)
    vertical_span = robust_span(height)

    passes = bool(
        len(support) >= MIN_WALL_SUPPORT_POINTS
        and vertical_span
        >= MIN_WALL_VERTICAL_SPAN_M
        and tangent_coverage
        >= MIN_WALL_TANGENT_COVERAGE
        and vertical_coverage
        >= MIN_WALL_VERTICAL_COVERAGE
        and tangent_span
        >= (
            MIN_WALL_RELATIVE_TANGENT_SPAN
            * side_length
        )
    )

    return {
        "support_points": int(len(support)),
        "tangent_span_m": tangent_span,
        "vertical_span_m": vertical_span,
        "tangent_coverage": tangent_coverage,
        "vertical_coverage": vertical_coverage,
        "passes": passes,
    }


wall_scores = {}
envelope = None

if x_pair is not None and y_pair is not None:
    x_min = float(x_pair[0]["coordinate_m"])
    x_max = float(x_pair[1]["coordinate_m"])
    y_min = float(y_pair[0]["coordinate_m"])
    y_max = float(y_pair[1]["coordinate_m"])

    wall_scores = {
        "x_min": score_wall(
            x_axis_points,
            axis=0,
            coordinate=x_min,
            tangent_min=y_min,
            tangent_max=y_max,
        ),
        "x_max": score_wall(
            x_axis_points,
            axis=0,
            coordinate=x_max,
            tangent_min=y_min,
            tangent_max=y_max,
        ),
        "y_min": score_wall(
            y_axis_points,
            axis=1,
            coordinate=y_min,
            tangent_min=x_min,
            tangent_max=x_max,
        ),
        "y_max": score_wall(
            y_axis_points,
            axis=1,
            coordinate=y_max,
            tangent_min=x_min,
            tangent_max=x_max,
        ),
    }

    width = x_max - x_min
    length = y_max - y_min
    envelope_area = width * length

    floor_envelope_coverage = float(
        floor["support_area_m2"]
        / max(envelope_area, 1e-12)
    )

    envelope = {
        "x_min_m": x_min,
        "x_max_m": x_max,
        "y_min_m": y_min,
        "y_max_m": y_max,
        "width_m": width,
        "length_m": length,
        "area_m2": envelope_area,
        "floor_support_fraction": (
            floor_envelope_coverage
        ),
    }
else:
    floor_envelope_coverage = 0.0


# ------------------------------------------------------------
# Envelope-level confidence and abstention decision
# ------------------------------------------------------------

# A room can remain structurally supported when one perimeter
# wall is strongly occluded or only partially traversed.
#
# We therefore require:
#   - both opposing plane pairs,
#   - sufficient floor support,
#   - at least three strongly supported walls,
#   - every remaining wall to be at least weakly supported.
#
# This changes only the acceptance policy. It does not relax
# plane extraction or alter the estimated envelope geometry.

MIN_ENVELOPE_MANHATTAN_CONCENTRATION = 0.20
MIN_STRONG_WALL_COUNT = 3

MIN_WEAK_WALL_SUPPORT_POINTS = 1000
MIN_WEAK_WALL_VERTICAL_SPAN_M = 1.50
MIN_WEAK_WALL_TANGENT_COVERAGE = 0.15
MIN_WEAK_WALL_VERTICAL_COVERAGE = 0.50
MIN_WEAK_WALL_RELATIVE_TANGENT_SPAN = 0.30


def weak_wall_supported(
    wall_name: str,
    score: dict,
) -> bool:
    if envelope is None:
        return False

    if wall_name.startswith("x_"):
        side_length = envelope["length_m"]
    else:
        side_length = envelope["width_m"]

    return bool(
        score["support_points"]
        >= MIN_WEAK_WALL_SUPPORT_POINTS
        and score["vertical_span_m"]
        >= MIN_WEAK_WALL_VERTICAL_SPAN_M
        and score["tangent_coverage"]
        >= MIN_WEAK_WALL_TANGENT_COVERAGE
        and score["vertical_coverage"]
        >= MIN_WEAK_WALL_VERTICAL_COVERAGE
        and score["tangent_span_m"]
        >= (
            MIN_WEAK_WALL_RELATIVE_TANGENT_SPAN
            * side_length
        )
    )


reasons = []
decision_notes = []

wall_support_class = {}
strong_wall_names = []
weak_wall_names = []
unsupported_wall_names = []

if x_pair is None:
    reasons.append(
        "No reliable opposing persistent X-wall pair."
    )

if y_pair is None:
    reasons.append(
        "No reliable opposing persistent Y-wall pair."
    )

if ceiling is None:
    reasons.append(
        "No reliable ceiling plane."
    )

if envelope is not None:
    if (
        manhattan_concentration
        < MIN_ENVELOPE_MANHATTAN_CONCENTRATION
    ):
        reasons.append(
            "Manhattan orientation evidence is too weak."
        )

    if (
        floor_envelope_coverage
        < MIN_FLOOR_ENVELOPE_COVERAGE
    ):
        reasons.append(
            "Floor evidence covers too little of the "
            "candidate envelope."
        )

    for wall_name, score in wall_scores.items():
        if score["passes"]:
            wall_support_class[wall_name] = "strong"
            strong_wall_names.append(wall_name)

        elif weak_wall_supported(wall_name, score):
            wall_support_class[wall_name] = (
                "weak_but_supported"
            )
            weak_wall_names.append(wall_name)

        else:
            wall_support_class[wall_name] = "unsupported"
            unsupported_wall_names.append(wall_name)

    if len(strong_wall_names) < MIN_STRONG_WALL_COUNT:
        reasons.append(
            "Fewer than three perimeter walls have "
            "strong structural coverage."
        )

    if unsupported_wall_names:
        reasons.append(
            "Unsupported perimeter walls: "
            + ", ".join(unsupported_wall_names)
            + "."
        )

    if weak_wall_names:
        decision_notes.append(
            "Envelope accepted with weak-but-supported "
            "wall evidence: "
            + ", ".join(weak_wall_names)
            + "."
        )


strong_wall_count = len(strong_wall_names)

enclosure_supported = bool(
    envelope is not None
    and x_pair is not None
    and y_pair is not None
    and manhattan_concentration
    >= MIN_ENVELOPE_MANHATTAN_CONCENTRATION
    and floor_envelope_coverage
    >= MIN_FLOOR_ENVELOPE_COVERAGE
    and strong_wall_count
    >= MIN_STRONG_WALL_COUNT
    and not unsupported_wall_names
)

if enclosure_supported and ceiling is not None:
    decision = "FULL_3D_ROOM_ENVELOPE_SUPPORTED"

elif enclosure_supported:
    decision = "PARTIAL_2D_ENVELOPE_ONLY"

else:
    decision = "ABSTAIN_INSUFFICIENT_ROOM_COVERAGE"


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 9),
    dpi=180,
)

rng = np.random.default_rng(42)

plot_count = min(
    100_000,
    len(points_m),
)

plot_indices = rng.choice(
    len(points_m),
    size=plot_count,
    replace=False,
)

ax.scatter(
    points_m[plot_indices, 0],
    points_m[plot_indices, 1],
    s=0.2,
    linewidths=0,
    alpha=0.20,
    label="Vertical TSDF evidence",
)

for candidate in x_candidates:
    if not candidate["persistent"]:
        continue

    ax.axvline(
        candidate["coordinate_m"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

for candidate in y_candidates:
    if not candidate["persistent"]:
        continue

    ax.axhline(
        candidate["coordinate_m"],
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
    )

if envelope is not None:
    rectangle = np.array(
        [
            [
                envelope["x_min_m"],
                envelope["y_min_m"],
            ],
            [
                envelope["x_max_m"],
                envelope["y_min_m"],
            ],
            [
                envelope["x_max_m"],
                envelope["y_max_m"],
            ],
            [
                envelope["x_min_m"],
                envelope["y_max_m"],
            ],
            [
                envelope["x_min_m"],
                envelope["y_min_m"],
            ],
        ]
    )

    ax.plot(
        rectangle[:, 0],
        rectangle[:, 1],
        linewidth=3,
        label="Candidate envelope",
    )

ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Manhattan X (m)")
ax.set_ylabel("Manhattan Y (m)")
ax.set_title(
    "Desk structural coverage\n"
    + decision
)
ax.grid(alpha=0.15)
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_TOPDOWN)
plt.close(fig)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    dpi=180,
)

for axis_plot, histogram_data, candidates, name in (
    (
        axes[0],
        x_hist,
        x_candidates,
        "X-oriented planes",
    ),
    (
        axes[1],
        y_hist,
        y_candidates,
        "Y-oriented planes",
    ),
):
    if histogram_data is not None:
        centers = 0.5 * (
            histogram_data["edges"][:-1]
            + histogram_data["edges"][1:]
        )

        axis_plot.plot(
            centers,
            histogram_data["smooth"],
        )

        for candidate in candidates:
            axis_plot.axvline(
                candidate["coordinate_m"],
                linestyle=(
                    "-"
                    if candidate["persistent"]
                    else ":"
                ),
                alpha=0.7,
            )

    axis_plot.set_title(name)
    axis_plot.set_xlabel("Plane coordinate (m)")
    axis_plot.set_ylabel("Smoothed support")

fig.tight_layout()
fig.savefig(OUTPUT_HISTOGRAM)
plt.close(fig)


summary = {
    "dataset": "rgbd_dataset_freiburg1_desk",
    "input_mesh": str(MESH_PATH),
    "decision": decision,
    "reasons": reasons,
    "notes": decision_notes,
    "manhattan": {
        "angle_rad": manhattan_angle,
        "angle_deg": math.degrees(
            manhattan_angle
        ),
        "concentration": (
            manhattan_concentration
        ),
    },
    "floor": floor,
    "ceiling": ceiling,
    "horizontal_plane_candidates": (
        horizontal_candidates
    ),
    "x_plane_candidates": x_candidates,
    "y_plane_candidates": y_candidates,
    "candidate_envelope": envelope,
    "wall_scores": wall_scores,
    "thresholds": {
        "minimum_opposing_plane_separation_m": (
            MIN_OPPOSING_PLANE_SEPARATION_M
        ),
        "minimum_wall_tangent_coverage": (
            MIN_WALL_TANGENT_COVERAGE
        ),
        "minimum_wall_vertical_coverage": (
            MIN_WALL_VERTICAL_COVERAGE
        ),
        "minimum_floor_envelope_coverage": (
            MIN_FLOOR_ENVELOPE_COVERAGE
        ),
    },
    "decision_policy": {
        "minimum_manhattan_concentration": (
            MIN_ENVELOPE_MANHATTAN_CONCENTRATION
        ),
        "minimum_strong_wall_count": (
            MIN_STRONG_WALL_COUNT
        ),
        "minimum_weak_wall_support_points": (
            MIN_WEAK_WALL_SUPPORT_POINTS
        ),
        "minimum_weak_wall_vertical_span_m": (
            MIN_WEAK_WALL_VERTICAL_SPAN_M
        ),
        "minimum_weak_wall_tangent_coverage": (
            MIN_WEAK_WALL_TANGENT_COVERAGE
        ),
        "minimum_weak_wall_vertical_coverage": (
            MIN_WEAK_WALL_VERTICAL_COVERAGE
        ),
        "minimum_weak_wall_relative_tangent_span": (
            MIN_WEAK_WALL_RELATIVE_TANGENT_SPAN
        ),
        "strong_walls": strong_wall_names,
        "weak_but_supported_walls": weak_wall_names,
        "unsupported_walls": unsupported_wall_names,
        "wall_support_class": wall_support_class,
    },
    "outputs": {
        "topdown": str(OUTPUT_TOPDOWN),
        "histograms": str(OUTPUT_HISTOGRAM),
    },
}

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, indent=2)


print()
print("========== STRUCTURAL COVERAGE ==========")
print("Decision:", decision)
print(
    "Persistent X planes:",
    sum(
        candidate["persistent"]
        for candidate in x_candidates
    ),
)
print(
    "Persistent Y planes:",
    sum(
        candidate["persistent"]
        for candidate in y_candidates
    ),
)
print("Ceiling found:", ceiling is not None)

if envelope is not None:
    print(
        "Candidate dimensions:",
        round(envelope["width_m"], 3),
        "x",
        round(envelope["length_m"], 3),
        "m",
    )

    print(
        "Floor support fraction:",
        round(
            envelope["floor_support_fraction"],
            3,
        ),
    )

for wall_name, score in wall_scores.items():
    print(
        wall_name,
        "support=", score["support_points"],
        "tangent coverage=",
        round(score["tangent_coverage"], 3),
        "vertical coverage=",
        round(score["vertical_coverage"], 3),
        "passes=", score["passes"],
    )

print("Strong walls:", strong_wall_names)
print(
    "Weak-but-supported walls:",
    weak_wall_names,
)
print(
    "Unsupported walls:",
    unsupported_wall_names,
)

print("Reasons:")

for reason in reasons:
    print(" -", reason)

print()
print("Saved:", OUTPUT_JSON)
print("Saved:", OUTPUT_TOPDOWN)
print("Saved:", OUTPUT_HISTOGRAM)
print("STRUCTURAL_COVERAGE_ASSESSMENT_OK")
