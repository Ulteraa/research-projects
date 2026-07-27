from __future__ import annotations

import bisect
import json
import math
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import open3d as o3d
from scipy.ndimage import (
    binary_closing,
    binary_opening,
    label,
)
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path("/workspace/interior-slam")
REPO = PROJECT_ROOT / "third_party/MASt3R-SLAM"

DATASET = (
    REPO / "datasets/tum/rgbd_dataset_freiburg1_room"
)

ESTIMATED_TRAJECTORY = (
    REPO / "logs/tum_fr1_room/"
    "rgbd_dataset_freiburg1_room.txt"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results/tum_fr1_room_baseline"
)

TSDF_SUMMARY = (
    RESULT_ROOT / "tsdf/tsdf_summary.json"
)

STRUCTURE_ROOT = (
    RESULT_ROOT
    / "tsdf/final/structure_baseline"
)

MANHATTAN_SUMMARY = (
    STRUCTURE_ROOT
    / "manhattan_walls/manhattan_wall_summary.json"
)

REFINED_SUMMARY = (
    STRUCTURE_ROOT
    / "refined_envelope/refined_envelope_summary.json"
)

WALL_CANDIDATES = (
    STRUCTURE_ROOT / "wall_candidates.ply"
)

OUTPUT_DIR = (
    STRUCTURE_ROOT / "visibility_openings"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# TUM Freiburg1 registered RGB-D intrinsics.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FX = 525.0
FY = 525.0
CX = 319.5
CY = 239.5
DEPTH_SCALE = 5000.0
MAX_DEPTH_M = 5.0

# Timestamp association thresholds.
MAX_RGB_DIFFERENCE_S = 0.020
MAX_DEPTH_DIFFERENCE_S = 0.030

# Wall-grid resolution.
TANGENT_RESOLUTION_M = 0.05
HEIGHT_RESOLUTION_M = 0.05

# Visibility restrictions.
MIN_CAMERA_DISTANCE_M = 0.15
MIN_INCIDENCE_COSINE = math.cos(math.radians(75.0))

# Depth-consistency thresholds.
#
# These are deliberately conservative because:
# - trajectory ATE is about 6 cm,
# - wall-plane RMSE is around 3–5 cm,
# - depth measurements contain additional noise.
WALL_MATCH_TOLERANCE_M = 0.12
FREE_SPACE_MARGIN_M = 0.18
OCCLUSION_MARGIN_M = 0.18

# Evidence aggregation.
MIN_VALID_OBSERVATIONS = 2
MIN_WALL_OBSERVATIONS = 2
MIN_FREE_OBSERVATIONS = 3
MIN_OCCLUSION_OBSERVATIONS = 2
MIN_FREE_RATIO = 0.70
MIN_WALL_RATIO = 0.60

# Connected free-space regions.
MIN_COMPONENT_CELLS = 8
MIN_DOOR_WIDTH_M = 0.55
MAX_DOOR_WIDTH_M = 1.60
MIN_DOOR_HEIGHT_M = 1.60
MIN_WINDOW_WIDTH_M = 0.50
MIN_WINDOW_HEIGHT_M = 0.50


def read_file_index(
    path: Path,
) -> list[tuple[float, str]]:
    records: list[tuple[float, str]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()
            records.append((float(fields[0]), fields[1]))

    records.sort(key=lambda record: record[0])
    return records


def read_trajectory(
    path: Path,
) -> list[tuple[float, np.ndarray]]:
    records: list[tuple[float, np.ndarray]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            fields = line.split()

            if len(fields) < 8:
                continue

            timestamp = float(fields[0])

            values = np.asarray(
                [float(value) for value in fields[1:8]],
                dtype=np.float64,
            )

            records.append((timestamp, values))

    records.sort(key=lambda record: record[0])
    return records


def nearest_record(
    records,
    timestamps: list[float],
    query_time: float,
    maximum_difference: float,
):
    insertion = bisect.bisect_left(
        timestamps,
        query_time,
    )

    candidate_indices = []

    if insertion < len(records):
        candidate_indices.append(insertion)

    if insertion > 0:
        candidate_indices.append(insertion - 1)

    if not candidate_indices:
        return None

    best_index = min(
        candidate_indices,
        key=lambda index: abs(
            records[index][0] - query_time
        ),
    )

    record = records[best_index]
    difference = abs(record[0] - query_time)

    if difference > maximum_difference:
        return None

    return record, difference


def tum_pose_matrix(
    values: np.ndarray,
) -> np.ndarray:
    """
    TUM trajectory format:
        tx ty tz qx qy qz qw

    Returns:
        T_WC, camera-to-world transformation.
    """
    transform = np.eye(4, dtype=np.float64)

    transform[:3, :3] = Rotation.from_quat(
        values[3:7]
    ).as_matrix()

    transform[:3, 3] = values[:3]

    return transform


with TSDF_SUMMARY.open(
    "r",
    encoding="utf-8",
) as handle:
    tsdf_data = json.load(handle)

with MANHATTAN_SUMMARY.open(
    "r",
    encoding="utf-8",
) as handle:
    manhattan_data = json.load(handle)

with REFINED_SUMMARY.open(
    "r",
    encoding="utf-8",
) as handle:
    refined_data = json.load(handle)


estimated_to_gt = np.asarray(
    tsdf_data["estimated_to_gt_se3"],
    dtype=np.float64,
)

floor_z = float(manhattan_data["floor_z_m"])
ceiling_z = float(manhattan_data["ceiling_z_m"])
room_height = ceiling_z - floor_z

manhattan_angle = float(
    manhattan_data["manhattan"]["angle_rad"]
)

cos_angle = math.cos(manhattan_angle)
sin_angle = math.sin(manhattan_angle)

rotation_to_world = np.array(
    [
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle],
    ],
    dtype=np.float64,
)

rotation_to_manhattan = np.array(
    [
        [cos_angle, sin_angle],
        [-sin_angle, cos_angle],
    ],
    dtype=np.float64,
)

bounds = refined_data["bounds_manhattan"]

x_min = float(bounds["x_min_m"])
x_max = float(bounds["x_max_m"])
y_min = float(bounds["y_min_m"])
y_max = float(bounds["y_max_m"])


# ------------------------------------------------------------
# Associate the same estimated poses with registered RGB-D data.
# ------------------------------------------------------------

rgb_records = read_file_index(DATASET / "rgb.txt")
depth_records = read_file_index(DATASET / "depth.txt")
trajectory_records = read_trajectory(ESTIMATED_TRAJECTORY)

rgb_times = [record[0] for record in rgb_records]
depth_times = [record[0] for record in depth_records]

associations = []

for timestamp, pose_values in trajectory_records:
    rgb_match = nearest_record(
        rgb_records,
        rgb_times,
        timestamp,
        MAX_RGB_DIFFERENCE_S,
    )

    depth_match = nearest_record(
        depth_records,
        depth_times,
        timestamp,
        MAX_DEPTH_DIFFERENCE_S,
    )

    if rgb_match is None or depth_match is None:
        continue

    rgb_record, rgb_difference = rgb_match
    depth_record, depth_difference = depth_match

    estimated_pose_wc = tum_pose_matrix(pose_values)

    # The reconstructed TSDF mesh and structural envelope were
    # generated in this aligned frame.
    aligned_pose_wc = (
        estimated_to_gt @ estimated_pose_wc
    )

    associations.append(
        {
            "timestamp": float(timestamp),
            "rgb_path": DATASET / rgb_record[1],
            "depth_path": DATASET / depth_record[1],
            "rgb_difference_s": float(rgb_difference),
            "depth_difference_s": float(depth_difference),
            "pose_wc": aligned_pose_wc,
        }
    )


if len(associations) < 10:
    raise RuntimeError(
        f"Only {len(associations)} valid RGB-D associations."
    )

print("Associated RGB-D frames:", len(associations))
print(
    "Maximum RGB timestamp difference:",
    max(item["rgb_difference_s"] for item in associations),
)
print(
    "Maximum depth timestamp difference:",
    max(item["depth_difference_s"] for item in associations),
)


WALL_DEFINITIONS = {
    "x_min": {
        "normal_axis": 0,
        "plane": x_min,
        "tangent_axis": 1,
        "tangent_min": y_min,
        "tangent_max": y_max,
        "inward_normal_m": np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    },
    "x_max": {
        "normal_axis": 0,
        "plane": x_max,
        "tangent_axis": 1,
        "tangent_min": y_min,
        "tangent_max": y_max,
        "inward_normal_m": np.array(
            [-1.0, 0.0, 0.0],
            dtype=np.float64,
        ),
    },
    "y_min": {
        "normal_axis": 1,
        "plane": y_min,
        "tangent_axis": 0,
        "tangent_min": x_min,
        "tangent_max": x_max,
        "inward_normal_m": np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        ),
    },
    "y_max": {
        "normal_axis": 1,
        "plane": y_max,
        "tangent_axis": 0,
        "tangent_min": x_min,
        "tangent_max": x_max,
        "inward_normal_m": np.array(
            [0.0, -1.0, 0.0],
            dtype=np.float64,
        ),
    },
}


def create_wall_grid(
    definition: dict,
):
    tangent_centers = np.arange(
        definition["tangent_min"]
        + 0.5 * TANGENT_RESOLUTION_M,
        definition["tangent_max"],
        TANGENT_RESOLUTION_M,
    )

    z_centers = np.arange(
        floor_z + 0.5 * HEIGHT_RESOLUTION_M,
        ceiling_z,
        HEIGHT_RESOLUTION_M,
    )

    tangent_grid, z_grid = np.meshgrid(
        tangent_centers,
        z_centers,
    )

    point_count = tangent_grid.size

    points_m = np.zeros(
        (point_count, 3),
        dtype=np.float64,
    )

    tangent_flat = tangent_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)

    if definition["normal_axis"] == 0:
        points_m[:, 0] = definition["plane"]
        points_m[:, 1] = tangent_flat
    else:
        points_m[:, 0] = tangent_flat
        points_m[:, 1] = definition["plane"]

    points_m[:, 2] = z_flat

    points_world = points_m.copy()

    points_world[:, :2] = (
        points_m[:, :2] @ rotation_to_world.T
    )

    normal_world = np.zeros(3, dtype=np.float64)

    normal_world[:2] = (
        definition["inward_normal_m"][:2]
        @ rotation_to_world.T
    )

    normal_world /= np.linalg.norm(normal_world)

    return {
        "points_world": points_world,
        "normal_world": normal_world,
        "tangent_centers": tangent_centers,
        "z_centers": z_centers,
        "shape": z_grid.shape,
    }


def sample_depth_neighborhood(
    depth_raw: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    samples = []

    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            values = (
                depth_raw[v + dv, u + du]
                .astype(np.float64)
                / DEPTH_SCALE
            )

            values[values <= 0.0] = np.nan
            samples.append(values)

    stacked = np.stack(samples, axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        measured_depth = np.nanmedian(
            stacked,
            axis=0,
        )

    return measured_depth


STATE_UNOBSERVED = 0
STATE_WALL = 1
STATE_FREE = 2
STATE_OCCLUDED = 3
STATE_UNCERTAIN = 4

STATE_LABELS = [
    "unobserved",
    "wall observed",
    "free space",
    "occluded",
    "uncertain",
]

STATE_COLORS = [
    "#4d4d4d",
    "#2f6fbd",
    "#d73027",
    "#fdae61",
    "#8073ac",
]

STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(
    np.arange(-0.5, 5.5, 1.0),
    STATE_CMAP.N,
)


def extract_free_components(
    state_grid: np.ndarray,
    free_ratio_grid: np.ndarray,
    free_count_grid: np.ndarray,
):
    free_mask = state_grid == STATE_FREE

    free_mask = binary_closing(
        free_mask,
        structure=np.ones((2, 2), dtype=bool),
    )

    free_mask = binary_opening(
        free_mask,
        structure=np.ones((2, 2), dtype=bool),
    )

    component_labels, component_count = label(
        free_mask
    )

    components = []

    for component_id in range(
        1,
        component_count + 1,
    ):
        rows, columns = np.where(
            component_labels == component_id
        )

        if len(rows) < MIN_COMPONENT_CELLS:
            continue

        minimum_row = int(rows.min())
        maximum_row = int(rows.max())
        minimum_column = int(columns.min())
        maximum_column = int(columns.max())

        width = (
            maximum_column
            - minimum_column
            + 1
        ) * TANGENT_RESOLUTION_M

        height = (
            maximum_row
            - minimum_row
            + 1
        ) * HEIGHT_RESOLUTION_M

        bottom = (
            floor_z
            + minimum_row * HEIGHT_RESOLUTION_M
        )

        top = (
            floor_z
            + (maximum_row + 1)
            * HEIGHT_RESOLUTION_M
        )

        touches_floor = (
            bottom <= floor_z + 0.25
        )

        if (
            touches_floor
            and MIN_DOOR_WIDTH_M <= width <= MAX_DOOR_WIDTH_M
            and height >= MIN_DOOR_HEIGHT_M
        ):
            classification = (
                "validated_door_candidate"
            )

        elif (
            not touches_floor
            and width >= MIN_WINDOW_WIDTH_M
            and height >= MIN_WINDOW_HEIGHT_M
        ):
            classification = (
                "validated_window_candidate"
            )

        else:
            classification = (
                "validated_free_space_region"
            )

        component_mask = (
            component_labels == component_id
        )

        mean_free_ratio = float(
            np.mean(
                free_ratio_grid[component_mask]
            )
        )

        mean_free_observations = float(
            np.mean(
                free_count_grid[component_mask]
            )
        )

        confidence = float(
            mean_free_ratio
            * min(
                1.0,
                mean_free_observations / 4.0,
            )
        )

        components.append(
            {
                "classification": classification,
                "minimum_column": minimum_column,
                "maximum_column": maximum_column,
                "minimum_row": minimum_row,
                "maximum_row": maximum_row,
                "width_m": float(width),
                "height_m": float(height),
                "bottom_z_m": float(bottom),
                "top_z_m": float(top),
                "cell_count": int(len(rows)),
                "mean_free_ratio": mean_free_ratio,
                "mean_free_observations": (
                    mean_free_observations
                ),
                "confidence": confidence,
            }
        )

    return components


def analyze_wall(
    wall_name: str,
    definition: dict,
):
    grid = create_wall_grid(definition)

    points_world = grid["points_world"]
    normal_world = grid["normal_world"]
    point_count = len(points_world)

    projected_count = np.zeros(
        point_count,
        dtype=np.int32,
    )

    valid_depth_count = np.zeros(
        point_count,
        dtype=np.int32,
    )

    wall_count = np.zeros(
        point_count,
        dtype=np.int32,
    )

    free_count = np.zeros(
        point_count,
        dtype=np.int32,
    )

    occluded_count = np.zeros(
        point_count,
        dtype=np.int32,
    )

    uncertain_count = np.zeros(
        point_count,
        dtype=np.int32,
    )

    for frame_index, association in enumerate(
        associations
    ):
        depth_raw = cv2.imread(
            str(association["depth_path"]),
            cv2.IMREAD_UNCHANGED,
        )

        if depth_raw is None:
            print(
                "Warning: failed to read",
                association["depth_path"],
            )
            continue

        transform_wc = association["pose_wc"]
        transform_cw = np.linalg.inv(transform_wc)

        rotation_cw = transform_cw[:3, :3]
        translation_cw = transform_cw[:3, 3]

        points_camera = (
            points_world @ rotation_cw.T
            + translation_cw
        )

        expected_depth = points_camera[:, 2]

        camera_center = transform_wc[:3, 3]

        camera_direction = (
            camera_center[None, :]
            - points_world
        )

        camera_distance = np.linalg.norm(
            camera_direction,
            axis=1,
        )

        camera_direction /= np.maximum(
            camera_distance[:, None],
            1e-8,
        )

        incidence = (
            camera_direction @ normal_world
        )

        u_float = (
            FX * points_camera[:, 0]
            / np.maximum(expected_depth, 1e-8)
            + CX
        )

        v_float = (
            FY * points_camera[:, 1]
            / np.maximum(expected_depth, 1e-8)
            + CY
        )

        u = np.rint(u_float).astype(np.int32)
        v = np.rint(v_float).astype(np.int32)

        projectable = (
            (expected_depth > MIN_CAMERA_DISTANCE_M)
            & (expected_depth < MAX_DEPTH_M)
            & (incidence >= MIN_INCIDENCE_COSINE)
            & (u >= 1)
            & (u < IMAGE_WIDTH - 1)
            & (v >= 1)
            & (v < IMAGE_HEIGHT - 1)
        )

        projected_indices = np.flatnonzero(
            projectable
        )

        if len(projected_indices) == 0:
            continue

        projected_count[projected_indices] += 1

        measured_depth = sample_depth_neighborhood(
            depth_raw,
            u[projected_indices],
            v[projected_indices],
        )

        valid_measurement = (
            np.isfinite(measured_depth)
            & (measured_depth > 0.10)
            & (measured_depth < MAX_DEPTH_M)
        )

        valid_indices = projected_indices[
            valid_measurement
        ]

        if len(valid_indices) == 0:
            continue

        valid_depth_count[valid_indices] += 1

        measured_valid = measured_depth[
            valid_measurement
        ]

        expected_valid = expected_depth[
            valid_indices
        ]

        depth_difference = (
            measured_valid - expected_valid
        )

        wall_evidence = (
            np.abs(depth_difference)
            <= WALL_MATCH_TOLERANCE_M
        )

        free_evidence = (
            depth_difference
            >= FREE_SPACE_MARGIN_M
        )

        occlusion_evidence = (
            depth_difference
            <= -OCCLUSION_MARGIN_M
        )

        uncertain_evidence = ~(
            wall_evidence
            | free_evidence
            | occlusion_evidence
        )

        wall_count[
            valid_indices[wall_evidence]
        ] += 1

        free_count[
            valid_indices[free_evidence]
        ] += 1

        occluded_count[
            valid_indices[occlusion_evidence]
        ] += 1

        uncertain_count[
            valid_indices[uncertain_evidence]
        ] += 1

        if (
            (frame_index + 1) % 10 == 0
            or frame_index == len(associations) - 1
        ):
            print(
                f"[{wall_name}] processed "
                f"{frame_index + 1}/"
                f"{len(associations)} frames"
            )

    comparable_count = (
        wall_count + free_count
    )

    free_ratio = np.divide(
        free_count,
        comparable_count,
        out=np.zeros_like(
            free_count,
            dtype=np.float64,
        ),
        where=comparable_count > 0,
    )

    wall_ratio = np.divide(
        wall_count,
        comparable_count,
        out=np.zeros_like(
            wall_count,
            dtype=np.float64,
        ),
        where=comparable_count > 0,
    )

    state = np.full(
        point_count,
        STATE_UNOBSERVED,
        dtype=np.uint8,
    )

    enough_evidence = (
        valid_depth_count >= MIN_VALID_OBSERVATIONS
    )

    state[enough_evidence] = STATE_UNCERTAIN

    occlusion_state = (
        enough_evidence
        & (
            occluded_count
            >= MIN_OCCLUSION_OBSERVATIONS
        )
        & (
            occluded_count
            >= wall_count + free_count
        )
    )

    wall_state = (
        enough_evidence
        & (wall_count >= MIN_WALL_OBSERVATIONS)
        & (wall_ratio >= MIN_WALL_RATIO)
    )

    free_state = (
        enough_evidence
        & (free_count >= MIN_FREE_OBSERVATIONS)
        & (free_ratio >= MIN_FREE_RATIO)
    )

    state[occlusion_state] = STATE_OCCLUDED
    state[wall_state] = STATE_WALL

    # Free-space evidence has the highest priority because it
    # requires repeated measurements behind the expected wall.
    state[free_state] = STATE_FREE

    state_grid = state.reshape(grid["shape"])
    free_ratio_grid = free_ratio.reshape(grid["shape"])
    free_count_grid = free_count.reshape(grid["shape"])

    components = extract_free_components(
        state_grid,
        free_ratio_grid,
        free_count_grid,
    )

    # Visibility-state map.
    fig, ax = plt.subplots(
        figsize=(12, 6),
        dpi=170,
    )

    image = ax.imshow(
        state_grid,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        cmap=STATE_CMAP,
        norm=STATE_NORM,
        extent=[
            definition["tangent_min"],
            definition["tangent_max"],
            floor_z,
            ceiling_z,
        ],
    )

    for component in components:
        tangent_start = (
            definition["tangent_min"]
            + component["minimum_column"]
            * TANGENT_RESOLUTION_M
        )

        ax.add_patch(
            Rectangle(
                (
                    tangent_start,
                    component["bottom_z_m"],
                ),
                component["width_m"],
                component["height_m"],
                fill=False,
                linewidth=2.0,
            )
        )

        ax.text(
            tangent_start
            + 0.5 * component["width_m"],
            component["bottom_z_m"]
            + 0.5 * component["height_m"],
            component["classification"],
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
        )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=np.arange(5),
    )

    colorbar.ax.set_yticklabels(STATE_LABELS)

    ax.set_xlabel("Position along wall (m)")
    ax.set_ylabel("Height Z (m)")
    ax.set_title(
        f"{wall_name}: RGB-D visibility classification"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{wall_name}_visibility.png"
    )
    plt.close(fig)

    # Free-space confidence map.
    fig, ax = plt.subplots(
        figsize=(12, 6),
        dpi=170,
    )

    confidence_image = ax.imshow(
        free_ratio_grid,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=[
            definition["tangent_min"],
            definition["tangent_max"],
            floor_z,
            ceiling_z,
        ],
        vmin=0.0,
        vmax=1.0,
    )

    fig.colorbar(
        confidence_image,
        ax=ax,
        label="Free-space ratio among wall/free evidence",
    )

    ax.set_xlabel("Position along wall (m)")
    ax.set_ylabel("Height Z (m)")
    ax.set_title(
        f"{wall_name}: free-space confidence"
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / f"{wall_name}_free_confidence.png"
    )
    plt.close(fig)

    cell_count = point_count

    state_counts = {
        STATE_LABELS[state_id]: int(
            np.sum(state == state_id)
        )
        for state_id in range(5)
    }

    return {
        "grid_shape": list(grid["shape"]),
        "cell_count": int(cell_count),
        "projected_cell_fraction": float(
            np.mean(projected_count > 0)
        ),
        "valid_depth_cell_fraction": float(
            np.mean(valid_depth_count > 0)
        ),
        "mean_projected_frames_per_cell": float(
            np.mean(projected_count)
        ),
        "mean_valid_depth_frames_per_cell": float(
            np.mean(valid_depth_count)
        ),
        "state_counts": state_counts,
        "state_fractions": {
            label_name: float(
                count / cell_count
            )
            for label_name, count
            in state_counts.items()
        },
        "validated_regions": components,
        "definition": {
            "plane_coordinate_m": float(
                definition["plane"]
            ),
            "tangent_min_m": float(
                definition["tangent_min"]
            ),
            "tangent_max_m": float(
                definition["tangent_max"]
            ),
        },
    }


wall_results = {}

for wall_name, definition in WALL_DEFINITIONS.items():
    print()
    print("=" * 60)
    print("Analyzing wall:", wall_name)

    wall_results[wall_name] = analyze_wall(
        wall_name,
        definition,
    )


# ------------------------------------------------------------
# Floor-plan visualization of validated free-space regions.
# ------------------------------------------------------------

candidate_cloud = o3d.io.read_point_cloud(
    str(WALL_CANDIDATES)
)

candidate_points = np.asarray(
    candidate_cloud.points,
    dtype=np.float64,
)

candidate_points = candidate_points[
    np.isfinite(candidate_points).all(axis=1)
]

candidate_points_m = (
    candidate_points[:, :2]
    @ rotation_to_manhattan.T
)

rng = np.random.default_rng(42)

plot_count = min(
    100_000,
    len(candidate_points_m),
)

plot_indices = rng.choice(
    len(candidate_points_m),
    plot_count,
    replace=False,
)

fig, ax = plt.subplots(
    figsize=(10, 9),
    dpi=180,
)

ax.scatter(
    candidate_points_m[plot_indices, 0],
    candidate_points_m[plot_indices, 1],
    s=0.18,
    linewidths=0,
    alpha=0.25,
)

room_rectangle = np.array(
    [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ],
    dtype=np.float64,
)

ax.plot(
    room_rectangle[:, 0],
    room_rectangle[:, 1],
    linewidth=2.5,
    label="Refined envelope",
)


def component_interval(
    component: dict,
    definition: dict,
):
    start = (
        definition["tangent_min"]
        + component["minimum_column"]
        * TANGENT_RESOLUTION_M
    )

    end = start + component["width_m"]
    return start, end


validated_count = 0

for wall_name, result in wall_results.items():
    definition = WALL_DEFINITIONS[wall_name]

    for component in result["validated_regions"]:
        start, end = component_interval(
            component,
            definition,
        )

        validated_count += 1

        if wall_name == "x_min":
            ax.plot(
                [x_min, x_min],
                [start, end],
                linewidth=7,
            )

        elif wall_name == "x_max":
            ax.plot(
                [x_max, x_max],
                [start, end],
                linewidth=7,
            )

        elif wall_name == "y_min":
            ax.plot(
                [start, end],
                [y_min, y_min],
                linewidth=7,
            )

        elif wall_name == "y_max":
            ax.plot(
                [start, end],
                [y_max, y_max],
                linewidth=7,
            )

ax.set_xlabel("Manhattan X (m)")
ax.set_ylabel("Manhattan Y (m)")
ax.set_title(
    "RGB-D validated free-space regions"
)

ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR
    / "validated_openings_floorplan.png"
)
plt.close(fig)


summary = {
    "method": (
        "Project perimeter-wall cells into aligned estimated-pose "
        "RGB-D frames and compare expected wall depth against measured "
        "depth. Repeated farther-depth measurements provide free-space "
        "evidence; closer depths indicate occlusion."
    ),
    "association_count": len(associations),
    "maximum_rgb_time_difference_s": max(
        item["rgb_difference_s"]
        for item in associations
    ),
    "maximum_depth_time_difference_s": max(
        item["depth_difference_s"]
        for item in associations
    ),
    "pose_frame": (
        "Estimated MASt3R-SLAM poses rigidly aligned using the same "
        "estimated_to_gt SE3 transform used for TSDF reconstruction."
    ),
    "parameters": {
        "tangent_resolution_m": TANGENT_RESOLUTION_M,
        "height_resolution_m": HEIGHT_RESOLUTION_M,
        "minimum_incidence_cosine": MIN_INCIDENCE_COSINE,
        "wall_match_tolerance_m": WALL_MATCH_TOLERANCE_M,
        "free_space_margin_m": FREE_SPACE_MARGIN_M,
        "occlusion_margin_m": OCCLUSION_MARGIN_M,
        "minimum_valid_observations": MIN_VALID_OBSERVATIONS,
        "minimum_wall_observations": MIN_WALL_OBSERVATIONS,
        "minimum_free_observations": MIN_FREE_OBSERVATIONS,
        "minimum_free_ratio": MIN_FREE_RATIO,
        "minimum_wall_ratio": MIN_WALL_RATIO,
    },
    "wall_results": wall_results,
    "validated_region_count": int(validated_count),
    "outputs": {
        "validated_floorplan": str(
            OUTPUT_DIR
            / "validated_openings_floorplan.png"
        ),
    },
    "warning": (
        "Validated free-space regions are geometric hypotheses. "
        "Door/window semantics still require shape checks or RGB evidence."
    ),
}

summary_path = (
    OUTPUT_DIR
    / "visibility_opening_summary.json"
)

with summary_path.open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, indent=2)

print()
print("========== VISIBILITY SUMMARY ==========")

print(
    "Associated frames:",
    summary["association_count"],
)

print(
    "Validated regions:",
    summary["validated_region_count"],
)

for wall_name, result in wall_results.items():
    print()
    print("WALL:", wall_name)
    print(
        "Projected cell fraction:",
        round(
            result["projected_cell_fraction"],
            3,
        ),
    )
    print(
        "Valid-depth cell fraction:",
        round(
            result["valid_depth_cell_fraction"],
            3,
        ),
    )
    print(
        "State fractions:",
        result["state_fractions"],
    )
    print(
        "Validated regions:",
        len(result["validated_regions"]),
    )

    for region in result["validated_regions"]:
        print(
            " ",
            region["classification"],
            "width=",
            round(region["width_m"], 3),
            "height=",
            round(region["height_m"], 3),
            "confidence=",
            round(region["confidence"], 3),
        )

print()
print("Saved:", summary_path)
print("VISIBILITY_OPENING_VALIDATION_OK")
