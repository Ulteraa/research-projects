from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh


ROOT = Path(
    "/workspace/interior-slam/results/tum_fr1_room_baseline/"
    "tsdf/final/structure_baseline"
)

MANHATTAN_SUMMARY = (
    ROOT / "manhattan_walls/manhattan_wall_summary.json"
)

REFINED_SUMMARY = (
    ROOT / "refined_envelope/refined_envelope_summary.json"
)

VISIBILITY_SUMMARY = (
    ROOT / "visibility_openings/visibility_opening_summary.json"
)

WALL_CANDIDATES = ROOT / "wall_candidates.ply"

OUTPUT_DIR = ROOT / "final_structured_model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Conservative architectural acceptance rules.
MIN_ACCEPTED_CONFIDENCE = 0.75

MIN_WINDOW_WIDTH_M = 0.60
MIN_WINDOW_HEIGHT_M = 0.60
MIN_WINDOW_SILL_HEIGHT_M = 0.30

MIN_DOOR_WIDTH_M = 0.70
MIN_DOOR_HEIGHT_M = 1.80
MAX_DOOR_BOTTOM_OFFSET_M = 0.20


with MANHATTAN_SUMMARY.open("r", encoding="utf-8") as handle:
    manhattan = json.load(handle)

with REFINED_SUMMARY.open("r", encoding="utf-8") as handle:
    refined = json.load(handle)

with VISIBILITY_SUMMARY.open("r", encoding="utf-8") as handle:
    visibility = json.load(handle)


floor_z = float(manhattan["floor_z_m"])
ceiling_z = float(manhattan["ceiling_z_m"])
room_height = ceiling_z - floor_z

angle = float(manhattan["manhattan"]["angle_rad"])

rotation_to_world = np.array(
    [
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)],
    ],
    dtype=np.float64,
)

rotation_to_manhattan = np.array(
    [
        [math.cos(angle),  math.sin(angle)],
        [-math.sin(angle), math.cos(angle)],
    ],
    dtype=np.float64,
)

bounds = refined["bounds_manhattan"]

x_min = float(bounds["x_min_m"])
x_max = float(bounds["x_max_m"])
y_min = float(bounds["y_min_m"])
y_max = float(bounds["y_max_m"])

tangent_resolution = float(
    visibility["parameters"]["tangent_resolution_m"]
)


# ------------------------------------------------------------
# Classify visibility-supported regions conservatively.
# ------------------------------------------------------------

accepted_openings = []
rejected_regions = []

for wall_name, wall_result in visibility["wall_results"].items():
    definition = wall_result["definition"]
    tangent_min = float(definition["tangent_min_m"])

    for region_index, region in enumerate(
        wall_result["validated_regions"]
    ):
        tangent_start = (
            tangent_min
            + region["minimum_column"] * tangent_resolution
        )

        tangent_end = tangent_start + float(region["width_m"])

        candidate = {
            "id": f"{wall_name}_{region_index:02d}",
            "wall": wall_name,
            "classification": region["classification"],
            "tangent_start_m": float(tangent_start),
            "tangent_end_m": float(tangent_end),
            "width_m": float(region["width_m"]),
            "height_m": float(region["height_m"]),
            "bottom_z_m": float(region["bottom_z_m"]),
            "top_z_m": float(region["top_z_m"]),
            "confidence": float(region["confidence"]),
            "mean_free_ratio": float(region["mean_free_ratio"]),
            "mean_free_observations": float(
                region["mean_free_observations"]
            ),
        }

        is_window = (
            region["classification"]
            == "validated_window_candidate"
            and candidate["confidence"] >= MIN_ACCEPTED_CONFIDENCE
            and candidate["width_m"] >= MIN_WINDOW_WIDTH_M
            and candidate["height_m"] >= MIN_WINDOW_HEIGHT_M
            and candidate["bottom_z_m"]
            >= floor_z + MIN_WINDOW_SILL_HEIGHT_M
        )

        is_door = (
            region["classification"]
            == "validated_door_candidate"
            and candidate["confidence"] >= MIN_ACCEPTED_CONFIDENCE
            and candidate["width_m"] >= MIN_DOOR_WIDTH_M
            and candidate["height_m"] >= MIN_DOOR_HEIGHT_M
            and candidate["bottom_z_m"]
            <= floor_z + MAX_DOOR_BOTTOM_OFFSET_M
        )

        if is_window:
            candidate["architectural_label"] = "probable_window"
            candidate["status"] = (
                "accepted_in_hypothesis_model_only"
            )
            candidate["semantic_certainty"] = (
                "geometry-supported; RGB semantics not confirmed"
            )
            accepted_openings.append(candidate)

        elif is_door:
            candidate["architectural_label"] = "probable_door"
            candidate["status"] = (
                "accepted_in_hypothesis_model_only"
            )
            candidate["semantic_certainty"] = (
                "geometry-supported; RGB semantics not confirmed"
            )
            accepted_openings.append(candidate)

        else:
            candidate["status"] = "rejected_from_architectural_model"
            candidate["rejection_reason"] = (
                "Did not satisfy conservative architectural size, "
                "placement, classification, and confidence rules."
            )
            rejected_regions.append(candidate)


print("Accepted probable openings:", len(accepted_openings))

for opening in accepted_openings:
    print(
        opening["architectural_label"],
        opening["wall"],
        "width=", round(opening["width_m"], 3),
        "height=", round(opening["height_m"], 3),
        "confidence=", round(opening["confidence"], 3),
    )

print("Rejected free-space regions:", len(rejected_regions))


# ------------------------------------------------------------
# Generate closed and opening-aware room-envelope meshes.
# ------------------------------------------------------------

WALL_DEFINITIONS = {
    "x_min": {
        "normal_axis": 0,
        "plane": x_min,
        "tangent_min": y_min,
        "tangent_max": y_max,
    },
    "x_max": {
        "normal_axis": 0,
        "plane": x_max,
        "tangent_min": y_min,
        "tangent_max": y_max,
    },
    "y_min": {
        "normal_axis": 1,
        "plane": y_min,
        "tangent_min": x_min,
        "tangent_max": x_max,
    },
    "y_max": {
        "normal_axis": 1,
        "plane": y_max,
        "tangent_min": x_min,
        "tangent_max": x_max,
    },
}


def transform_points_to_world(
    points_m: np.ndarray,
) -> np.ndarray:
    points_world = points_m.copy()

    points_world[:, :2] = (
        points_m[:, :2] @ rotation_to_world.T
    )

    return points_world


def add_quad(
    vertices: list,
    faces: list,
    face_colors: list,
    p0,
    p1,
    p2,
    p3,
    color,
):
    start = len(vertices)

    vertices.extend([p0, p1, p2, p3])

    faces.extend(
        [
            [start + 0, start + 1, start + 2],
            [start + 0, start + 2, start + 3],
        ]
    )

    face_colors.extend([color, color])


def cell_is_inside_opening(
    tangent_center: float,
    z_center: float,
    openings: list[dict],
) -> bool:
    for opening in openings:
        if (
            opening["tangent_start_m"]
            <= tangent_center
            <= opening["tangent_end_m"]
            and opening["bottom_z_m"]
            <= z_center
            <= opening["top_z_m"]
        ):
            return True

    return False


def add_wall_panels(
    wall_name: str,
    definition: dict,
    wall_openings: list[dict],
    vertices: list,
    faces: list,
    face_colors: list,
):
    tangent_breaks = [
        definition["tangent_min"],
        definition["tangent_max"],
    ]

    z_breaks = [floor_z, ceiling_z]

    for opening in wall_openings:
        tangent_breaks.extend(
            [
                opening["tangent_start_m"],
                opening["tangent_end_m"],
            ]
        )

        z_breaks.extend(
            [
                opening["bottom_z_m"],
                opening["top_z_m"],
            ]
        )

    tangent_breaks = sorted(
        {
            float(np.clip(
                value,
                definition["tangent_min"],
                definition["tangent_max"],
            ))
            for value in tangent_breaks
        }
    )

    z_breaks = sorted(
        {
            float(np.clip(value, floor_z, ceiling_z))
            for value in z_breaks
        }
    )

    wall_color = [205, 215, 225, 255]

    for tangent_index in range(len(tangent_breaks) - 1):
        tangent_0 = tangent_breaks[tangent_index]
        tangent_1 = tangent_breaks[tangent_index + 1]

        if tangent_1 - tangent_0 <= 1e-6:
            continue

        for z_index in range(len(z_breaks) - 1):
            z_0 = z_breaks[z_index]
            z_1 = z_breaks[z_index + 1]

            if z_1 - z_0 <= 1e-6:
                continue

            tangent_center = 0.5 * (tangent_0 + tangent_1)
            z_center = 0.5 * (z_0 + z_1)

            if cell_is_inside_opening(
                tangent_center,
                z_center,
                wall_openings,
            ):
                continue

            plane = definition["plane"]

            if definition["normal_axis"] == 0:
                p0 = [plane, tangent_0, z_0]
                p1 = [plane, tangent_1, z_0]
                p2 = [plane, tangent_1, z_1]
                p3 = [plane, tangent_0, z_1]

            else:
                p0 = [tangent_0, plane, z_0]
                p1 = [tangent_1, plane, z_0]
                p2 = [tangent_1, plane, z_1]
                p3 = [tangent_0, plane, z_1]

            add_quad(
                vertices,
                faces,
                face_colors,
                p0,
                p1,
                p2,
                p3,
                wall_color,
            )


def build_room_mesh(
    openings: list[dict],
) -> trimesh.Trimesh:
    vertices = []
    faces = []
    face_colors = []

    # Floor.
    add_quad(
        vertices,
        faces,
        face_colors,
        [x_min, y_min, floor_z],
        [x_max, y_min, floor_z],
        [x_max, y_max, floor_z],
        [x_min, y_max, floor_z],
        [175, 185, 195, 255],
    )

    # Ceiling.
    add_quad(
        vertices,
        faces,
        face_colors,
        [x_min, y_min, ceiling_z],
        [x_min, y_max, ceiling_z],
        [x_max, y_max, ceiling_z],
        [x_max, y_min, ceiling_z],
        [225, 225, 230, 255],
    )

    for wall_name, definition in WALL_DEFINITIONS.items():
        wall_openings = [
            opening
            for opening in openings
            if opening["wall"] == wall_name
        ]

        add_wall_panels(
            wall_name,
            definition,
            wall_openings,
            vertices,
            faces,
            face_colors,
        )

    vertices_m = np.asarray(vertices, dtype=np.float64)
    vertices_world = transform_points_to_world(vertices_m)

    mesh = trimesh.Trimesh(
        vertices=vertices_world,
        faces=np.asarray(faces, dtype=np.int64),
        face_colors=np.asarray(face_colors, dtype=np.uint8),
        process=False,
    )

    return mesh


def save_mesh_variants(
    mesh: trimesh.Trimesh,
    stem: str,
):
    ply_path = OUTPUT_DIR / f"{stem}.ply"
    glb_path = OUTPUT_DIR / f"{stem}.glb"

    mesh.export(ply_path)

    scene = trimesh.Scene(mesh)
    scene.export(glb_path)

    print("Saved:", ply_path)
    print("Saved:", glb_path)

    return ply_path, glb_path


closed_mesh = build_room_mesh(openings=[])

closed_ply, closed_glb = save_mesh_variants(
    closed_mesh,
    "room_envelope_conservative",
)

hypothesis_mesh = build_room_mesh(
    openings=accepted_openings
)

hypothesis_ply, hypothesis_glb = save_mesh_variants(
    hypothesis_mesh,
    "room_envelope_probable_openings",
)


# ------------------------------------------------------------
# Generate a clean structural floor-plan visualization.
# ------------------------------------------------------------

cloud = o3d.io.read_point_cloud(str(WALL_CANDIDATES))

candidate_points = np.asarray(
    cloud.points,
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

plot_count = min(100_000, len(candidate_points_m))

plot_indices = rng.choice(
    len(candidate_points_m),
    plot_count,
    replace=False,
)

rectangle = np.array(
    [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ],
    dtype=np.float64,
)

fig, ax = plt.subplots(figsize=(10, 9), dpi=180)

ax.scatter(
    candidate_points_m[plot_indices, 0],
    candidate_points_m[plot_indices, 1],
    s=0.18,
    linewidths=0,
    alpha=0.18,
    label="Vertical reconstruction evidence",
)

ax.plot(
    rectangle[:, 0],
    rectangle[:, 1],
    linewidth=3,
    label="Refined room envelope",
)

for opening in accepted_openings:
    start = opening["tangent_start_m"]
    end = opening["tangent_end_m"]

    if opening["wall"] == "x_min":
        x_values = [x_min, x_min]
        y_values = [start, end]

    elif opening["wall"] == "x_max":
        x_values = [x_max, x_max]
        y_values = [start, end]

    elif opening["wall"] == "y_min":
        x_values = [start, end]
        y_values = [y_min, y_min]

    else:
        x_values = [start, end]
        y_values = [y_max, y_max]

    ax.plot(
        x_values,
        y_values,
        linewidth=8,
        label="Probable window/opening",
    )

    ax.text(
        np.mean(x_values),
        np.mean(y_values),
        (
            f"{opening['architectural_label']}\n"
            f"{opening['width_m']:.2f} × "
            f"{opening['height_m']:.2f} m\n"
            f"confidence {opening['confidence']:.2f}"
        ),
        ha="center",
        va="center",
        fontsize=8,
    )

# Show persistent internal planes as diagnostics only.
for plane in refined["internal_persistent_planes"]["y_planes"]:
    coordinate = float(plane["coordinate_m"])

    ax.plot(
        [x_min, x_max],
        [coordinate, coordinate],
        linestyle="--",
        linewidth=1.8,
        alpha=0.75,
        label="Persistent internal plane",
    )

ax.set_xlabel("Manhattan X (m)")
ax.set_ylabel("Manhattan Y (m)")
ax.set_title("Uncertainty-aware structured floor plan")
ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.15)

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys())

fig.tight_layout()

floorplan_path = OUTPUT_DIR / "structured_floorplan.png"
fig.savefig(floorplan_path)
plt.close(fig)

print("Saved:", floorplan_path)


# ------------------------------------------------------------
# Final machine-readable structured-room representation.
# ------------------------------------------------------------

wall_quality = {}

for wall_name, result in visibility["wall_results"].items():
    wall_quality[wall_name] = {
        "projected_cell_fraction": result[
            "projected_cell_fraction"
        ],
        "valid_depth_cell_fraction": result[
            "valid_depth_cell_fraction"
        ],
        "state_fractions": result["state_fractions"],
    }


structured_room = {
    "project_stage": "Stage 2D — uncertainty-aware structured model",
    "coordinate_frames": {
        "manhattan_angle_rad": angle,
        "manhattan_angle_deg": math.degrees(angle),
        "output_mesh_frame": "aligned reconstruction/world frame",
        "floorplan_frame": "Manhattan frame",
    },
    "room": {
        "bounds_manhattan": {
            "x_min_m": x_min,
            "x_max_m": x_max,
            "y_min_m": y_min,
            "y_max_m": y_max,
            "floor_z_m": floor_z,
            "ceiling_z_m": ceiling_z,
        },
        "dimensions_m": refined["dimensions_m"],
        "corners_manhattan": refined["corners_manhattan"],
        "corners_world": refined["corners_world"],
    },
    "openings": {
        "accepted_probable_openings": accepted_openings,
        "rejected_free_space_regions": rejected_regions,
        "confirmed_doors": [],
        "confirmed_windows": [],
        "interpretation": (
            "Accepted openings are geometric hypotheses included only "
            "in the hypothesis model. No opening is semantically confirmed."
        ),
    },
    "internal_structure": refined[
        "internal_persistent_planes"
    ],
    "wall_visibility_quality": wall_quality,
    "model_variants": {
        "conservative": {
            "description": (
                "Closed room envelope. Recommended where false-positive "
                "openings are more costly than missing openings."
            ),
            "ply": str(closed_ply),
            "glb": str(closed_glb),
        },
        "probable_openings": {
            "description": (
                "Room envelope containing only visibility-validated "
                "architectural hypotheses passing conservative rules."
            ),
            "ply": str(hypothesis_ply),
            "glb": str(hypothesis_glb),
        },
    },
    "outputs": {
        "structured_floorplan": str(floorplan_path),
    },
    "limitations": [
        "The ceiling has relatively sparse direct support.",
        "The probable window has geometric depth support but no explicit RGB semantic confirmation.",
        "Persistent internal planes are retained as diagnostics rather than automatically modeled as complete walls.",
        "The model is a structured room envelope, not a watertight architectural CAD reconstruction of every object.",
    ],
}

json_path = OUTPUT_DIR / "structured_room.json"

with json_path.open("w", encoding="utf-8") as handle:
    json.dump(structured_room, handle, indent=2)

print("Saved:", json_path)

print()
print("========== FINAL STRUCTURED MODEL ==========")
print("Room width:", round(x_max - x_min, 3), "m")
print("Room length:", round(y_max - y_min, 3), "m")
print("Room height:", round(room_height, 3), "m")
print("Accepted probable openings:", len(accepted_openings))
print("Rejected free-space anomalies:", len(rejected_regions))
print("Confirmed doors: 0")
print("Confirmed windows: 0")
print("FINAL_STRUCTURED_MODEL_OK")
