from pathlib import Path
import json

import trimesh


root = Path(
    "/workspace/interior-slam/results/tum_fr1_room_baseline/"
    "tsdf/final/structure_baseline/final_structured_model"
)

models = {
    "conservative": root / "room_envelope_conservative.glb",
    "probable_openings": root / "room_envelope_probable_openings.glb",
}

summary = {}

for name, input_path in models.items():
    scene = trimesh.load(
        input_path,
        force="scene",
        process=False,
    )

    mesh = trimesh.util.concatenate(
        tuple(scene.geometry.values())
    )

    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)

    # Weld duplicated vertices shared by neighboring wall panels.
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()

    # Correct inconsistent face winding and normals.
    trimesh.repair.fix_normals(
        mesh,
        multibody=True,
    )

    output_glb = root / f"{input_path.stem}_final.glb"
    output_ply = root / f"{input_path.stem}_final.ply"

    trimesh.Scene(mesh).export(output_glb)
    mesh.export(output_ply)

    summary[name] = {
        "input": str(input_path),
        "output_glb": str(output_glb),
        "output_ply": str(output_ply),
        "original_vertices": original_vertices,
        "original_faces": original_faces,
        "final_vertices": int(len(mesh.vertices)),
        "final_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(
            mesh.is_winding_consistent
        ),
        "surface_area_m2": float(mesh.area),
        "volume_m3": (
            float(mesh.volume)
            if mesh.is_watertight
            else None
        ),
    }

    print()
    print(name)
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Watertight:", mesh.is_watertight)
    print(
        "Winding consistent:",
        mesh.is_winding_consistent,
    )

    if mesh.is_watertight:
        print("Volume m³:", mesh.volume)

    print("Saved:", output_glb)
    print("Saved:", output_ply)

summary_path = root / "final_mesh_validation.json"

with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

assert summary["conservative"]["watertight"]
assert summary["conservative"]["winding_consistent"]

print()
print("Saved:", summary_path)
print("FINAL_STRUCTURED_MESHES_OK")
