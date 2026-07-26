from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def read_ply_xyz(path: Path) -> tuple[np.memmap, int]:
    """Memory-map the XYZ vertex block of an XYZ-only binary PLY."""
    with path.open("rb") as handle:
        vertex_count = None

        while True:
            line = handle.readline()

            if not line:
                raise RuntimeError(f"No end_header found in {path}")

            text = line.decode("ascii").strip()

            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])

            if text == "end_header":
                data_offset = handle.tell()
                break

    if vertex_count is None:
        raise RuntimeError(f"No vertex count found in {path}")

    xyz = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=data_offset,
        shape=(vertex_count, 3),
    )

    return xyz, vertex_count


def read_alignment_matrices(path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(path).getroot()
    matrices: dict[str, np.ndarray] = {}

    for mesh in root.findall(".//MLMesh"):
        filename = Path(mesh.attrib["filename"]).name
        matrix_text = mesh.findtext("MLMatrix44")

        if matrix_text is None:
            raise RuntimeError(f"Missing transform for {filename}")

        values = np.fromstring(
            matrix_text,
            sep=" ",
            dtype=np.float64,
        )

        if values.size != 16:
            raise RuntimeError(
                f"{filename}: expected 16 matrix values, got {values.size}"
            )

        matrices[filename] = values.reshape(4, 4)

    return matrices


def sample_and_transform(
    path: Path,
    transform: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, int]:
    xyz_memmap, total_count = read_ply_xyz(path)

    sample_count = min(total_count, max_points)

    indices = np.linspace(
        0,
        total_count - 1,
        sample_count,
        dtype=np.int64,
    )

    xyz = np.asarray(
        xyz_memmap[indices],
        dtype=np.float64,
    )

    homogeneous = np.concatenate(
        [
            xyz,
            np.ones((len(xyz), 1), dtype=np.float64),
        ],
        axis=1,
    )

    xyz_global = (
        transform @ homogeneous.T
    ).T[:, :3]

    finite = np.isfinite(xyz_global).all(axis=1)

    return xyz_global[finite], total_count


def write_binary_ply(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz, dtype="<f4")

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as handle:
        handle.write(header)
        xyz.tofile(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_ply",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--points_per_scan",
        type=int,
        default=1_000_000,
    )
    args = parser.parse_args()

    matrices = read_alignment_matrices(
        args.scan_dir / "scan_alignment.mlp"
    )

    merged = []
    report = {
        "points_per_scan_limit": args.points_per_scan,
        "scans": {},
    }

    for filename in ["scan1.ply", "scan2.ply"]:
        path = args.scan_dir / filename

        if filename not in matrices:
            raise RuntimeError(
                f"No alignment matrix found for {filename}"
            )

        xyz_global, total_count = sample_and_transform(
            path,
            matrices[filename],
            args.points_per_scan,
        )

        merged.append(xyz_global)

        report["scans"][filename] = {
            "source_vertex_count": total_count,
            "sampled_vertex_count": int(len(xyz_global)),
            "global_bounds_min": xyz_global.min(axis=0).tolist(),
            "global_bounds_max": xyz_global.max(axis=0).tolist(),
            "matrix": matrices[filename].tolist(),
        }

        print(
            f"{filename}: sampled {len(xyz_global):,} "
            f"of {total_count:,} points"
        )

    merged_xyz = np.concatenate(merged, axis=0)

    report["merged_vertex_count"] = int(len(merged_xyz))
    report["merged_bounds_min"] = (
        merged_xyz.min(axis=0).tolist()
    )
    report["merged_bounds_max"] = (
        merged_xyz.max(axis=0).tolist()
    )

    write_binary_ply(
        args.output_ply,
        merged_xyz,
    )

    args.output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_json.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved PLY: {args.output_ply}")
    print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
