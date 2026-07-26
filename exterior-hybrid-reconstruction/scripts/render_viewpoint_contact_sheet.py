from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def read_binary_ply(
    path: Path,
) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        format_name = None
        vertex_count = None
        active_element = None
        properties: list[tuple[str, str]] = []

        while True:
            line = handle.readline()

            if not line:
                raise RuntimeError(
                    f"No end_header found in {path}"
                )

            text = line.decode("ascii").strip()
            tokens = text.split()

            if tokens[:1] == ["format"]:
                format_name = tokens[1]

            elif tokens[:2] == ["element", "vertex"]:
                vertex_count = int(tokens[2])
                active_element = "vertex"

            elif tokens[:1] == ["element"]:
                active_element = tokens[1]

            elif (
                tokens[:1] == ["property"]
                and active_element == "vertex"
            ):
                property_type = tokens[1]
                property_name = tokens[2]

                if property_type not in PLY_TYPES:
                    raise RuntimeError(
                        f"Unsupported property type: "
                        f"{property_type}"
                    )

                properties.append(
                    (
                        property_name,
                        PLY_TYPES[property_type],
                    )
                )

            elif text == "end_header":
                offset = handle.tell()
                break

    if format_name != "binary_little_endian":
        raise RuntimeError(
            f"Expected binary little-endian PLY; "
            f"received {format_name}"
        )

    if vertex_count is None:
        raise RuntimeError("Missing vertex count")

    dtype = np.dtype(properties)

    vertices = np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=offset,
        shape=(vertex_count,),
    )

    return {
        name: np.asarray(vertices[name])
        for name, _ in properties
    }


def equalize_axes(
    axis,
    xyz: np.ndarray,
) -> None:
    lower = np.percentile(
        xyz,
        0.5,
        axis=0,
    )

    upper = np.percentile(
        xyz,
        99.5,
        axis=0,
    )

    center = 0.5 * (lower + upper)
    radius = 0.55 * float(
        np.max(upper - lower)
    )

    axis.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )

    axis.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )

    axis.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )

    axis.set_box_aspect((1, 1, 1))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_ply",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output_png",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--maximum_points",
        type=int,
        default=30000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    properties = read_binary_ply(
        args.input_ply
    )

    xyz = np.stack(
        [
            properties["x"],
            properties["y"],
            properties["z"],
        ],
        axis=1,
    ).astype(np.float64)

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]

    if all(
        name in properties
        for name in ["red", "green", "blue"]
    ):
        rgb = np.stack(
            [
                properties["red"],
                properties["green"],
                properties["blue"],
            ],
            axis=1,
        )[finite].astype(np.float64) / 255.0
    else:
        rgb = np.full(
            (len(xyz), 3),
            0.65,
            dtype=np.float64,
        )

    if len(xyz) > args.maximum_points:
        generator = np.random.default_rng(
            args.seed
        )

        selected = generator.choice(
            len(xyz),
            size=args.maximum_points,
            replace=False,
        )

        xyz = xyz[selected]
        rgb = rgb[selected]

    viewpoints = [
        (15, 0),
        (15, 45),
        (15, 90),
        (15, 135),
        (25, 180),
        (25, 225),
        (25, 270),
        (25, 315),
        (40, 20),
        (40, 110),
        (40, 200),
        (40, 290),
    ]

    figure = plt.figure(
        figsize=(16, 12),
        dpi=160,
    )

    figure.patch.set_facecolor("white")

    for index, (elevation, azimuth) in enumerate(
        viewpoints,
        start=1,
    ):
        axis = figure.add_subplot(
            3,
            4,
            index,
            projection="3d",
        )

        axis.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=rgb,
            s=0.20,
            linewidths=0,
            depthshade=False,
        )

        equalize_axes(axis, xyz)

        axis.view_init(
            elev=elevation,
            azim=azimuth,
        )

        axis.set_axis_off()

        axis.set_title(
            (
                f"View {index}\n"
                f"elev={elevation}°, "
                f"azim={azimuth}°"
            ),
            fontsize=9,
        )

    figure.suptitle(
        (
            "Dense Hybrid Reconstruction — "
            "Candidate Project-Page Views"
        ),
        fontsize=16,
        fontweight="bold",
    )

    figure.tight_layout(
        rect=(0, 0, 1, 0.96)
    )

    args.output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        args.output_png,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    print("Input points:", len(xyz))
    print("Saved:", args.output_png)


if __name__ == "__main__":
    main()
