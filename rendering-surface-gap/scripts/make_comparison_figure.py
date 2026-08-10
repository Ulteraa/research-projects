#!/usr/bin/env python3
"""Compose identically sized method outputs into a reviewer-readable grid."""
from pathlib import Path
import argparse

from PIL import Image, ImageOps, ImageDraw


CHANNELS = (("rgb.png", "RGB"), ("root_depth.png", "Depth"),
            ("normal.png", "Normal"), ("proposal_sdf_residual.png", "SDF residual"))


def panel(path, title, size):
    im = ImageOps.fit(Image.open(path).convert("RGB"), size, Image.Resampling.LANCZOS)
    out = Image.new("RGB", (size[0], size[1] + 28), "white"); out.paste(im, (0, 28))
    ImageDraw.Draw(out).text((7, 7), title, fill="black"); return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", action="append", nargs=2, metavar=("NAME", "DIR"), required=True)
    p.add_argument("--output", required=True); p.add_argument("--width", type=int, default=280)
    p.add_argument("--height", type=int, default=210); a = p.parse_args()
    rows = []
    for name, directory in a.method:
        cells = [panel(Path(directory) / f, f"{name}: {label}", (a.width, a.height))
                 for f, label in CHANNELS if (Path(directory) / f).exists()]
        if not cells:
            raise FileNotFoundError(
                f"{directory} contains none of: " + ", ".join(f for f, _ in CHANNELS))
        row = Image.new("RGB", (sum(c.width for c in cells), cells[0].height), "white")
        x = 0
        for cell in cells: row.paste(cell, (x, 0)); x += cell.width
        rows.append(row)
    canvas = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
    y = 0
    for row in rows: canvas.paste(row, (0, y)); y += row.height
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); canvas.save(a.output)


if __name__ == "__main__":
    main()
