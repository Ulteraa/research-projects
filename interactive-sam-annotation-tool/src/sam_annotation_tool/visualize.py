"""Render COCO polygons and boxes without requiring Detectron2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


COLORS = (
    (0, 180, 255, 105),
    (255, 120, 0, 105),
    (60, 200, 80, 105),
    (180, 80, 255, 105),
    (255, 60, 100, 105),
)


def render_document(data: dict, image_root: Path, output: Path, limit: int | None) -> int:
    output.mkdir(parents=True, exist_ok=True)
    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    grouped = defaultdict(list)
    for annotation in data["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)

    count = 0
    font = ImageFont.load_default()
    for image_info in data["images"]:
        if limit is not None and count >= limit:
            break
        source = image_root / image_info["file_name"]
        if not source.is_file():
            print("Skipping missing image:", source)
            continue
        image = Image.open(source).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for index, annotation in enumerate(grouped[int(image_info["id"])]):
            color = COLORS[index % len(COLORS)]
            for polygon in annotation["segmentation"]:
                points = list(zip(polygon[0::2], polygon[1::2]))
                if len(points) >= 3:
                    draw.polygon(points, fill=color, outline=color[:3] + (255,))
            x, y, width, height = annotation["bbox"]
            draw.rectangle((x, y, x + width, y + height), outline=color[:3] + (255,), width=2)
            label = categories.get(int(annotation["category_id"]), str(annotation["category_id"]))
            draw.text((x + 2, max(0, y - 11)), label, fill=color[:3] + (255,), font=font)
        rendered = Image.alpha_composite(image, overlay).convert("RGB")
        destination = output / image_info["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(destination)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path, help="COCO JSON file")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    count = render_document(data, args.image_root, args.output, args.limit)
    print(f"Rendered {count} image(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
