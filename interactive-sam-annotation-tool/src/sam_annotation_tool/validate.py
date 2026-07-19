"""Validate a COCO polygon annotation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .coco import validate_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path, help="COCO JSON file")
    parser.add_argument("--image-root", type=Path, help="Check that referenced images exist")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = json.loads(args.annotation.read_text(encoding="utf-8"))
    errors, warnings = validate_document(data, args.image_root)
    print(
        f"images={len(data.get('images', []))} "
        f"annotations={len(data.get('annotations', []))} "
        f"categories={len(data.get('categories', []))}"
    )
    for warning in warnings:
        print("WARNING:", warning)
    for error in errors:
        print("ERROR:", error)
    if errors:
        return 1
    print("COCO validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
