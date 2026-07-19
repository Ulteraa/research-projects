#!/usr/bin/env python3
"""Export a trained YOLOv8+ model to a deployment format."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--format",
        default="onnx",
        choices=("onnx", "torchscript", "engine"),
    )
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640])
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imgsz: int | tuple[int, int]
    if len(args.imgsz) == 1:
        imgsz = args.imgsz[0]
    elif len(args.imgsz) == 2:
        imgsz = tuple(args.imgsz)
    else:
        raise ValueError("--imgsz accepts one value or height and width.")

    model = YOLO(str(args.model), task="pose_segment")
    export_args = {
        "format": args.format,
        "imgsz": imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
    }
    if args.device is not None:
        export_args["device"] = args.device
    model.export(**export_args)


if __name__ == "__main__":
    main()
