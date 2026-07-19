#!/usr/bin/env python3
"""Train the YOLOv8+ joint pose and instance-segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML file.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "configs/yolov8n-pose-seg.yaml",
        help="YOLOv8+ model YAML or a compatible checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="For example: 0, 0,1, cpu, or mps.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/pose_segment")
    parser.add_argument("--name", default="train")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model), task="pose_segment")
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "seed": args.seed,
        "exist_ok": True,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
