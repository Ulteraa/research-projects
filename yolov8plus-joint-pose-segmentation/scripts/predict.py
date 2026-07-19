#!/usr/bin/env python3
"""Run YOLOv8+ joint mask and keypoint inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained .pt or exported model.")
    parser.add_argument("--source", required=True, help="Image, directory, video, or stream source.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=ROOT / "runs/predict")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model), task="pose_segment")
    predict_args = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "project": str(args.project),
        "name": args.name,
        "save": True,
        "exist_ok": True,
    }
    if args.device is not None:
        predict_args["device"] = args.device
    model.predict(**predict_args)


if __name__ == "__main__":
    main()
