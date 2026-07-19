#!/usr/bin/env python3
"""Report whether the conversion environment has its required modules."""

from __future__ import annotations

import importlib.util
import sys


MODULES = (
    ("numpy", "NumPy"),
    ("torch", "PyTorch with the matching CUDA build"),
    ("detectron2", "the patched local Detectron2 checkout"),
    ("onnx", "ONNX"),
    ("onnx_graphsurgeon", "ONNX GraphSurgeon"),
    ("cv2", "OpenCV"),
    ("timm", "timm"),
    ("tensorrt", "TensorRT 8.6.x Python bindings"),
    ("cuda", "cuda-python"),
)


def main() -> int:
    print("Python", sys.version.split()[0])
    missing = []
    for module, description in MODULES:
        available = importlib.util.find_spec(module) is not None
        print(f"{'OK' if available else 'MISSING':7} {module:20} {description}")
        if not available:
            missing.append(module)

    if missing:
        print("\nMissing modules:", ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
