from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def list_images(image_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not paths:
        raise RuntimeError(
            f"No supported images found in {image_dir}"
        )

    return paths


def to_scene_array(
    tensor: torch.Tensor,
    number_of_views: int,
    name: str,
) -> np.ndarray:
    array = (
        tensor.detach()
        .float()
        .cpu()
        .numpy()
    )

    # Convert [1, S, ...] to [S, ...].
    if (
        array.ndim >= 2
        and array.shape[0] == 1
        and array.shape[1] == number_of_views
    ):
        array = array[0]

    if array.shape[0] != number_of_views:
        raise RuntimeError(
            f"{name}: expected first dimension "
            f"{number_of_views}, received {array.shape}"
        )

    return array


def normalize_map(
    tensor: torch.Tensor,
    number_of_views: int,
    name: str,
) -> np.ndarray:
    array = to_scene_array(
        tensor,
        number_of_views,
        name,
    )

    # Convert [S, H, W, 1] to [S, H, W].
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]

    if array.ndim != 3:
        raise RuntimeError(
            f"{name}: expected [S,H,W] or [S,H,W,1], "
            f"received {array.shape}"
        )

    return array.astype(np.float32)


def normalize_rgb(
    images: torch.Tensor,
    number_of_views: int,
) -> np.ndarray:
    array = to_scene_array(
        images,
        number_of_views,
        "processed_images",
    )

    if array.ndim != 4:
        raise RuntimeError(
            f"Unexpected processed-image shape: {array.shape}"
        )

    if array.shape[1] == 3:
        array = np.transpose(
            array,
            (0, 2, 3, 1),
        )
    elif array.shape[-1] != 3:
        raise RuntimeError(
            f"Could not identify RGB dimension: {array.shape}"
        )

    return np.clip(array, 0.0, 1.0)


def statistics(
    array: np.ndarray,
    positive_only: bool = False,
) -> dict[str, Any]:
    valid = np.isfinite(array)

    if positive_only:
        valid &= array > 0

    values = array[valid]

    if values.size == 0:
        return {
            "valid_count": 0,
            "minimum": None,
            "p02": None,
            "median": None,
            "p98": None,
            "maximum": None,
        }

    return {
        "valid_count": int(values.size),
        "minimum": float(np.min(values)),
        "p02": float(np.percentile(values, 2)),
        "median": float(np.median(values)),
        "p98": float(np.percentile(values, 98)),
        "maximum": float(np.max(values)),
    }


def save_preview(
    array: np.ndarray,
    output_path: Path,
    positive_only: bool = False,
) -> None:
    array = np.asarray(array, dtype=np.float32)

    valid = np.isfinite(array)

    if positive_only:
        valid &= array > 0

    preview = np.zeros(
        array.shape,
        dtype=np.uint8,
    )

    if np.any(valid):
        values = array[valid]
        lower = float(np.percentile(values, 2))
        upper = float(np.percentile(values, 98))

        if upper <= lower:
            upper = lower + 1e-6

        normalized = np.clip(
            (array - lower) / (upper - lower),
            0.0,
            1.0,
        )

        preview[valid] = (
            normalized[valid] * 255.0
        ).astype(np.uint8)

    Image.fromarray(preview).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        default="facebook/VGGT-1B",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {args.image_dir}"
        )

    image_paths = list_images(args.image_dir)
    number_of_views = len(image_paths)

    print(f"Found {number_of_views} images")
    for index, path in enumerate(image_paths):
        print(f"  {index:02d}: {path.name}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists; "
                "use --overwrite"
            )

        shutil.rmtree(args.output_dir)

    depth_dir = args.output_dir / "depth"
    confidence_dir = args.output_dir / "depth_confidence"
    rgb_dir = args.output_dir / "processed_rgb"
    depth_preview_dir = args.output_dir / "depth_preview"
    confidence_preview_dir = (
        args.output_dir / "confidence_preview"
    )

    for directory in [
        depth_dir,
        confidence_dir,
        rgb_dir,
        depth_preview_dir,
        confidence_preview_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability()

    inference_dtype = (
        torch.bfloat16
        if major >= 8
        else torch.float16
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Inference dtype:", inference_dtype)
    print("Loading:", args.checkpoint)

    model = (
        VGGT.from_pretrained(args.checkpoint)
        .to(device)
        .eval()
    )

    images = load_and_preprocess_images(
        [str(path) for path in image_paths]
    ).to(device)

    print("Processed input shape:", tuple(images.shape))

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=inference_dtype,
        ):
            predictions = model(images)

            extrinsics, intrinsics = (
                pose_encoding_to_extri_intri(
                    predictions["pose_enc"],
                    images.shape[-2:],
                )
            )

    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - start

    print("\nPrediction tensors:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {tuple(value.shape)}")

    if "depth" not in predictions:
        raise KeyError(
            f"'depth' missing; keys={list(predictions)}"
        )

    if "depth_conf" not in predictions:
        raise KeyError(
            f"'depth_conf' missing; keys={list(predictions)}"
        )

    depth = normalize_map(
        predictions["depth"],
        number_of_views,
        "depth",
    )

    confidence = normalize_map(
        predictions["depth_conf"],
        number_of_views,
        "depth_conf",
    )

    pose_encoding = to_scene_array(
        predictions["pose_enc"],
        number_of_views,
        "pose_enc",
    ).astype(np.float32)

    extrinsics_array = to_scene_array(
        extrinsics,
        number_of_views,
        "extrinsics",
    ).astype(np.float32)

    intrinsics_array = to_scene_array(
        intrinsics,
        number_of_views,
        "intrinsics",
    ).astype(np.float32)

    processed_rgb = normalize_rgb(
        images,
        number_of_views,
    )

    if depth.shape != confidence.shape:
        raise RuntimeError(
            f"Depth/confidence mismatch: "
            f"{depth.shape} versus {confidence.shape}"
        )

    view_reports = []

    for index, image_path in enumerate(image_paths):
        stem = image_path.stem

        np.save(
            depth_dir / f"{stem}.npy",
            depth[index],
        )

        np.save(
            confidence_dir / f"{stem}.npy",
            confidence[index],
        )

        rgb_uint8 = (
            processed_rgb[index] * 255.0
        ).round().astype(np.uint8)

        Image.fromarray(rgb_uint8).save(
            rgb_dir / f"{stem}.png"
        )

        save_preview(
            depth[index],
            depth_preview_dir / f"{stem}.png",
            positive_only=True,
        )

        save_preview(
            confidence[index],
            confidence_preview_dir / f"{stem}.png",
        )

        with Image.open(image_path) as original:
            original_width, original_height = original.size

        view_reports.append(
            {
                "index": index,
                "image_name": image_path.name,
                "original_size": [
                    original_width,
                    original_height,
                ],
                "processed_size": [
                    int(depth.shape[2]),
                    int(depth.shape[1]),
                ],
                "depth_statistics": statistics(
                    depth[index],
                    positive_only=True,
                ),
                "confidence_statistics": statistics(
                    confidence[index],
                ),
            }
        )

    np.savez_compressed(
        args.output_dir / "cameras.npz",
        image_names=np.asarray(
            [path.name for path in image_paths]
        ),
        pose_encoding=pose_encoding,
        extrinsics_camera_from_world=extrinsics_array,
        intrinsics=intrinsics_array,
    )

    metadata = {
        "milestone": "Dense Hybrid Fusion 2A",
        "checkpoint": args.checkpoint,
        "image_directory": str(
            args.image_dir.resolve()
        ),
        "image_count": number_of_views,
        "runtime_seconds": runtime_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "inference_dtype": str(inference_dtype),
        "depth_shape": list(depth.shape),
        "confidence_shape": list(confidence.shape),
        "pose_encoding_shape": list(
            pose_encoding.shape
        ),
        "extrinsics_shape": list(
            extrinsics_array.shape
        ),
        "intrinsics_shape": list(
            intrinsics_array.shape
        ),
        "camera_convention": (
            "OpenCV camera-from-world: "
            "x right, y down, z forward"
        ),
        "views": view_reports,
    }

    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nExport completed")
    print(f"Runtime: {runtime_seconds:.2f} seconds")
    print("Depth shape:", depth.shape)
    print("Confidence shape:", confidence.shape)
    print("Extrinsics shape:", extrinsics_array.shape)
    print("Intrinsics shape:", intrinsics_array.shape)
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
