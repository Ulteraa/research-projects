"""Matplotlib application for SAM-assisted and manual COCO annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .coco import CocoDataset, polygons_to_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class AnnotationWindow:
    """One interactive image window backed by a resumable COCO dataset."""

    def __init__(
        self,
        *,
        image_path: Path,
        file_name: str,
        dataset: CocoDataset,
        predictor,
        categories: Sequence[dict],
        min_area: float,
        simplify: float,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button, RadioButtons

        self.plt = plt
        self.Rectangle = Rectangle
        self.image_path = image_path
        self.image = np.asarray(Image.open(image_path).convert("RGB"))
        self.height, self.width = self.image.shape[:2]
        self.dataset = dataset
        self.image_id = dataset.add_image(file_name, self.width, self.height)
        self.predictor = predictor
        self.categories = list(categories)
        self.min_area = min_area
        self.simplify = simplify

        self.mode = "manual" if predictor is None else "box"
        self.candidate_mask: np.ndarray | None = None
        self.candidate_source = "manual"
        self.point_coords: list[list[float]] = []
        self.point_labels: list[int] = []
        self.manual_points: list[list[float]] = []
        self.drag_start: tuple[float, float] | None = None
        self.drag_patch = None
        self.prompt_artists = []

        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.figure.canvas.manager.set_window_title(f"SAM annotation - {file_name}")
        self.figure.subplots_adjust(bottom=0.16, right=0.80)
        self.image_artist = self.ax.imshow(self.image)
        self.ax.set_axis_off()
        self.status = self.figure.text(0.02, 0.01, "", fontsize=9)

        if predictor is not None:
            predictor.set_image(self.image)

        labels = [f"{item['id']}: {item['name']}" for item in self.categories]
        category_ax = self.figure.add_axes([0.82, 0.48, 0.16, min(0.4, 0.06 * len(labels) + 0.08)])
        self.category_radio = RadioButtons(category_ax, labels, active=0)
        category_ax.set_title("Category", fontsize=10)

        button_specs = [
            ("box", self.activate_box),
            ("point +", self.activate_positive_point),
            ("point -", self.activate_negative_point),
            ("manual", self.activate_manual),
            ("erase", self.activate_erase),
            ("clear", self.clear_candidate),
            ("undo", self.undo),
            ("save", self.save_candidate),
            ("next", self.next_image),
        ]
        self.buttons = []
        for index, (label, callback) in enumerate(button_specs):
            button_ax = self.figure.add_axes([0.015 + index * 0.108, 0.055, 0.098, 0.055])
            button = Button(button_ax, label)
            button.on_clicked(callback)
            self.buttons.append(button)

        self.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.figure.canvas.mpl_connect("button_release_event", self.on_release)
        self._refresh(f"Mode: {self.mode}")

    def show(self) -> None:
        self.plt.show()

    def activate_box(self, _event=None) -> None:
        if self._require_predictor():
            self._reset_prompt()
            self.mode = "box"
            self._refresh("Box mode: drag around an object")

    def activate_positive_point(self, _event=None) -> None:
        if self._require_predictor():
            self.mode = "point_positive"
            self._refresh("Positive point mode: click on the target")

    def activate_negative_point(self, _event=None) -> None:
        if self._require_predictor():
            self.mode = "point_negative"
            self._refresh("Negative point mode: click background to exclude it")

    def activate_manual(self, _event=None) -> None:
        self._reset_prompt()
        self.mode = "manual"
        self._refresh("Manual mode: click at least three boundary vertices")

    def activate_erase(self, _event=None) -> None:
        self._reset_prompt()
        self.mode = "erase"
        self._refresh("Erase mode: drag a box over saved mask regions")

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = float(np.clip(event.xdata, 0, self.width - 1))
        y = float(np.clip(event.ydata, 0, self.height - 1))
        if self.mode in {"point_positive", "point_negative"}:
            self.point_coords.append([x, y])
            self.point_labels.append(1 if self.mode == "point_positive" else 0)
            self._predict_points()
        elif self.mode == "manual":
            self.manual_points.append([x, y])
            if len(self.manual_points) >= 3:
                flattened = np.asarray(self.manual_points).reshape(-1).tolist()
                self.candidate_mask = polygons_to_mask([flattened], self.height, self.width)
                self.candidate_source = "manual"
            self._refresh(f"Manual vertices: {len(self.manual_points)}")
        elif self.mode in {"box", "erase"}:
            self.drag_start = (x, y)
            self._remove_drag_patch()
            self.drag_patch = self.Rectangle((x, y), 0, 0, fill=False, color="red", linewidth=1.5)
            self.ax.add_patch(self.drag_patch)

    def on_motion(self, event) -> None:
        if self.drag_start is None or self.drag_patch is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = self.drag_start
        self.drag_patch.set_x(min(x0, event.xdata))
        self.drag_patch.set_y(min(y0, event.ydata))
        self.drag_patch.set_width(abs(event.xdata - x0))
        self.drag_patch.set_height(abs(event.ydata - y0))
        self.figure.canvas.draw_idle()

    def on_release(self, event) -> None:
        if self.drag_start is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            self.drag_start = None
            self._remove_drag_patch()
            return
        x0, y0 = self.drag_start
        x1 = float(np.clip(event.xdata, 0, self.width))
        y1 = float(np.clip(event.ydata, 0, self.height))
        self.drag_start = None
        self._remove_drag_patch()
        if abs(x1 - x0) < 3 or abs(y1 - y0) < 3:
            self._refresh("Box was too small")
            return
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
        if self.mode == "box":
            masks, _scores, _logits = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=False
            )
            self.candidate_mask = masks[0].astype(bool)
            self.candidate_source = "sam_box"
            self._refresh("SAM box mask ready; Save to commit")
        elif self.mode == "erase":
            changed = self.dataset.erase_box(
                image_id=self.image_id,
                box_xyxy=box,
                min_area=self.min_area,
                simplify=self.simplify,
            )
            if changed:
                self.dataset.save()
            self._refresh(f"Erase updated {changed} saved annotation(s)")

    def _predict_points(self) -> None:
        coordinates = np.asarray(self.point_coords, dtype=np.float32)
        labels = np.asarray(self.point_labels, dtype=np.int32)
        masks, scores, _logits = self.predictor.predict(
            point_coords=coordinates, point_labels=labels, multimask_output=True
        )
        self.candidate_mask = masks[int(np.argmax(scores))].astype(bool)
        self.candidate_source = "sam_points"
        self._refresh(f"SAM point mask ready from {len(self.point_coords)} prompt(s)")

    def save_candidate(self, _event=None) -> None:
        if self.candidate_mask is None or not self.candidate_mask.any():
            self._refresh("Nothing to save")
            return
        category_id = self._selected_category_id()
        try:
            annotation_id = self.dataset.add_mask(
                image_id=self.image_id,
                category_id=category_id,
                mask=self.candidate_mask,
                source=self.candidate_source,
                min_area=self.min_area,
                simplify=self.simplify,
            )
        except ValueError as error:
            self._refresh(str(error))
            return
        self.dataset.save()
        self._reset_prompt()
        self._refresh(f"Saved annotation {annotation_id}")

    def undo(self, _event=None) -> None:
        removed = self.dataset.remove_last_annotation(self.image_id)
        if removed is not None:
            self.dataset.save()
            self._refresh(f"Removed annotation {removed['id']}")
        else:
            self._refresh("No saved annotation to undo")

    def clear_candidate(self, _event=None) -> None:
        self._reset_prompt()
        self._refresh(f"Cleared current prompt; mode: {self.mode}")

    def next_image(self, _event=None) -> None:
        self.dataset.save()
        self.plt.close(self.figure)

    def _refresh(self, message: str) -> None:
        canvas = self.image.astype(np.float32).copy()
        colors = np.asarray(
            [[0, 180, 255], [255, 120, 0], [60, 200, 80], [180, 80, 255]],
            dtype=np.float32,
        )
        for index, annotation in enumerate(self.dataset.annotations_for_image(self.image_id)):
            mask = self.dataset.annotation_mask(annotation)
            canvas[mask] = 0.58 * canvas[mask] + 0.42 * colors[index % len(colors)]
        if self.candidate_mask is not None:
            mask = self.candidate_mask.astype(bool)
            canvas[mask] = 0.55 * canvas[mask] + 0.45 * np.asarray([255, 230, 0])
        self.image_artist.set_data(np.clip(canvas, 0, 255).astype(np.uint8))
        self._draw_prompts()
        count = len(self.dataset.annotations_for_image(self.image_id))
        self.status.set_text(f"{message} | saved objects in image: {count}")
        self.figure.canvas.draw_idle()

    def _draw_prompts(self) -> None:
        for artist in self.prompt_artists:
            artist.remove()
        self.prompt_artists = []
        if self.point_coords:
            points = np.asarray(self.point_coords)
            labels = np.asarray(self.point_labels)
            for value, color in ((1, "lime"), (0, "red")):
                selected = points[labels == value]
                if len(selected):
                    self.prompt_artists.append(
                        self.ax.scatter(selected[:, 0], selected[:, 1], c=color, s=28, marker="o")
                    )
        if self.manual_points:
            points = np.asarray(self.manual_points)
            line, = self.ax.plot(points[:, 0], points[:, 1], "r.-", linewidth=1, markersize=4)
            self.prompt_artists.append(line)

    def _reset_prompt(self) -> None:
        self.candidate_mask = None
        self.point_coords.clear()
        self.point_labels.clear()
        self.manual_points.clear()
        self.drag_start = None
        self._remove_drag_patch()

    def _remove_drag_patch(self) -> None:
        if self.drag_patch is not None:
            self.drag_patch.remove()
            self.drag_patch = None

    def _selected_category_id(self) -> int:
        label = self.category_radio.value_selected
        return int(label.split(":", 1)[0])

    def _require_predictor(self) -> bool:
        if self.predictor is None:
            self._refresh("SAM is disabled in --manual-only mode")
            return False
        return True


def build_predictor(checkpoint: Path, model_type: str, device: str):
    """Load SAM once and return a predictor."""
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=torch.device(device))
    model.eval()
    print(f"Loaded {model_type} on {device}")
    return SamPredictor(model)


def load_categories(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = data.get("categories") if isinstance(data, dict) else data
    if not isinstance(categories, list):
        raise ValueError("Category file must be a list or contain a 'categories' list")
    return categories


def find_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Image directory")
    parser.add_argument("--output", required=True, type=Path, help="COCO JSON output")
    parser.add_argument(
        "--categories",
        type=Path,
        default=Path("configs/categories.example.json"),
        help="JSON category definition",
    )
    parser.add_argument("--checkpoint", type=Path, help="SAM checkpoint")
    parser.add_argument("--model-type", choices=["vit_h", "vit_l", "vit_b"], default="vit_h")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--manual-only", action="store_true", help="Run without loading SAM")
    parser.add_argument("--recursive", action="store_true", help="Search image subdirectories")
    parser.add_argument("--min-area", type=float, default=10.0, help="Minimum contour area")
    parser.add_argument("--simplify", type=float, default=1.0, help="Polygon simplification in pixels")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.images.is_dir():
        raise SystemExit(f"Image directory does not exist: {args.images}")
    categories = load_categories(args.categories)
    images = find_images(args.images, args.recursive)
    if not images:
        raise SystemExit(f"No supported images found in {args.images}")

    predictor = None
    if not args.manual_only:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise SystemExit("--checkpoint is required unless --manual-only is used")
        predictor = build_predictor(args.checkpoint, args.model_type, args.device)

    dataset = CocoDataset.open(args.output, categories)
    for index, image_path in enumerate(images, start=1):
        file_name = image_path.relative_to(args.images).as_posix()
        print(f"[{index}/{len(images)}] {file_name}")
        window = AnnotationWindow(
            image_path=image_path,
            file_name=file_name,
            dataset=dataset,
            predictor=predictor,
            categories=categories,
            min_area=args.min_area,
            simplify=args.simplify,
        )
        window.show()
    dataset.save()
    print(f"Saved {len(dataset.data['annotations'])} annotations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
