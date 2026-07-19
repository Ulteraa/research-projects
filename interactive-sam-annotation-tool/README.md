# Interactive SAM Annotation Tool

A desktop research tool for creating COCO polygon annotations with the
[Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything).
Annotators can prompt SAM with boxes or positive/negative points, draw polygons
manually, revise saved masks, and inspect the resulting dataset without a
Detectron2 installation.

This folder is a curated, runnable reconstruction of the prototype described in
Fariborz Taherkhani's *Interactive Annotation and Segmentation Tool Using
Segment Anything Model*. The accompanying report and archived implementation
are available through [Zenodo (DOI 10.5281/zenodo.14630607)](https://doi.org/10.5281/zenodo.14630607).

> **Validation status:** working research prototype. The data writer, validator,
> and renderer have automated tests. The supplied report shows qualitative
> examples but contains no annotation-time benchmark, inter-annotator study, or
> quantitative accuracy comparison; this project therefore makes no speed or
> quality-improvement claim.

## What is included

- Box-prompted SAM segmentation.
- Mixed positive and negative point prompts.
- Manual polygon annotation without loading SAM.
- Multiple configurable COCO categories.
- Erase-region editing that updates the saved polygon geometry.
- Image-scoped undo, candidate clearing, atomic saves, and resumable sessions.
- COCO validation and dependency-light Pillow visualization commands.
- Correct integer IDs, image dimensions, mask-pixel area, and XYWH boxes.

| Control | Action |
| --- | --- |
| `box` | Drag around an object to produce a SAM mask. |
| `point +` | Add a foreground prompt. Multiple point prompts accumulate. |
| `point -` | Add a background prompt to exclude an area. |
| `manual` | Click at least three boundary vertices to form a polygon. |
| `erase` | Drag over saved masks; affected COCO polygons are recomputed. |
| `clear` | Discard the current unsaved mask and prompt points. |
| `undo` | Remove the latest saved annotation for the current image. |
| `save` | Commit the candidate mask and atomically update the COCO JSON file. |
| `next` | Save the dataset and advance to the next image. |

## Repository layout

```text
configs/                    category definition example
scripts/                    source-checkout entry points
src/sam_annotation_tool/    application, COCO writer, validator, renderer
tests/                      dependency-light unit tests
CHANGESET.md                differences from the uploaded research archive
RESULTS.md                  evidence and validation boundaries
THIRD_PARTY_NOTICES.md      provenance and license notes
```

## Installation

Python 3.9–3.11 is recommended. Create an isolated environment and install a
PyTorch build that matches the machine first. For example, for a CUDA 11.8
system:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
```

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for other CUDA versions or CPU-only installation. A desktop Matplotlib backend
is also required; on many Debian/Ubuntu systems this means installing
`python3-tk`, or installing a Qt backend in the virtual environment.

The SAM dependency is pinned to commit
`7fa17d78c45f4f642faa89f4c0e590c400f74225`, the exact version found in the
uploaded archive. Model weights are not included. Download a checkpoint from
the [official SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints)
and keep it outside Git.

## Configure categories

Copy and edit `configs/categories.example.json`. Category IDs must be unique.
Once an output file exists, the same category configuration must be used when
resuming it, preventing silent changes to category meanings.

```json
{
  "categories": [
    {"id": 1, "name": "person", "supercategory": "object"},
    {"id": 2, "name": "vehicle", "supercategory": "object"}
  ]
}
```

## Annotate images

From an installed checkout:

```bash
sam-annotate \
  --images /path/to/images \
  --output output/annotations.json \
  --categories configs/categories.example.json \
  --checkpoint /path/to/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --device auto \
  --recursive
```

`--model-type` must match the downloaded checkpoint. `--device auto` selects
CUDA when PyTorch reports it available and otherwise uses CPU. CPU inference is
supported by SAM but is substantially slower.

Manual annotation can run without PyTorch, SAM, or a checkpoint after installing
the other requirements:

```bash
sam-annotate \
  --images /path/to/images \
  --output output/annotations.json \
  --categories configs/categories.example.json \
  --manual-only
```

The application writes one resumable COCO file for the full image directory.
Images and annotations already present in that file retain their IDs when a
session restarts.

## Validate and visualize

Validate references, dimensions, IDs, polygons, areas, and boxes. Supplying the
image root additionally checks that every referenced image exists:

```bash
sam-validate-coco output/annotations.json --image-root /path/to/images
```

Render masks, boxes, and category labels without Detectron2:

```bash
sam-visualize-coco output/annotations.json \
  --image-root /path/to/images \
  --output output/preview
```

From a source checkout, the equivalent entry points are
`python scripts/annotate.py`, `python scripts/validate_coco.py`, and
`python scripts/visualize_coco.py`.

## Tests

The dependency-light test suite does not download SAM or a checkpoint:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

The interactive GUI and model inference require a graphical session and the
runtime dependencies described above; they are not exercised by these unit
tests.

## Limitations

- COCO polygon encoding cannot preserve holes. The converter intentionally
  retains external contours; use COCO RLE for hole-fidelity requirements.
- Erasing a region rasterizes and then re-vectorizes affected polygons, which
  can slightly change boundaries according to `--simplify` and `--min-area`.
- This is a single-user desktop application with no concurrent editing or
  dataset version-control layer.
- SAM proposes masks but does not assign semantic categories; the annotator
  remains responsible for category choice and quality control.
- Large images and many saved objects increase redraw cost because overlays are
  rasterized in process.
- The supplied evidence is qualitative. See [RESULTS.md](RESULTS.md) for the
  validation boundary and [CHANGESET.md](CHANGESET.md) for curation details.

## License and provenance

The uploaded project included the MIT text preserved as
[LICENSE-ARCHIVE](LICENSE-ARCHIVE), but that file names Hu Ye and appears
inherited from Swin Transformer material in the archive. It is not presented as
a project-wide license for this reconstruction. The license status of the
original custom annotation script is therefore not unambiguous. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), retain all upstream notices,
and confirm rights with the author before redistribution or commercial reuse.
