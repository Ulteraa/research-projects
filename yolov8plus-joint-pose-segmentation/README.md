# YOLOv8+: Joint Pose Estimation and Instance Segmentation

YOLOv8+ is a research framework that extends Ultralytics YOLOv8 with a unified
`pose_segment` task. A shared backbone and feature pyramid feed one
`PoseSegment` head that predicts person boxes/classes, instance-mask
coefficients and prototypes, and human keypoints in a single forward pass.

**Status:** Working research prototype with documented qualitative training and
deployment outputs. The available report does not contain a controlled numeric
comparison against separate pose and segmentation baselines, so no accuracy or
speedup claim is made here. See [RESULTS.md](RESULTS.md).

**Paper/report:** [DOI 10.5281/zenodo.14630612](https://doi.org/10.5281/zenodo.14630612)

## Why this project matters

Standard YOLOv8 exposes pose and instance segmentation as separate tasks. This
prototype explores a shared computation path for human analysis:

- one backbone/neck instead of two independent models;
- a joint head for detection, mask coefficients/prototypes, and keypoints;
- one combined optimization objective;
- unified prediction, validation, and export paths;
- ONNX, TorchScript, and TensorRT-oriented deployment support.

The research question is whether shared features and a single pass can simplify
a perception pipeline without sacrificing task quality. The current evidence
demonstrates feasibility; a controlled baseline/ablation remains future work.

## Repository design

This folder stores only the files changed from Ultralytics v8.2.74 under
`overlay/ultralytics/`. The setup script checks out the exact upstream version,
applies the overlay, and installs the resulting local fork. This makes the
research contribution easier to inspect than committing hundreds of unchanged
upstream files.

See [CHANGESET.md](CHANGESET.md) for the framework integration points and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Installation

Python 3.9-3.11 is recommended. Create an isolated environment:

```bash
cd yolov8plus-joint-pose-segmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For an NVIDIA GPU, install the PyTorch/torchvision build matching your CUDA
runtime first. Then reconstruct and install the local fork:

```bash
./scripts/setup_local_fork.sh
```

The script pins the upstream source to tag `v8.2.74` and commit
`9f593318542e9fc38de6b30b070673104a3c6f28`, copies the overlay, installs it in
editable mode, and verifies that `PoseSegment` is importable.

The lightweight annotation-tool smoke test does not require model weights or a
GPU:

```bash
python -m unittest discover -s tests
```

## Dataset layout and annotation format

Copy `configs/dataset.example.yaml` and update its dataset root. The expected
layout is:

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image has one `.txt` label file with one object per line:

```text
id: <class> bbox: <cx> <cy> <w> <h> keypoints: <x y v> ... segmentations: <x1 y1> <x2 y2> ...
```

Class IDs are zero-based. Box, keypoint x/y, and polygon coordinates are
normalized to `[0, 1]`. Visibility is `0`, `1`, or `2`. The number of keypoint
triples must match `kpt_shape` in the dataset YAML; the supplied configuration
uses the 17 COCO human keypoints.

Convert polygon-based COCO annotations and validate the result with:

```bash
python tools/convert_coco_joint.py \
  --coco-json /path/to/person_keypoints_train.json \
  --output-dir /path/to/data/labels/train

python tools/validate_joint_labels.py \
  --labels /path/to/data/labels/train \
  --keypoints 17 \
  --classes 1
```

The converter selects the largest polygon when a COCO instance contains
multiple polygons and skips RLE/crowd or incomplete records by default. Inspect
the printed skipped-instance count before training.

## Training

Start with the nano-scale joint model and a modest image size, then scale based
on GPU memory:

```bash
python scripts/train.py \
  --data configs/dataset.example.yaml \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

Outputs are written to `runs/pose_segment/train/`. Larger inputs such as the
1920-pixel setting used in the original experimentation require substantially
more GPU memory; begin at 640 for a reproducibility check.

## Inference

```bash
python scripts/predict.py \
  --model runs/pose_segment/train/weights/best.pt \
  --source /path/to/images \
  --imgsz 640 \
  --device 0
```

The prediction path returns and plots boxes, instance masks, and keypoints.

## Export

Install the optional ONNX dependencies when needed:

```bash
python -m pip install -r requirements-export.txt
python scripts/export.py \
  --model runs/pose_segment/train/weights/best.pt \
  --format onnx \
  --imgsz 640
```

Use `--format torchscript` for TorchScript. TensorRT export requires a compatible
NVIDIA driver, CUDA, TensorRT installation, GPU device, and `--format engine`.
Export compatibility is tied to this historical Ultralytics base; use the exact
version reconstructed by the setup script.

## Reproducibility checklist

- Record the dataset release, split, category mapping, and skipped annotations.
- Save the full training arguments, random seed, checkpoint, and curves.
- Report box mAP, mask mAP, and pose OKS/AP separately.
- Benchmark against separate YOLOv8-pose and YOLOv8-seg models using the same
  image size, data split, hardware, precision, and batch size.
- Measure end-to-end latency and peak memory for PyTorch and each exported
  runtime.

## Limitations

- The custom parser accepts polygon masks, not COCO RLE masks.
- One polygon is retained per instance by the supplied converter.
- Joint training increases output/loss complexity and may require loss-weight
  tuning.
- The implementation is based on Ultralytics v8.2.74 and should not be mixed
  blindly with newer releases.
- Checkpoints, datasets, and full experiment logs are not included.

## License

This modified Ultralytics-based implementation is distributed under the
[GNU Affero General Public License v3.0](LICENSE). Review the upstream license
and your deployment obligations before commercial use.
