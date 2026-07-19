# Detectron2 TensorRT: Joint Pose Estimation and Instance Segmentation

This research prototype extends NVIDIA's Detectron2 Mask R-CNN TensorRT 8.x
conversion path with a keypoint branch. A joint Detectron2 model is exported to
ONNX, graph surgery reconnects TensorRT-compatible NMS and pyramid ROIAlign
operations for the box, mask, and keypoint heads, and the resulting engine emits
detections, instance masks, and per-keypoint heatmaps.

**Status:** Conversion research prototype with qualitative results in the
accompanying report. The report does not provide a reproducible numeric table
for accuracy, latency, memory, hardware, or PyTorch-versus-TensorRT parity, so
no speedup or accuracy-preservation claim is made here. See
[RESULTS.md](RESULTS.md).

**Paper/report:** [DOI 10.5281/zenodo.14629778](https://doi.org/10.5281/zenodo.14629778)

## Research contribution

NVIDIA's reference TensorRT 8.6 sample converts the box and instance-mask paths
of a standard Detectron2 Mask R-CNN. This project adds:

- a second post-NMS pyramid ROIAlign path for keypoints;
- reconnection of the traced Detectron2 keypoint head;
- a TensorRT output shaped `[batch, detections, keypoints, 28, 28]`;
- heatmap decoding and joint box/mask/keypoint visualization;
- optional COCO evaluation fields for the decoded keypoints;
- Swin-T/FPN configuration and backbone registration.

The uploaded historical archive hard-coded two classes and one keypoint. That
configuration is preserved in
`configs/mask_rcnn_swint_archive_single_keypoint.yaml`. The cleaned conversion
code no longer assumes one keypoint; the main example config uses 17.

## Repository design

The uploaded ZIP bundled more than 1,000 Detectron2 framework, test,
documentation, bytecode, and duplicate experiment files. This folder keeps only
the research extension and pins its actual base:

- Detectron2 commit `80307d2d5e06f06a8a677cc2653f23a4c56402ac`;
- NVIDIA TensorRT release/8.6 sample commit
  `a0215c1a16c6413c7ac566a498871a4fa36f6f62`;
- Swin-T Detectron2 source commit
  `69d0a2ca47821934907c52621aac64dde8fa8257`.

`scripts/setup_local_fork.sh` checks out the exact Detectron2 base, applies the
small ONNX tracing patch, installs it in editable mode, and leaves the extension
code in this folder. See [CHANGESET.md](CHANGESET.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## What is and is not included

Included:

- ONNX export, graph surgery, TensorRT engine building, inference, and COCO
  evaluation scripts;
- Swin-T/FPN registration and cleaned joint-model configs;
- a pinned Detectron2 reconstruction script;
- dependency manifests and CPU-only keypoint-decoder tests.

Not included:

- trained checkpoints or pretrained Swin-T weights;
- COCO or the custom `experiment` dataset;
- complete training logs, numeric evaluation artifacts, or engine benchmarks;
- serialized ONNX/TensorRT engines, which depend on weights, TensorRT, and GPU.

## System requirements

This code uses TensorRT's legacy binding API and the TensorRT 8.x
`EfficientNMS_TRT` and `PyramidROIAlign_TRT` plugins. It is not compatible with
TensorRT 10 without an API and plugin migration.

A practical historical compatibility target is:

- Linux on x86-64 with an NVIDIA GPU;
- Python 3.10;
- CUDA 11.8 and a compatible NVIDIA driver;
- PyTorch 2.0.1 / torchvision 0.15.2 built for CUDA 11.8;
- TensorRT 8.6.x, including its Python bindings and plugins.

The paper and uploaded source did not record a complete environment or hardware
manifest, so treat this as a compatibility target, not a claimed tested matrix.
Keep CUDA, PyTorch, torchvision, TensorRT, and the NVIDIA driver mutually
compatible.

## Installation

Create an isolated environment and install the CUDA-specific PyTorch build
first:

```bash
cd detectron2-tensorrt-pose-segmentation
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install TensorRT 8.6.x and its Python bindings using NVIDIA's installation
method for your CUDA platform. Then install the remaining dependencies and the
pinned local Detectron2 fork:

```bash
python -m pip install -r requirements-tensorrt.txt
./scripts/setup_local_fork.sh
python scripts/check_environment.py
```

The bootstrap target defaults to `.vendor/detectron2`; pass another directory
as its first argument if desired. It refuses to overwrite a non-git path or an
overlapping Detectron2 edit.

The lightweight tests require only NumPy:

```bash
python -m unittest discover -s tests
```

## Model and data preparation

The checkpoint must use Detectron2's standard GeneralizedRCNN FPN, RPN,
StandardROIHeads, mask head, and keypoint head node structure. Its class and
keypoint counts must match the config used at every conversion step.

The main example is a one-class, 17-keypoint joint model:

```text
configs/mask_rcnn_swint_joint.yaml
```

For the historical two-class/one-keypoint checkpoint described by the uploaded
archive, use:

```text
configs/mask_rcnn_swint_archive_single_keypoint.yaml
```

Update dataset names or register a custom COCO-format dataset before training.
A typical Detectron2 training command is:

```bash
python .vendor/detectron2/tools/train_net.py \
  --config-file configs/mask_rcnn_swint_joint.yaml \
  MODEL.WEIGHTS /path/to/swin_tiny_pretrained.pth \
  OUTPUT_DIR output/joint_swint
```

This repository does not provide the pretrained weights, fine-tuned checkpoint,
or dataset needed to execute that command.

## Conversion workflow

Use the same config and checkpoint for all four stages. Prepare a 1344×1344
sample image with at least one visible target instance; tracing an empty output
can omit the ROI branches that graph surgery expects.

### 1. Export Detectron2 to ONNX

```bash
python deployment/export_model.py \
  --config-file configs/mask_rcnn_swint_joint.yaml \
  --sample-image /path/to/sample-1344x1344.jpg \
  --output output/export \
  --export-method tracing \
  --format onnx \
  MODEL.WEIGHTS /path/to/model_final.pth \
  MODEL.DEVICE cuda
```

### 2. Add TensorRT-compatible graph operations

```bash
python deployment/create_onnx.py \
  --exported_onnx output/export/model.onnx \
  --onnx output/joint-converted.onnx \
  --det2_config configs/mask_rcnn_swint_joint.yaml \
  --det2_weights /path/to/model_final.pth \
  --sample_image /path/to/sample-1344x1344.jpg \
  --batch_size 1
```

The converted graph has six outputs: detection count, boxes, scores, classes,
28×28 instance masks, and 28×28 heatmaps for each configured keypoint.

### 3. Build an FP16 TensorRT engine

```bash
python deployment/build_engine.py \
  --onnx output/joint-converted.onnx \
  --engine output/joint-fp16.engine \
  --precision fp16
```

TensorRT engines are normally specific to the TensorRT/CUDA/GPU environment in
which they are built. Build them on the target deployment stack.

### 4. Run joint inference

```bash
python deployment/infer.py \
  --engine output/joint-fp16.engine \
  --input /path/to/images \
  --det2_config configs/mask_rcnn_swint_joint.yaml \
  --output results \
  --labels person
```

Each image produces a visualization and a tab-separated text file. After the
box, score, and class fields, the text row contains flattened `x, y, score`
triples for every keypoint. The keypoint score is the raw heatmap maximum, not a
calibrated probability.

## Evaluation

To populate Detectron2's COCO evaluator with boxes, masks, and keypoints:

```bash
python deployment/eval_coco.py \
  --engine output/joint-fp16.engine \
  --input /path/to/val2017 \
  --det2_config configs/mask_rcnn_swint_joint.yaml \
  --det2_weights /path/to/model_final.pth
```

Run evaluation for both the original PyTorch checkpoint and the converted
engine on exactly the same dataset, resize, thresholds, and class/keypoint
metadata. Record box AP, mask AP, keypoint OKS/AP, end-to-end latency, warm-up,
batch size, peak memory, GPU, driver, CUDA, TensorRT, and precision.

## Limitations

- Conversion depends on traced node names from a historical Detectron2 graph;
  newer framework or model structures may require new graph lookups.
- Input resolution is static and square; the supplied config uses 1344×1344.
- TensorRT batch sizes greater than one were not validated in the supplied
  research evidence.
- The repository decoder supports configurable keypoint counts, but the
  original uploaded experiment config used only one keypoint.
- Visualization plots points only; it does not infer a dataset-specific
  skeleton.
- INT8 calibration is inherited from NVIDIA's upstream sample and was not
  evaluated for this joint keypoint extension.
- Full GPU conversion/inference could not be validated without the missing
  checkpoint, dataset, TensorRT 8.6 stack, and NVIDIA GPU.

## License

The project-level license is [Apache License 2.0](LICENSE). Individual derived
files retain their upstream notices. The Swin-T implementation is MIT-licensed;
see `deployment/swint/LICENSE` and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
