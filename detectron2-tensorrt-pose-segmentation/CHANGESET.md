# Framework changes and archive curation

## Source-base identification

The uploaded ZIP was compared file-by-file against Detectron2 history. Excluding
added research/deployment files, it matches Detectron2 commit
`80307d2d5e06f06a8a677cc2653f23a4c56402ac` with only these edited files:

- root `README.md`;
- `configs/Base-RCNN-FPN.yaml`;
- `detectron2/structures/keypoints.py`;
- `tools/deploy/export_model.py`.

The deployment scripts are derived from NVIDIA's TensorRT `release/8.6`
Detectron2 sample. The uploaded archive also contained Swin-T configuration and
backbone code derived from `xiaohu2015/SwinT_detectron2`.

## Research extension

- `deployment/create_onnx.py` adds a post-NMS keypoint ROIAlign, reconnects the
  traced keypoint head, and exports keypoint heatmaps beside boxes and masks.
- `patches/detectron2-keypoints-onnx.patch` removes two indexed keypoint-score
  assignments from the traced path, matching the uploaded framework edit.
- `deployment/infer.py` decodes the extra output and emits joint detections.
- `deployment/visualize.py` overlays masks, boxes, and all configured points.
- `deployment/eval_coco.py` passes decoded keypoints to Detectron2's evaluator.
- `deployment/swint/` registers the Swin-T/FPN backbone and configuration.

## Reproducibility and correctness cleanup

The curated version also:

- removes hard-coded developer paths, checkpoint names, class counts, dataset
  paths, and forced command-line overrides;
- preserves the archive's two-class/one-keypoint configuration separately;
- generalizes the ONNX output from one heatmap to `K` keypoint heatmaps;
- exports the correctly reshaped keypoint tensor rather than the pre-reshape
  ConvTranspose tensor;
- indexes heatmaps as `[batch, detection, keypoint, y, x]`;
- decodes row/column maxima into `x/y` box coordinates with half-pixel centers;
- derives preprocessing color order and pixel mean from the Detectron2 config;
- uses width and height independently when scaling output coordinates;
- fixes XYXY order passed to Detectron2 `Boxes`;
- pins upstream commits and adds a safe reconstruction script and smoke tests.

## Excluded archive content

Unchanged Detectron2 source, documentation, tests, generated `__pycache__` and
`.pyc` files, sample images, duplicate configs, abandoned TorchScript variants,
hard-coded experimental clients, and dated backup folders were intentionally
excluded. Model weights, datasets, and result artifacts were not present.
