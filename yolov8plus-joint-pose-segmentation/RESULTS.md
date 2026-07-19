# Validation status and evidence

## What was demonstrated

The accompanying YOLOv8+ report documents training on human annotations from
COCO and shows qualitative predictions containing all three outputs from one
forward pass: person detections, instance masks, and 17-keypoint poses. It also
shows training/confidence curves and qualitative outputs from PyTorch, ONNX,
TorchScript, and TensorRT paths.

The implementation archive supplied for this project contains the matching
joint head, combined loss, data parser, validator/metrics, prediction path, and
export changes. The curated overlay passes Python syntax compilation. Full
model construction, training, and export require the dependencies and hardware
described in the README.

## What is not established

The report does not include a numeric comparison table against separate
YOLOv8-pose and YOLOv8-seg baselines. It also does not report a controlled
ablation, end-to-end latency measurements with hardware details, or a complete
set of reproducible checkpoints/logs. Therefore this repository does **not**
claim state-of-the-art accuracy, an accuracy improvement over the baselines, or
a verified real-time speedup.

The defensible result is narrower: this is a working research implementation
of a single-pass, shared-backbone architecture for joint person instance
segmentation and pose estimation, supported by qualitative experimental and
deployment outputs.

## Reference

- Fariborz Taherkhani, *YOLOv8+: A Unified Framework for Joint Pose Estimation
  and Instance Segmentation*, Zenodo, DOI
  [10.5281/zenodo.14630612](https://doi.org/10.5281/zenodo.14630612).
