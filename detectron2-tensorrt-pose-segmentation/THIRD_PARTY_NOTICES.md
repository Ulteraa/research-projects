# Third-party notices

This project contains or reconstructs code from the following projects.

## Detectron2

- Source: <https://github.com/facebookresearch/detectron2>
- Pinned commit: `80307d2d5e06f06a8a677cc2653f23a4c56402ac`
- License: Apache License 2.0
- Use: reconstructed framework base, deployment exporter, and keypoint tracing
  patch target.

## NVIDIA TensorRT Detectron2 sample

- Source: <https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/python/detectron2>
- Pinned commit: `a0215c1a16c6413c7ac566a498871a4fa36f6f62`
- License: Apache License 2.0
- Use: graph-surgery helpers, TensorRT engine builder, CUDA utilities, image
  batching, inference, visualization, and evaluation foundations. Derived files
  retain NVIDIA's SPDX notice.

## SwinT_detectron2

- Source: <https://github.com/xiaohu2015/SwinT_detectron2>
- Pinned commit: `69d0a2ca47821934907c52621aac64dde8fa8257`
- License: MIT
- Use: Swin-T Detectron2 backbone and configuration under `deployment/swint/`.
  The MIT text is included at `deployment/swint/LICENSE`.

The Swin-T backbone notes that it was modified from Microsoft's
[Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)
implementation, also released under the MIT License.

No third-party checkpoints or datasets are redistributed by this folder.
