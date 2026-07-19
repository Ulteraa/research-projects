# Third-party notices

This project is an extension of the BEVFusion implementation distributed with [OpenMMLab MMDetection3D](https://github.com/open-mmlab/mmdetection3d), which in turn credits the original [MIT-HAN-Lab BEVFusion](https://github.com/mit-han-lab/bevfusion) implementation.

The modified `src/transfusion_head.py` retains its upstream attribution header and adds custom research code for:

- `ReliabilityAwareFuser`; and
- `ObjectRefineTransFusionHead`.

MMDetection3D v1.4.0 is distributed under the Apache License 2.0. See `UPSTREAM_LICENSE` in this folder. Dataset users must also comply with the nuScenes terms and licenses.
