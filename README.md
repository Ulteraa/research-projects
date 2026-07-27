# Research Projects

Selected research prototypes and implementations in computer vision, 3D vision, generative modeling, and machine learning.

These projects emphasize hands-on system design and implementation. Each folder documents its validation status, requirements, usage, and limitations.

## Projects


### [Learning-Guided Dense SLAM for Structured Interior Reconstruction](learning-guided-interior-slam)

A metric and uncertainty-aware interior reconstruction pipeline extending MASt3R-SLAM with RGB-D scale anchoring, TSDF fusion, Manhattan-structure recovery, multi-view opening validation, and selective structural abstention.

**Topics:** Dense SLAM · RGB-D odometry · Metric scale · TSDF fusion · Manhattan structure · Interior reconstruction · Uncertainty and abstention

**Status:** Working TUM Freiburg1 Room/Desk prototype. Ground-truth-free RGB-D scale anchoring reduced repeated TSDF surface RMSE from 4.405 cm to 2.106 cm on Desk, while the unchanged structural-confidence policy accepted Room and abstained on the partial Desk sequence. Dense evaluation reuses the same RGB-D observations and is not independent scan evaluation.



### [Hybrid Exterior Reconstruction](exterior-hybrid-reconstruction)

A hybrid outdoor 3D reconstruction pipeline combining VGGT feed-forward predictions, classical bundle adjustment, dense depth reconstruction, and adaptive multiview consistency filtering.

**Topics:** Multi-view reconstruction · VGGT · COLMAP · Bundle adjustment · Dense point clouds · ETH3D

**Status:** Working ETH3D Courtyard prototype. Bundle adjustment improved camera-pose accuracy, while adaptive filtering improved dense-cloud precision at the cost of some completeness.

### [Hybrid Geometry-Aware Gaussian Splatting](hybrid-gs-sdf)

A joint training prototype combining the official 3D Gaussian Splatting pipeline with an image-conditioned signed distance function (SDF) branch. It explores auxiliary geometric supervision and zero-level-set consistency on Gaussian centers.

**Topics:** Gaussian Splatting · SDF · Novel-view synthesis · Multi-view learning · 3D scene representation

**Status:** Working research prototype with qualitative inspection. Controlled GS-only comparisons and ablation studies remain future work.


### [Offline RL Benchmarking and Stability Analysis with CQL](offline-rl-cql)

A reproducible offline reinforcement learning project on Minari Walker2d, covering CQL training, periodic rollout evaluation, checkpoint selection, hyperparameter sweeps, critic-count ablations, cloud GPU execution, and failure analysis.

**Topics:** Offline RL · CQL · Continuous control · MuJoCo · Minari · d3rlpy · PyTorch

**Status:** Working research prototype. The training and evaluation pipeline runs successfully; increasing critic count did not reliably improve performance, and no universal improvement over standard CQL is claimed.


### [Safe Offline-to-Online RL for Vision-Based Robotic Manipulation](safe-o2o-robomimic)

A RoboMimic research prototype that combines a visual behavior-cloning prior, conservative actor-critic updates, critic-ensemble disagreement, and multi-camera RoboSuite evaluation. The validated implementation covers the offline policy-improvement stage; online adaptation remains future work.

**Topics:** Robot learning · Offline RL · Behavior cloning · RoboMimic · RoboSuite · Multi-camera perception · Uncertainty

**Status:** Working Lift prototype with a measured improvement from 68% BC success to 76% actor success over 50 episodes. The Can extension achieved 0/20 BC success with both one and two cameras, and is documented as an honest negative result requiring sequence modeling and stronger baselines.

### [Reliable Camera–LiDAR BEV Fusion with Proposal-Level 3D Box Refinement](bevfusion-reliable-refinement)

A multimodal 3D detection research prototype extending MMDetection3D/BEVFusion with a reliability-aware camera–LiDAR fusion block and a proposal-level refinement head that samples local fused-BEV context to predict residual corrections for 3D box geometry and motion.

**Topics:** Multimodal 3D detection · BEVFusion · MMDetection3D · Camera–LiDAR fusion · nuScenes · 3D box refinement

**Status:** Working nuScenes-mini research prototype with implemented training, evaluation, and qualitative visualization paths. The reliability-aware fuser did not beat the baseline in the recorded six-epoch ablation, and final quantitative results for the proposal-refinement experiment remain incomplete; no general improvement claim is made.

### [YOLOv8+: Joint Pose Estimation and Instance Segmentation](yolov8plus-joint-pose-segmentation)

A modified Ultralytics YOLOv8 framework that introduces a unified `pose_segment` task, combining person detection, instance masks, and human keypoints in one shared-backbone forward pass. The project includes the framework overlay, joint loss and metrics, annotation tools, reproducible setup, training/inference entry points, and deployment-oriented export support.

**Topics:** Human analysis · Multi-task learning · Pose estimation · Instance segmentation · YOLOv8 · ONNX · TensorRT

**Status:** Working research prototype with qualitative COCO and converted-model outputs. A controlled numeric comparison against separate pose and segmentation baselines is not available, so no accuracy or speedup claim is made.

### [Detectron2 TensorRT: Joint Pose Estimation and Instance Segmentation](detectron2-tensorrt-pose-segmentation)

A TensorRT 8.x conversion prototype for joint Detectron2 Mask R-CNN/Swin-T inference. It extends NVIDIA's box-and-mask graph-surgery path with a post-NMS keypoint ROIAlign branch, configurable keypoint heatmap output, joint decoding and visualization, pinned framework reconstruction, and evaluation support.

**Topics:** Human analysis · Detectron2 · Mask R-CNN · Swin Transformer · Pose estimation · Instance segmentation · ONNX · TensorRT

**Status:** Working research prototype with qualitative before/after conversion evidence. The supplied report contains no reproducible accuracy or latency table, so no speedup or accuracy-preservation claim is made.

### [Interactive SAM Annotation Tool for COCO Segmentation](interactive-sam-annotation-tool)

A curated desktop annotation tool that combines Segment Anything box and positive/negative point prompts with manual polygons, saved-mask editing, resumable COCO export, validation, and dependency-light visualization.

**Topics:** Data annotation · Interactive segmentation · Segment Anything · COCO · Human-in-the-loop vision · Computer vision tooling

**Status:** Working research prototype with qualitative examples and automated COCO writer/renderer tests. The supplied report contains no timing, user-study, or mask-accuracy comparison, so no productivity or quality-improvement claim is made.

## About

This repository is a portfolio of research-oriented implementations. Projects may range from completed systems to clearly labeled experimental prototypes. Performance claims are limited to the evidence documented within each project.
