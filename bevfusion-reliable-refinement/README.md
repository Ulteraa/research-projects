# Reliable Camera–LiDAR BEV Fusion with Proposal-Level 3D Box Refinement

A research extension of **MMDetection3D / BEVFusion** for multimodal 3D object detection on **nuScenes mini**.

The project studies a practical failure mode: camera–LiDAR fusion can improve semantic confidence without producing the same improvement in 3D localization. Two architectural changes were implemented:

1. **Reliability-aware BEV fusion** — replaces the stock concatenate-and-convolve fuser with a learned spatial gate over projected image-BEV and LiDAR-BEV features.
2. **Proposal-level box refinement** — samples a local fused-BEV patch around every coarse TransFusion proposal and predicts residual corrections for center, height, dimensions, rotation, and velocity.

> **Project status:** working research prototype. The custom modules were integrated into MMDetection3D, trained on nuScenes mini, and used to generate qualitative 3D box/point-cloud visualizations. The first reliability-aware fusion ablation did not outperform the baseline in the recorded six-epoch run. A final quantitative log for the proposal-refinement experiment is not currently included, so this repository does not claim a general improvement over BEVFusion.

## Architecture

```text
Six camera views ──> image backbone + LSS ──> image BEV ──┐
                                                         ├─> ReliabilityAwareFuser
LiDAR sweeps ──> voxel encoder + SECOND/SECFPN ─> LiDAR BEV┘          │
                                                                    v
                                                            fused BEV features
                                                                    │
                                                 TransFusion coarse proposals
                                                                    │
                                    local 5×5 fused-BEV patch sampling per proposal
                                                                    │
                                            ObjectRefineTransFusionHead
                                                                    │
                                residual center / size / yaw / velocity corrections
```

## What is custom

### `ReliabilityAwareFuser`

The custom fuser:

- projects image-BEV and LiDAR-BEV features into a shared hidden space;
- computes a LiDAR occupancy proxy from feature magnitude;
- predicts a learned camera-reliability map;
- predicts a per-cell modality gate; and
- retains a LiDAR residual path as a geometric anchor.

### `ObjectRefineTransFusionHead`

The refinement head:

- starts from normal TransFusion coarse proposals;
- constructs a local sampling grid around each proposal center;
- uses bilinear `grid_sample` on the fused BEV feature map;
- averages the local patch into a proposal-level context feature;
- fuses context with the decoder query feature; and
- predicts residual corrections to box geometry and motion.

With `patch_radius=2`, each proposal uses a **5×5 local BEV neighborhood**.

## Repository contents

| Path | Purpose |
|---|---|
| `src/transfusion_head.py` | Modified BEVFusion head containing `ReliabilityAwareFuser` and `ObjectRefineTransFusionHead` |
| `src/__init__.py` | Exports and registers the custom classes |
| `configs/bevfusion_lidar-cam_reliablefuser_*.py` | Clean config overlay for reliability-aware fusion |
| `configs/bevfusion_lidar-cam_reliablefuser_objectrefine_6e_mini.py` | Six-epoch nuScenes-mini object-refinement experiment overlay |
| `scripts/install_into_mmdet3d.sh` | Copies the modified files into an MMDetection3D checkout and backs up originals |
| `scripts/check_install.py` | Verifies that the custom modules are registered |
| `RESULTS.md` | Recorded ablation results and evidence limitations |
| `assets/README.md` | Recommended encoding for the qualitative demo video |

## Tested software family

The uploaded project files came from an **MMDetection3D 1.4.0** codebase with compatibility constraints equivalent to:

- MMDetection3D 1.4.0;
- MMCV `>=2.0.0rc4,<2.2.0`;
- MMEngine `>=0.8.0,<1.0.0`;
- MMDetection `>=3.0.0rc5,<3.4.0`;
- PyTorch and CUDA versions supported by the chosen MMCV build; and
- `nuscenes-devkit`.

GPU builds are sensitive to the PyTorch/CUDA/MMCV combination. Install PyTorch first, then install OpenMMLab packages with OpenMIM.

## Installation

### 1. Clone MMDetection3D

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.4.0
```

### 2. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the correct PyTorch build for the machine's CUDA version. Then install the OpenMMLab stack:

```bash
pip install -U openmim
mim install "mmengine>=0.8.0,<1.0.0"
mim install "mmcv>=2.0.0rc4,<2.2.0"
mim install "mmdet>=3.0.0rc5,<3.4.0"
pip install -v -e .
pip install -r /path/to/bevfusion-reliable-refinement/requirements-openmmlab.txt
```

### 3. Install this extension into the checkout

From this project folder:

```bash
bash scripts/install_into_mmdet3d.sh /absolute/path/to/mmdetection3d
```

The installer backs up the original `transfusion_head.py` and `__init__.py` before copying the modified files.

### 4. Verify registration

From the MMDetection3D root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python /path/to/bevfusion-reliable-refinement/scripts/check_install.py
```

Expected output:

```text
Custom BEVFusion modules are registered: ReliabilityAwareFuser, ObjectRefineTransFusionHead
```

## Dataset preparation

Download the official **nuScenes mini** dataset and place it under:

```text
mmdetection3d/data/nuscenes/
```

The directory should contain the nuScenes metadata and sensor folders, including `v1.0-mini`, `samples`, and `sweeps`.

From the MMDetection3D root, create the data information files:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini
```

Dataset files are not included in this repository.

## Experiments

Run all commands from the MMDetection3D root after applying the extension.

### A. Upstream joint camera–LiDAR baseline

Use the upstream config as the baseline:

```bash
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --amp \
  --cfg-options \
    train_cfg.max_epochs=6 \
    train_cfg.val_interval=1 \
    train_dataloader.batch_size=1 \
    train_dataloader.num_workers=2 \
    val_dataloader.num_workers=2 \
    test_dataloader.num_workers=2 \
    train_dataloader.dataset.dataset.metainfo.version=v1.0-mini \
    val_dataloader.dataset.metainfo.version=v1.0-mini \
    test_dataloader.dataset.metainfo.version=v1.0-mini
```

### B. Reliability-aware fusion

```bash
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_reliablefuser_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py \
  --amp \
  --cfg-options \
    train_cfg.max_epochs=6 \
    train_cfg.val_interval=1 \
    train_dataloader.batch_size=1 \
    train_dataloader.num_workers=2 \
    val_dataloader.num_workers=2 \
    test_dataloader.num_workers=2 \
    train_dataloader.dataset.dataset.metainfo.version=v1.0-mini \
    val_dataloader.dataset.metainfo.version=v1.0-mini \
    test_dataloader.dataset.metainfo.version=v1.0-mini
```

### C. Reliability-aware fusion + proposal refinement

The cleanest comparison initializes the refinement experiment from a trained reliability-fuser checkpoint while allowing the new refinement parameters to initialize normally:

```bash
python tools/train.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_reliablefuser_objectrefine_6e_mini.py \
  --amp \
  --cfg-options \
    load_from=/absolute/path/to/reliable_fuser/epoch_6.pth
```

A warning about missing `bbox_head.refine_*` parameters is expected when loading a checkpoint that predates the refinement head. The new parameters should be initialized and trained.

For distributed training:

```bash
bash tools/dist_train.sh \
  projects/BEVFusion/configs/bevfusion_lidar-cam_reliablefuser_objectrefine_6e_mini.py \
  2 --amp \
  --cfg-options load_from=/absolute/path/to/reliable_fuser/epoch_6.pth
```

The mini config enables `find_unused_parameters=True`, matching the debugging setup used while integrating the new refinement branch.

## Evaluation

```bash
python tools/test.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_reliablefuser_objectrefine_6e_mini.py \
  /absolute/path/to/epoch_4.pth
```

Track at least:

- mAP;
- NDS;
- mATE;
- mASE;
- mAOE;
- mAVE; and
- per-class AP for common vehicle and pedestrian classes.

## Qualitative visualization

MMDetection3D can save multimodal 3D visualizations during testing:

```bash
python tools/test.py \
  projects/BEVFusion/configs/bevfusion_lidar-cam_reliablefuser_objectrefine_6e_mini.py \
  /absolute/path/to/epoch_4.pth \
  --show-dir work_dirs/objectrefine_visualization
```

Generated frames can be assembled into an MP4 with FFmpeg. See `assets/README.md` for a GitHub-friendly web encoding command.

## Recorded experiment result

The preserved experiment summary reports:

| Variant | mAP | NDS |
|---|---:|---:|
| Joint BEVFusion baseline, six epochs | 0.2578 | 0.2520 |
| Reliability-aware fuser, six epochs | 0.2392 | 0.2415 |

The reliability-aware fuser therefore did **not** beat the baseline in that short run. This motivated targeting geometry more directly with proposal-level box refinement. The raw logs and final numerical result for the refinement experiment are not currently included; see `RESULTS.md`.

## Current limitations

- Experiments were performed on nuScenes mini rather than the full benchmark.
- The reliability signals are feature-derived proxies, not explicit camera depth entropy or raw LiDAR point density.
- The recorded reliability-fuser experiment underperformed the baseline.
- Final proposal-refinement metrics cannot be independently reconstructed without the original logs.
- The config uses a short six-epoch debugging budget and batch size 1.
- Upstream OpenMMLab APIs and compiled CUDA operators are version-sensitive.
- Checkpoints, nuScenes data, generated predictions, logs, and large videos are excluded.

## Recommended next experiments

A stronger follow-up should use matched initialization, training budgets, and multiple seeds for:

1. upstream `ConvFuser` baseline;
2. `ReliabilityAwareFuser`;
3. upstream fuser + `ObjectRefineTransFusionHead`;
4. reliability-aware fuser + object refinement; and
5. refinement-weight and patch-radius ablations.

The most informative ablations are `patch_radius ∈ {1,2,3}`, `refine_weight ∈ {0.25,0.5,1.0}`, and removing center, yaw, or velocity residual branches separately.

## Attribution and licensing

This project modifies code from the MMDetection3D BEVFusion implementation and retains its upstream attribution. See `THIRD_PARTY_NOTICES.md` and `UPSTREAM_LICENSE`.

The repository contains only the custom/modified integration files and lightweight configs. It does not redistribute nuScenes data, checkpoints, or the complete MMDetection3D source tree.

## Responsible project claim

This repository demonstrates architecture design, framework-level integration, controlled ablation, debugging, and qualitative analysis for multimodal 3D detection. It does not claim state-of-the-art performance, production readiness, or a verified general improvement over BEVFusion.
