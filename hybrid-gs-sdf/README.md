# Hybrid Geometry-Aware Gaussian Splatting

A research prototype that combines the official [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) training pipeline with an image-conditioned signed distance function (SDF) branch.

The implementation keeps Gaussian Splatting as the rendering representation and adds:

- a trainable multi-view image encoder;
- a latent-conditioned SDF decoder;
- point-cloud surface and approximate outside-point supervision;
- an Eikonal regularizer; and
- a zero-level-set consistency loss on Gaussian centers.

> **Project status:** implementation-focused research prototype. The unified trainer has run successfully and produced qualitatively plausible GS-side novel views. Controlled GS-only baselines and ablation studies have not yet been completed, so this repository does not claim that the SDF branch quantitatively improves the baseline.

## Architecture

```text
Training images ──> Multi-view encoder ──> Scene latent ──> SDF decoder
      │                                                   │
      └────────> Official GS renderer <── Gaussian centers┘
                         │
                  Novel-view image
```

The two branches are trained in one optimization loop. The standard GS photometric objective trains the rendering branch. Point-cloud and geometric consistency objectives train the SDF branch and couple it to the Gaussian centers.

## Repository contents

- `train_unified_world_model.py`: unified GS + SDF training entry point.
- `requirements-extension.txt`: dependency added beyond the upstream Gaussian Splatting environment.

## Requirements

This file is designed to run **inside the root of the official Graphdeco Gaussian Splatting repository**. It imports `Scene`, `GaussianModel`, the renderer, argument classes, and utility functions from that project.

You need:

- Linux
- an NVIDIA CUDA-capable GPU
- the CUDA/PyTorch environment supported by the selected upstream commit
- the official Gaussian Splatting repository and its submodules
- `plyfile` for PLY point-cloud supervision

Because upstream APIs can change, record the exact Gaussian Splatting commit used for an experiment:

```bash
git -C gaussian-splatting rev-parse HEAD
```

## Installation

First install the official implementation according to its instructions:

```bash
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
# Create and activate the upstream environment as documented by Graphdeco.
```

Then copy this trainer into the upstream repository root and install the extension dependency:

```bash
cp /path/to/hybrid-gs-sdf/train_unified_world_model.py .
python -m pip install -r /path/to/hybrid-gs-sdf/requirements-extension.txt
```

## Data

The `-s` source directory must follow the input format supported by the upstream Gaussian Splatting implementation.

For SDF supervision, provide a PLY point cloud containing vertex properties named `x`, `y`, and `z`. The point cloud and Gaussian scene must use the same coordinate system and scale. A CO3D-derived point cloud was used during development, but the loader itself only requires those PLY vertex fields.

Datasets, pretrained weights, and generated checkpoints are not included.

## Training

From the official Gaussian Splatting repository root:

```bash
python train_unified_world_model.py \
  -s /path/to/scene \
  -m ./output/unified_v1 \
  --pointcloud_path /path/to/pointcloud.ply
```

A short smoke run can be requested with upstream optimization arguments such as:

```bash
python train_unified_world_model.py \
  -s /path/to/scene \
  -m ./output/smoke_test \
  --pointcloud_path /path/to/pointcloud.ply \
  --iterations 10 \
  --sdf_start_iter 1 \
  --consistency_start_iter 5 \
  --test_iterations 10 \
  --save_iterations 10
```

This confirms code-path integration; it is not a meaningful training or evaluation run.

### Main extension arguments

| Argument | Default | Purpose |
|---|---:|---|
| `--pointcloud_path` | `None` | PLY point cloud for SDF supervision |
| `--encoder_views` | 4 | Views sampled for the scene encoder |
| `--latent_dim` | 256 | Scene-latent dimension |
| `--sdf_start_iter` | 1000 | Start of SDF supervision |
| `--consistency_start_iter` | 3000 | Start of GS–SDF coupling |
| `--lambda_sdf_surface` | 1.0 | Surface zero-level loss weight |
| `--lambda_sdf_outside` | 0.2 | Outside-point loss weight |
| `--lambda_eikonal` | 0.05 | Eikonal loss weight |
| `--lambda_gaussian_surface` | 0.5 | Gaussian-center consistency weight |

Run `python train_unified_world_model.py --help` for the complete set of upstream and extension arguments.

## Outputs

The output directory contains the regular Gaussian Splatting scene saves and unified checkpoints named `unified_ckpt_<iteration>.pth`. A unified checkpoint stores the encoder, SDF decoder, auxiliary optimizer state, and iteration.

## Current limitations

- No controlled GS-only comparison or loss ablation is included.
- Evaluation currently reports L1 and PSNR on a small subset of training cameras, not held-out test performance.
- Approximate outside samples are inferred from distance to a finite point cloud; they are not guaranteed signed-distance labels.
- The SDF branch conditions on a pooled scene latent and does not directly modify the official renderer internals.
- Training currently assumes CUDA.
- Compatibility depends on the upstream Gaussian Splatting API version.

## Attribution and licensing

This project depends on and imports code interfaces from the official Graphdeco Gaussian Splatting implementation. Review and follow the [upstream license](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md), especially its restrictions, before using or redistributing the combined system.

No dataset or upstream source code is redistributed in this repository. Add a license for the original extension code only after confirming that it is compatible with the upstream project and your intended use.

## Responsible project claim

This repository demonstrates architecture design and implementation of a joint GS–SDF training prototype. Its current evidence supports successful integration and qualitative inspection—not a claim of state-of-the-art performance or measured improvement over Gaussian Splatting.


