# Learning-Guided Dense SLAM for Structured Interior Reconstruction

A research and systems prototype that extends a learned visual-SLAM backend into a metric RGB-D reconstruction pipeline with scale diagnosis, TSDF fusion, Manhattan-structure recovery, visibility-aware opening validation, and evidence-based structural abstention.

**Project page:** https://ulteraa.github.io/projects/learning-guided-interior-slam.html  
**Full experiment bundle:** https://ulteraa.github.io/downloads/interior-slam-v1.tar.gz

## Why this project

Learned SLAM can recover camera motion and dense scene geometry, but its output is not automatically suitable for measurement, floor-plan recovery, or design-oriented modeling. A usable interior reconstruction stack must also handle:

- scene-dependent metric-scale error;
- pose-induced dense-fusion distortion;
- clutter and furniture planes competing with architectural boundaries;
- partially observed walls and ceilings;
- false opening hypotheses caused by occlusion or missing depth;
- uncertainty cases where the system should abstain instead of hallucinating a room.

This project treats those issues as system-level responsibilities rather than assuming that a successful SLAM trajectory is equivalent to a reliable structured model.

## System overview

```text
RGB-D sequence
    ↓
MASt3R-SLAM trajectory and dense reconstruction
    ↓
SE(3) / Sim(3) trajectory diagnosis
    ↓
Ground-truth-free RGB-D translation-scale anchoring
    ↓
Metric TSDF fusion and repeated surface evaluation
    ↓
Floor, ceiling, and vertical-plane extraction
    ↓
Manhattan-frame and perimeter reasoning
    ↓
Occupancy opening proposals + multi-view depth validation
    ↓
Conservative / hypothesis-aware structured model export
    ↓
Coverage confidence assessment: accept or abstain
```

## Verified results

### TUM Freiburg1 Desk: metric-scale diagnosis and correction

The learned trajectory exhibited approximately 11% translation-scale compression. A scale estimate derived from RGB-D odometry—without using ground-truth poses for estimation—recovered a median scale of **1.09905**. The ground-truth-derived diagnostic reference was **1.11323**, an absolute difference of **0.01417**.

Across five deterministic sampled-surface evaluations:

| Pose / scale variant | Surface RMSE | Surface within 2 cm |
|---|---:|---:|
| Estimated SE(3) | **4.405 ± 0.015 cm** | **33.37%** |
| Ground-truth-derived Sim(3) diagnostic | **2.267 ± 0.074 cm** | **69.83%** |
| Deployable RGB-D scale anchor | **2.106 ± 0.008 cm** | **71.04%** |

The RGB-D scale anchor reduced measured TSDF surface RMSE by **52.2%** relative to the uncorrected estimated-pose fusion.

### TUM Freiburg1 Room: structured reconstruction

The refined architectural envelope measured approximately:

- **5.599 m × 4.347 m × 2.898 m**;
- **24.34 m²** floor area;
- **70.54 m³** conservative watertight volume.

The opening-analysis stage retained one geometry-supported probable window hypothesis on the `y_min` wall. It is intentionally labeled as a hypothesis: no RGB semantic classifier was used to confirm a window category, and no door was confirmed.

### Selective structural reasoning

The same coverage-confidence policy produced different decisions on the two sequences:

| Sequence | Decision | Interpretation |
|---|---|---|
| Freiburg1 Room | `FULL_3D_ROOM_ENVELOPE_SUPPORTED` | Three strongly observed walls, one weak-but-supported wall, floor, ceiling, and opposing plane pairs |
| Freiburg1 Desk | `ABSTAIN_INSUFFICIENT_ROOM_COVERAGE` | No reliable opposing wall pairs and no reliable ceiling |

The negative control is important: the system does not force a room model from a partial, furniture-dominated sequence.

## Repository structure

```text
learning-guided-interior-slam/
├── configs/
├── data/
├── docs/
├── figures/
├── outputs/
├── results/
├── src/
├── environment.yml
├── requirements.txt
└── THIRD_PARTY.md
```

## Environment

The validated experiment used Python 3.11, PyTorch 2.5.1 with CUDA 12.4 for MASt3R-SLAM, Open3D 0.19, NumPy 1.26.4, SciPy, OpenCV, Matplotlib, and Trimesh.

```bash
conda env create -f environment.yml
conda activate interior-slam
```

MASt3R-SLAM and its learned-model dependencies must be installed separately according to the upstream repository. The exact upstream commit is pinned in `configs/mast3r_slam_commit.txt`.

## Data preparation

Download these TUM RGB-D sequences directly from the benchmark:

- `rgbd_dataset_freiburg1_room`
- `rgbd_dataset_freiburg1_desk`

No TUM images, depth maps, trajectories, or MASt3R model weights are redistributed here. See `data/README.md` for the expected layout.

## Execution model

These files preserve the validated experiment implementation. The scripts use explicit path constants matching the original RunPod workspace:

```text
/workspace/interior-slam/
```

For exact reproduction, mirror that layout. For another workstation, update the path constants near the top of each script. The experiment protocol documents the intended order and distinguishes deployable estimates from ground-truth-only diagnostics.

```bash
python src/diagnose_desk_scale.py
python src/estimate_rgbd_metric_scale.py
python src/fuse_desk_tsdf_ablation.py
python src/evaluate_tsdf_ablation_repeated.py

python src/extract_structure_candidates.py
python src/extract_manhattan_walls.py
python src/refine_room_envelope.py
python src/analyze_wall_openings.py
python src/validate_openings_with_depth.py
python src/build_final_structured_model.py
python src/finalize_structured_meshes.py
```

The structural coverage controls are then run with the same confidence policy on Room and Desk.

## Evaluation interpretation

The TSDF comparison uses the same RGB-D observations fused with different pose/scale variants. It isolates the effect of trajectory alignment and metric scale, but it is **not** an absolute surface-accuracy comparison against an independent laser scan.

Ground truth is used for reporting trajectory diagnostics, constructing a diagnostic reference fusion, and measuring how close the deployable scale estimate is to the ground-truth-derived scale. Ground truth is **not** used to estimate the reported RGB-D metric scale.

## Engineering contributions

- Learned-SLAM trajectory and scale failure diagnosis.
- RGB-D odometry-based metric translation-scale anchoring.
- Four-way TSDF ablation and repeated deterministic surface evaluation.
- Floor, ceiling, vertical-plane, and Manhattan-frame extraction.
- Persistent perimeter versus internal-plane reasoning.
- Occupancy-derived opening proposals with multi-view depth validation.
- Conservative and probable-opening structured mesh export.
- Evidence-aware envelope acceptance with a negative-control abstention test.
- Machine-readable metrics, provenance, checksums, and visual diagnostics.

## Limitations

- The verified study covers two TUM sequences rather than a broad benchmark suite.
- The source files preserve the validated experiment layout and are not yet a unified config-driven CLI package.
- The probable window is supported geometrically but not semantically confirmed.
- Dense evaluation reuses the same RGB-D observations and is not independent scan evaluation.
- The coverage policy was validated with one positive and one negative control; broader threshold calibration remains future work.
- MASt3R-SLAM runtime, weights, and third-party code are not vendored here.

## Status

**Working research prototype with quantitative scale/fusion evaluation, structured-room reconstruction, opening validation, and selective abstention.** Claims are limited to the documented TUM Freiburg1 Room and Desk experiments.
