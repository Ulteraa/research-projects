# Hybrid Exterior Reconstruction

A reproducible research prototype that combines learned multi-view reconstruction with classical geometric refinement and multiview consistency filtering.

The pipeline evaluates three camera/reconstruction paths on a 12-view subset of **ETH3D Courtyard**:

1. **COLMAP** as a classical SfM baseline.
2. **VGGT feed-forward** camera and geometry prediction.
3. **VGGT + bundle adjustment**, followed by dense reconstruction and adaptive multiview filtering.

## Verified results

On the selected 12-view sequence:

- VGGT camera-center RMSE improved from **3.87 cm** to **1.27 cm** after bundle adjustment.
- Mean rotation error improved from **1.19°** to **0.45°**.
- Adaptive multiview filtering reduced median point-to-scan error from **10.26 cm** to **6.96 cm**.
- Precision within 5 cm improved from **29.1%** to **38.7%**.
- The 20 cm F1 score retained **99.1%** of the raw dense baseline.

The dense evaluation is a diagnostic comparison against a sampled ETH3D laser scan and a shared visible-region crop; it is **not** an official leaderboard score.

## Repository structure

```text
exterior-hybrid-reconstruction/
├── configs/
│   └── milestone1.yaml
├── data/
│   └── README.md
├── docs/
│   └── experiment_protocol.md
├── outputs/
│   └── README.md
├── results/
│   ├── milestone1_12views/
│   └── milestone2_final/
├── scripts/
│   ├── run_colmap.sh
│   ├── run_vggt.sh
│   ├── run_pycolmap_baseline.py
│   ├── align_and_visualize_m1.py
│   ├── evaluate_camera_poses_eth3d.py
│   ├── export_vggt_dense_predictions.py
│   ├── build_ba_refined_dense_clouds.py
│   ├── multiview_consistency_fusion.py
│   ├── adaptive_multiview_fusion.py
│   ├── evaluate_sparse_geometry_eth3d.py
│   ├── evaluate_dense_ba_variants_eth3d.py
│   └── render_viewpoint_contact_sheet.py
├── requirements.txt
└── README.md
```

## Environment

```bash
conda create -n exterior-recon python=3.10 -y
conda activate exterior-recon
pip install -r requirements.txt
bash scripts/setup_vggt.sh
```

COLMAP must be installed separately. VGGT code and checkpoints are not vendored here; follow the upstream repository and license terms.

## Data preparation

Download the ETH3D high-resolution multi-view **Courtyard** training scene directly from ETH3D and place it under:

```text
data/eth3d_courtyard_full/courtyard/
```

No ETH3D imagery, calibration data, laser scans, or VGGT weights are redistributed in this repository.

## Core execution flow

Validate the image directory:

```bash
python scripts/check_scene.py \
  --scene_dir data/eth3d_courtyard_full/courtyard
```

Run the classical baseline and VGGT export:

```bash
bash scripts/run_colmap.sh \
  data/eth3d_courtyard_full/courtyard \
  outputs/colmap sparse

bash scripts/run_vggt.sh \
  data/eth3d_courtyard_full/courtyard \
  outputs/vggt_feedforward feedforward

bash scripts/run_vggt.sh \
  data/eth3d_courtyard_full/courtyard \
  outputs/vggt_ba ba
```

The remaining Python scripts implement alignment, pose evaluation, dense export, BA-refined fusion, adaptive multiview filtering, quantitative evaluation, and visualization. Run each script with `--help` before execution because the exact input paths depend on the local ETH3D and VGGT layouts.

## Method summary

The learned model provides fast camera and dense-scene predictions. Classical bundle adjustment then improves camera consistency. Dense points are reconstructed with BA-refined cameras and filtered using view-dependent multiview support. The selected adaptive filter is intentionally precision-oriented: it rejects more weakly supported geometry while retaining nearly the same coarse-scale F1 score.

## Limitations

- The current verified experiment uses one ETH3D scene and 12 images.
- The result is a colored point cloud rather than a watertight, textured surface.
- The adaptive filter improves accuracy but lowers completeness.
- Metric evaluation uses a diagnostic crop and sampled reference scan.
- Exterior semantic decomposition, plane extraction, measurement, and design editing are later milestones.
