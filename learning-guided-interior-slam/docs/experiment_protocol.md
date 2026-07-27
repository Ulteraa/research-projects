# Experiment Protocol

## Scope

The project evaluates a learned-SLAM-to-structured-reconstruction stack on two TUM RGB-D sequences:

- Freiburg1 Room: positive control for complete architectural structure.
- Freiburg1 Desk: scale-failure study and negative control for structural abstention.

## 1. Learned SLAM

Run MASt3R-SLAM using the pinned commit in `../configs/mast3r_slam_commit.txt` and the camera calibration in `../configs/mast3r_slam_calib.yaml`.

Preserve estimated trajectories and dense point clouds for each sequence.

## 2. Trajectory diagnostics

For Desk:

1. Associate estimated and ground-truth trajectories by timestamp.
2. Report SE(3) ATE.
3. Report Sim(3) ATE as a scale-diagnostic alignment.
4. Compare matched path lengths and recover the diagnostic scale ratio.

The Sim(3) result is evaluation-only and not deployable.

## 3. Ground-truth-free RGB-D scale anchor

Run `src/estimate_rgbd_metric_scale.py`.

The script:

1. associates RGB and depth frames;
2. selects nearby keyframe pairs;
3. estimates relative RGB-D motion with Open3D odometry;
4. compares metric RGB-D baselines with learned-SLAM baselines;
5. rejects geometrically inconsistent pairs;
6. reports a robust median scale and bootstrap interval.

Ground-truth poses are not used to estimate this scale.

## 4. TSDF ablation

Run `src/fuse_desk_tsdf_ablation.py` to produce:

- estimated SE(3) fusion;
- ground-truth-derived Sim(3) diagnostic fusion;
- deployable RGB-D-scale fusion;
- ground-truth-pose diagnostic fusion.

Then run `src/evaluate_tsdf_ablation_repeated.py` with five deterministic seeds and 100,000 surface samples per seed.

Interpretation: all variants fuse the same RGB-D observations. The comparison isolates pose/scale effects and is not independent laser-scan evaluation.

## 5. Room structure

Run, in order:

```bash
python src/extract_structure_candidates.py
python src/extract_manhattan_walls.py
python src/refine_room_envelope.py
python src/analyze_wall_openings.py
python src/validate_openings_with_depth.py
python src/build_final_structured_model.py
python src/finalize_structured_meshes.py
```

The stages recover floor/ceiling candidates, vertical planes, a Manhattan frame, perimeter/internal planes, opening hypotheses, and conservative/hypothesis-aware structured models.

## 6. Selective structural controls

Positive control:

```bash
python src/assess_room_positive_control_fullmesh_envelope_confidence.py
```

Negative control:

```bash
python src/assess_desk_room_coverage_envelope_confidence.py
```

The same confidence policy must accept Room and abstain on Desk.

## 7. Claim policy

Permitted claims are limited to the recorded experiments:

- 52.2% TSDF RMSE reduction versus uncorrected estimated-pose fusion on Freiburg1 Desk.
- Room dimensions and mesh properties reported in machine-readable result files.
- One geometry-supported probable opening, not a semantic window confirmation.
- Selective acceptance on Room and abstention on Desk.

Do not present these measurements as official TUM benchmark scores or as broad cross-dataset generalization.
