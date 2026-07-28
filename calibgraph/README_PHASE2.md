# Phase 2 — Synthetic Eye-in-Hand Calibration

This phase verifies the complete transform chain and all five OpenCV hand-eye
solvers using noise-free synthetic data with known ground truth.

## Frames

- `B`: robot base
- `G`: robot gripper
- `C`: camera
- `T`: calibration target

`T_A_B` maps coordinates from frame `B` into frame `A`.

The synthetic sequence satisfies:

```text
T_B_T = T_B_G @ T_G_C @ T_C_T
```

The fixed unknown is `T_G_C`.

OpenCV receives:

- `gripper2base = T_B_G`
- `target2cam = T_C_T`

and returns:

- `cam2gripper = T_G_C`

## Run

```bash
pytest -q
python scripts/benchmark_zero_noise.py
```

Expected:

- 12 tests pass
- all five methods recover ground truth
- `results/phase2_zero_noise.csv` is created
- the script prints `Zero-noise correctness gate: PASS`

## Why this phase matters

Noise experiments are meaningless until the frame conventions, synthetic
equations, API directions, and error metrics are proven correct in exact data.
