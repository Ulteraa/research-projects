# Phase 5 — Articulated Multi-Camera Independent Calibration

This phase extends the project from one eye-in-hand camera to three cameras
mounted on different moving robot links:

- upper-arm camera
- forearm camera
- wrist camera

A synthetic four-joint articulated chain produces the known link poses
`T_B_L_i(q_t)`. Each camera has a fixed unknown mount `T_L_i_C_i`, and all
cameras observe the same target fixed in the robot base frame:

```text
T_B_T = T_B_L_i @ T_L_i_C_i @ T_C_i_T
```

## Independent baseline

Each camera is calibrated independently with the classical `AX = XB` solver,
using the carrying-link trajectory in place of a gripper trajectory.

The benchmark reports:

- camera-to-link translation and rotation error
- per-camera robot-frame target error
- pairwise cross-camera target disagreement
- observability of each carrying-link trajectory
- total calibration runtime

## Run

```bash
python -m pytest -q
python scripts/benchmark_multicamera_independent.py
```

## Why this phase matters

The independent solution is the baseline for the next phase. It ignores shared
information between cameras. Phase 6 will use these estimates to initialize a
joint graph/nonlinear optimization that enforces a common target and
cross-camera consistency.
