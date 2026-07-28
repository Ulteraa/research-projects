# Phase 6 — Joint Multi-Camera Graph Refinement

Phase 5 calibrated each camera independently. Phase 6 jointly optimizes:

- `T_upper_arm_camera`
- `T_forearm_camera`
- `T_wrist_camera`
- one shared target pose `T_B_T`

For every camera and robot pose, the graph contributes a six-dimensional
SE(3) consistency factor:

```text
T_B_L_i(t) @ T_L_i_C_i @ T_C_i_T(t) == T_B_T
```

## Methods

- `PARK`: independent closed-form AX=XB baseline
- `JOINT_LINEAR`: joint nonlinear least squares
- `JOINT_HUBER`: joint nonlinear least squares with Huber robust loss

The independent estimates initialize the joint optimization.

## Scenarios

1. Gaussian target-pose noise: 0.5 mm / 0.25 degrees
2. The same base noise plus 8% gross outliers: 20 mm / 5 degrees

## Metrics

- camera-to-link translation and rotation error
- robot-frame target localization error
- cross-camera target disagreement
- convergence rate
- runtime and function evaluations

## Run

```bash
python -m pytest -q
python scripts/benchmark_joint_multicamera.py
```

The implementation uses SciPy nonlinear least squares to remain CPU-only and
compatible with Python 3.13. The factorization is explicit and can later be
ported to GTSAM without changing the mathematical model.
