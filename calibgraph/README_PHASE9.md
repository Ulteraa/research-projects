# Phase 9 — Held-Out Downstream Task-Space Validation

Calibration is not evaluated only through transform error. This phase measures
how estimated calibration changes robot-frame task geometry on trajectories
that were not used for fitting.

## Simulated workpiece

The target frame represents a rigid 300 mm x 200 mm planar parcel/workpiece.

Task geometry:

- one center grasp/placement point
- eight surface verification points
- one approach normal

## Held-out metrics

- center grasp-point error
- surface-point RMSE
- maximum surface-point error
- approach-normal angular error
- precision action acceptance: <= 2 mm and <= 1 degree
- standard action acceptance: <= 5 mm and <= 2 degrees

These thresholds are explicit simulated geometric criteria, not claims about a
specific physical gripper.

## Scenarios

1. Gaussian calibration noise:
   - PARK
   - JOINT_HUBER

2. Gross calibration outliers:
   - PARK
   - JOINT_HUBER

3. Camera/robot timestamp offsets:
   - PARK_ZERO_OFFSET
   - TIME_AWARE_PARK

Calibration and validation use distinct synthetic trajectories. Camera
predictions are also fused with a coordinate-wise median, which is robust to
one disagreeing camera.

## Run

```bash
python -m pytest -q
python scripts/benchmark_task_space_validation.py
```

This phase connects calibration quality to robot action quality and supplies the
main deployment-facing results for the final project page.
