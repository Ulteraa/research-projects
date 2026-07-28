# Phase 8 — Camera/Robot Time Synchronization

A camera mounted on a moving robot link must be paired with the robot pose at
the physical image-capture time. A timestamp offset can therefore appear as an
extrinsic-calibration error.

For camera `i`, an observation timestamped at `t` is generated from:

```text
T_B_T
  = T_B_L_i(t + delta_i)
    @ T_L_i_C_i
    @ T_C_i_T(t)
```

where `delta_i` is the camera-to-robot time offset.

## Methods

- `PARK_ZERO_OFFSET`: classical independent hand-eye calibration that assumes
  camera and robot clocks are synchronized.
- `TIME_AWARE_PARK`: alternating coordinate descent:
  1. calibrate camera-to-link transforms using the current offsets;
  2. estimate one shared target pose;
  3. optimize each camera offset with a bounded robust 1D objective;
  4. repeat and recalibrate.

The synthetic trajectory is analytic and smoothly varying. In a physical
system, the same structure would interpolate timestamped robot encoder states.

## Benchmark

Maximum offset levels:

```text
0 ms
20 ms
50 ms
100 ms
```

Each level assigns different signed offsets to the upper-arm, forearm, and
wrist cameras.

Metrics:

- camera-to-link translation and rotation error
- camera time-offset estimation error
- robot-frame target error
- cross-camera disagreement
- runtime

## Run

```bash
python -m pytest -q
python scripts/benchmark_time_synchronization.py
```

The default benchmark is CPU-only and uses three Monte Carlo trials per offset
level. Use `--trials 5` for the final portfolio experiment.
