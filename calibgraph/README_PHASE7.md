# Phase 7 — Calibration Health Monitoring and Mechanical Drift Detection

This phase turns the calibration benchmark into a production-oriented health
monitor.

## Scenario

A three-camera articulated robot is calibrated during an initial healthy
window. One camera mount can then move mechanically by a known translation and
rotation. The camera continues observing a fixed validation target.

## Runtime residuals

For each camera and frame, the monitor computes:

- robot-frame target translation residual
- robot-frame target rotation residual
- cross-camera target disagreement

Robust baseline statistics are learned from the calibration window using the
median and median absolute deviation (MAD). The normalized residuals are
combined into a health score.

## State machine

```text
BASELINE
HEALTHY
DEGRADED
RECALIBRATION_REQUIRED
```

A critical score must persist for three consecutive frames before the monitor
requests recalibration.

## Benchmarks

- no drift
- small: 1 mm / 0.2 degrees
- medium: 3 mm / 0.5 degrees
- large: 8 mm / 1.5 degrees

Metrics:

- detection rate
- median detection delay
- drift-camera localization accuracy
- false-alarm rate

## Run

```bash
python -m pytest -q
python scripts/benchmark_drift_monitoring.py
```

This is a target-based health check. A later extension could add target-free
scene landmarks, natural-feature tracks, or downstream task residuals.
