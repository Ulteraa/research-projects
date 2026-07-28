# Phase 4 — Motion Observability and Degeneracy

A hand-eye solver cannot recover information that the robot trajectory never
excited. This phase evaluates calibration under four motion regimes:

- `diverse`: large rotations around varied axes
- `single_axis`: large rotation, but only around one axis
- `small_rotation`: multiple axes with only a few degrees of motion
- `translation_only`: no rotational excitation

## Observability report

The report includes:

- maximum and mean relative rotation
- translation baseline
- rotation-axis singular-value ratio
- rank of the linearized `AX = XB` rotation design matrix
- a `GOOD`, `WEAK`, or `POOR` classification
- a concrete trajectory recommendation

For a uniquely constrained linearized rotation problem, the expected design
rank is 8, leaving one scale/null direction before enforcing `SO(3)` structure.

## Robustness protocol

- 5 OpenCV hand-eye solvers
- 4 motion regimes
- 10 Monte Carlo trials by default
- 25 poses per trial
- 0.50 mm translation and 0.25 degree target-pose noise

The benchmark records:

- ordinary solver failures
- median translation and rotation error
- catastrophic estimates above 10 mm or 5 degrees
- observability diagnostics

## Run

```bash
pytest -q
python scripts/benchmark_motion_observability.py
```

## Key engineering lesson

Calibration quality depends on data collection, not just solver selection.
Large translation alone cannot replace multi-axis rotational excitation.
