# Phase 3 — Target-Pose Noise Robustness

Phase 2 proved that all transform directions and OpenCV solver interfaces are
correct in exact data. Phase 3 asks a different question:

> How robust are the five classical hand–eye solvers when the observed
> calibration-target poses are noisy?

## Noise model

Each observed `T_C_T` is perturbed independently:

```text
T_C_T_noisy = delta_T @ T_C_T
```

The perturbation is expressed in the camera frame.

- Translation: independent zero-mean Gaussian noise per Cartesian axis
- Rotation: random isotropic axis with a zero-mean Gaussian angle

This is a pose-space approximation to target detection plus PnP uncertainty.
It is not yet a raw image-corner-noise experiment; that will be a later phase.

## Monte Carlo protocol

- 5 OpenCV hand–eye methods
- 6 coupled translation/rotation noise regimes
- 10 trials per regime in the fast default run
- 25 diverse robot poses per trial
- A new synthetic robot trajectory and noise realization for every trial

Outputs:

```text
results/phase3_target_pose_noise_trials.csv
results/phase3_target_pose_noise_summary.csv
results/phase3_translation_error_vs_noise.png
results/phase3_rotation_error_vs_noise.png
```

## Run

```bash
pytest -q
python scripts/benchmark_target_pose_noise.py

# Higher-confidence final run for the portfolio report:
python scripts/benchmark_target_pose_noise.py --trials 50
```

The script records solver failures rather than silently discarding them and
keeps the zero-noise correctness gate from Phase 2.
