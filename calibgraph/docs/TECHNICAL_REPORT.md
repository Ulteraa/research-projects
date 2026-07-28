# CalibGraph Technical Report

## 1. Problem statement

Articulated robots increasingly carry cameras on multiple moving links. A
wrist camera, forearm camera, and upper-arm camera do not share the same rigid
body motion, so each extrinsic must be expressed relative to its own carrying
link. Calibration quality also depends on robot motion design, target
observation quality, clock synchronization, and mechanical stability.

CalibGraph studies the complete chain:

```text
motion collection
→ observability check
→ independent AX=XB initialization
→ optional robust joint refinement
→ optional time-offset estimation
→ online health monitoring
→ held-out task-space validation
```

## 2. Mathematical model

For camera `i` on link `L_i`:

```text
T_B_T = T_B_L_i(q_t) T_L_i_C_i T_C_i_T(t)
```

The fixed unknown is `T_L_i_C_i`. For synchronized measurements, relative
motions yield `A X = X B`. For multi-camera refinement, all camera extrinsics
and one common target pose are optimized together.

The normalized residual is

```text
r_i,t =
[ translation(E_i,t) / sigma_t,
  LogSO3(rotation(E_i,t)) / sigma_R ]
```

with

```text
E_i,t = inverse(T_B_T) T_B_L_i(t) T_L_i_C_i T_C_i_T(t).
```

The robust variant applies a Huber loss to the scalar least-squares residuals.

For unknown clock offset `delta_i`:

```text
T_B_T =
T_B_L_i(t + delta_i) T_L_i_C_i T_C_i_T(t).
```

## 3. Observability

The project diagnoses motion quality through:

- maximum and mean relative rotation;
- translation baseline;
- rotation-axis singular-value ratio;
- rank of the linearized rotation design matrix.

The exact benchmark found:

| Motion regime | Rank | Quality | Interpretation |
|---|---:|---|---|
| Diverse | 8 | GOOD | Broad multi-axis rotational excitation |
| Single axis | 6 | POOR | Rotation not uniquely constrained |
| Small rotation | 8 | POOR | Full rank but too weak relative to noise |
| Translation only | 0 | POOR | No rotational observability |

This distinction is important: algebraic rank alone does not guarantee useful
conditioning.

## 4. Robust multi-camera refinement

The three cameras are initialized independently with PARK. The graph then
optimizes all camera mounts and a shared target pose.

### Clean Gaussian condition

PARK remained best:

- mean extrinsic translation error: 0.204 mm;
- cross-camera disagreement: 0.818 mm;
- runtime: 13.7 ms.

Joint Huber was slower and did not improve the clean case.

### 8% gross outliers

Joint Huber reduced:

- mean extrinsic translation error: 2.081 → 0.573 mm;
- cross-camera disagreement: 5.459 → 1.094 mm;
- held-out fused grasp error: 2.186 → 0.882 mm;
- precision acceptance: 45.6% → 96.1%.

The result supports conditional use of robust optimization rather than a claim
that graph refinement always wins.

## 5. Drift monitoring

The healthy calibration window estimates robust residual distributions using
the median and MAD. A persistent critical score triggers
`RECALIBRATION_REQUIRED`.

The refined monitor separates system-level alarm detection from camera-level
fault isolation. At the first alert, the leave-one-camera-out isolation score
identifies the camera that disagrees with an otherwise consistent peer pair.

| Drift | Detection | Localization | False alarms |
|---|---:|---:|---:|
| 1 mm / 0.2° | 55% | 55% | 0% |
| 3 mm / 0.5° | 100% | 95% | 0% |
| 8 mm / 1.5° | 100% | 100% | 0% |

## 6. Time synchronization

At the 100 ms offset level, time-aware PARK produced:

| Metric | Zero-offset PARK | Time-aware PARK |
|---|---:|---:|
| Extrinsic translation error | 18.03 mm | 2.39 mm |
| Target localization error | 66.35 mm | 8.81 mm |
| Cross-camera disagreement | 70.97 mm | 9.38 mm |
| Mean offset estimation error | 70.0 ms baseline assumption | 8.80 ms |

The time-aware method is intentionally treated as offline calibration or
diagnostics because it is much slower than closed-form PARK.

## 7. Held-out task-space validation

The target frame represents a 300 mm × 200 mm rigid planar workpiece. The
validation trajectory was not used for fitting.

Metrics include:

- center grasp-point error;
- surface-point RMSE;
- maximum surface error;
- approach-normal angular error;
- precision acceptance: ≤2 mm and ≤1°;
- standard acceptance: ≤5 mm and ≤2°.

| Scenario | Method | Mean grasp error | P95 | Precision success | Standard success |
|---|---|---:|---:|---:|---:|
| Gaussian | PARK | 0.727 mm | 1.431 mm | 98.9% | 100% |
| Gaussian | Joint Huber | 0.765 mm | 1.427 mm | 98.9% | 100% |
| Outliers | PARK | 2.186 mm | 3.983 mm | 45.6% | 98.3% |
| Outliers | Joint Huber | 0.882 mm | 1.940 mm | 96.1% | 100% |
| Time offset | Zero-offset PARK | 19.783 mm | 49.514 mm | 0% | 0% |
| Time offset | Time-aware PARK | 2.802 mm | 6.193 mm | 37.2% | 90% |

## 8. Deployment policy suggested by the evidence

```text
clean + synchronized + observable
    → PARK

outlier-contaminated target observations
    → PARK initialization + joint Huber refinement

suspected clock mismatch
    → time-aware PARK

suspected mechanical mount movement
    → health monitor + camera isolation

poor motion observability
    → reject the calibration sequence and recollect poses
```

## 9. Limitations

The prototype remains synthetic. The next real validation step would connect
raw image corner detection, measured encoder timestamps, real kinematic
uncertainty, and a physical downstream task.
