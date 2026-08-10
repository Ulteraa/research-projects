# Result provenance

The committed `results/metrics_snapshot.json` is the paper snapshot assembled from the archived RunPod result bundles listed below. The snapshot is deliberately small and auditable; it preserves the exact scalar values used in the report.

Selected unmodified JSON reports from each bundle are retained under `results/raw/` so the aggregate snapshot can be traced back to the archived evaluator and gate outputs without redistributing trained models or datasets.

| Experiment | Archived bundle | Role |
|---|---|---|
| VP0 | `rootsplat_vp0_pilot_v012_results.zip` | constrained proximal diagnostic |
| RayOT/PGSR | `rayot_pgsr_dtu24_pilot_results.tar.gz` | transport/projection diagnostic and PGSR baseline |
| GaugeSplat/PGSR | `gaugesplat_pgsr_dtu24_pilot_results.tar.gz` | teacher-anchored refinement diagnostic |
| TraceSplat/MILo | `tracesplat_milo_dtu24_pilot_results.tar.gz` | paired MILo diagnostic |
| TSGS | `tsgs_v13_scan24_results.tar.gz` | published baseline and first-surface extraction diagnostic |

## Claim policy

1. PGSR, MILo, and TSGS are labeled as published baselines, not project inventions.
2. A project intervention is not called an improvement unless it passes both its image and geometry criteria.
3. The first-surface TSGS result is labeled post-hoc because it changes extraction, not training.
4. No result from this single scan is extrapolated to DTU as a whole.
5. Representative images are included for inspection, but the scalar snapshot is the source of truth for the committed tables.
6. `results/raw/tsgs/work/final_summary.json` records a failed official-geometry reproduction gate (2.752 mm measured versus 0.391 mm author-reported); the project does not conceal or relabel that outcome.

## Integrity check

Run:

```bash
python scripts/summarize_results.py --check
python -m unittest discover -s tests -v
```

The validator recomputes relative Chamfer improvements from the committed parent baselines with a strict numerical tolerance.
