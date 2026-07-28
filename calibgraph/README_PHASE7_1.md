# Phase 7.1 — Separate Drift Detection from Fault Localization

The validated Phase 7 health score remains the alarm trigger. A new
leave-one-camera-out isolation score identifies the likely faulty camera.

For camera `i`:

```text
isolation(i)
  = mean distance(camera i, peers)
  - mean distance among the peers
```

This prevents a drifting camera from making both healthy cameras appear
equally responsible merely because they disagree with the bad camera.

Run:

```bash
python -m pytest -q
python scripts/benchmark_drift_monitoring.py
```
