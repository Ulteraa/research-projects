# Reproducibility

## Environment

The final validation used Python 3.13 and the dependencies pinned or
constrained in `requirements.txt` and `pyproject.toml`.

## Test suite

```bash
python -m pytest -q
```

Expected final result:

```text
34 passed
```

## Experiment order

```bash
python scripts/check_environment.py
python scripts/benchmark_zero_noise.py
python scripts/benchmark_target_pose_noise.py
python scripts/benchmark_motion_observability.py
python scripts/benchmark_multicamera_independent.py
python scripts/benchmark_joint_multicamera.py
python scripts/benchmark_drift_monitoring.py
python scripts/benchmark_time_synchronization.py
python scripts/benchmark_task_space_validation.py
```

The benchmark scripts use deterministic seeds. Small runtime differences
are expected across machines; metric values should reproduce within
ordinary floating-point variation.
