# CalibGraph — Phase 1

CPU-only foundation for a senior-level multi-camera robot calibration project.

## Transform convention

`T_A_B` maps coordinates from frame `B` into frame `A`:

```text
p_A = T_A_B @ p_B
T_A_C = T_A_B @ T_B_C
```

## Setup

```bash
cd calibgraph_phase1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/check_environment.py
```

Expected: `6 passed` and `Phase 1 environment: PASS`.
