# Offline RL Benchmarking and Stability Analysis with CQL

A reproducible offline reinforcement learning project built around Conservative Q-Learning (CQL) on the Minari `mujoco/walker2d/medium-v0` benchmark.

The central engineering question is not whether one additional trick always improves CQL. It is how offline-RL performance changes with conservatism, checkpoint selection, critic configuration, training horizon, evaluation variance, and dataset/runtime integrity.

> **Project status:** working research prototype. The training, evaluation, checkpointing, sweep, and rollout pipelines have run successfully. Increasing the critic count did not reliably improve closed-loop performance, so this repository does not claim a universally improved CQL variant.

## What this project demonstrates

- end-to-end offline-RL experiment design;
- reproducible CQL training and checkpoint selection;
- periodic closed-loop evaluation rather than loss-only reporting;
- structured hyperparameter sweeps;
- a critic-count extension study with an honest negative result;
- dataset, dependency, GPU, and headless-rendering debugging;
- export of models, policies, histories, CSV summaries, and rollout videos.

## Project structure

| File | Purpose |
|---|---|
| `train_cql_walker.py` | Main single-run CQL training entry point |
| `walker_cql_utils.py` | Shared dataset, seeding, CQL construction, evaluation, and checkpoint utilities |
| `eval_cql_walker_rollout.py` | Evaluate a saved CQL model over multiple environment episodes |
| `record_cql_walker_video.py` | Record rollout videos from a trained model |
| `sweep_cql_walker.py` | Sweep key CQL hyperparameters and save the best run |
| `extend_cql_walker_critic_sweep.py` | Test whether larger critic ensembles improve the best baseline |
| `train_bc_v2.py` | Optional behavior-cloning baseline for pre-generated `bc_X.npy` and `bc_Y.npy` arrays |

## Requirements

Recommended environment:

- Linux
- Python 3.12
- NVIDIA GPU for practical training speed
- a CUDA-compatible PyTorch installation
- MuJoCo-compatible OpenGL support for video rendering

The stack used during the successful cloud run included PyTorch 2.8 with CUDA 12.8, d3rlpy 2.8.1, Minari 0.5.3, Gymnasium 1.0.0, and MuJoCo. The dataset metadata recommends MuJoCo 3.2.3, so that version is pinned in `requirements.txt`.

## Installation

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build appropriate for your CPU/CUDA platform, then install this project's dependencies:

```bash
pip install -r requirements.txt
```

Verify the environment:

```bash
python - <<'PY'
import torch, d3rlpy, minari, gymnasium, mujoco
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("d3rlpy:", d3rlpy.__version__)
print("minari:", minari.__version__)
print("gymnasium:", gymnasium.__version__)
print("mujoco:", mujoco.__version__)
PY
```

## Dataset

The project uses:

```text
mujoco/walker2d/medium-v0
```

Minari downloads the dataset automatically when it is not already available. `walker_cql_utils.py` first checks for a project-local copy under:

```text
datasets/mujoco/walker2d/medium-v0
```

If that directory exists but is incomplete, remove only the broken dataset folder and rerun so Minari can download a clean copy:

```bash
rm -rf datasets/mujoco/walker2d/medium-v0
```

Downloaded datasets are intentionally excluded from Git.

## Quick smoke test

Use this to validate the full code path without treating the result as a meaningful benchmark:

```bash
python train_cql_walker.py \
  --device cuda:0 \
  --n_steps 10000 \
  --n_steps_per_epoch 5000 \
  --n_eval_episodes 2 \
  --seed 0
```

A 10k-step run verifies dataset loading, GPU execution, training, evaluation, and checkpoint writing. It is too short for a serious algorithmic conclusion.

## Longer baseline run

```bash
python train_cql_walker.py \
  --device cuda:0 \
  --n_steps 200000 \
  --n_steps_per_epoch 10000 \
  --n_eval_episodes 20 \
  --seed 0
```

Useful options include:

```text
--actor_lr
--critic_lr
--batch_size
--conservative_weight
--alpha_threshold
--n_action_samples
--n_critics
--soft_q_backup
--max_q_backup
```

## Evaluation

```bash
python eval_cql_walker_rollout.py \
  --model_path outputs/walker_cql_runs/<run_name>/best_model.d3 \
  --n_eval_episodes 100 \
  --seed 0
```

The evaluator reports raw return statistics, episode-length statistics, and a D4RL-style normalized score.

## Recording a rollout video

```bash
python record_cql_walker_video.py \
  --model_path outputs/walker_cql_runs/<run_name>/best_model.d3 \
  --n_episodes 3 \
  --fps 5
```

On a normal desktop with a display, the default backend may work directly. On a headless cloud machine, try EGL:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

or the CPU-rendering fallback:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

Headless video rendering depends on the system OpenGL/EGL/OSMesa libraries and may fail even when training works correctly. A practical alternative is to download the trained model and record the video on a local workstation.

## Hyperparameter sweep

```bash
python sweep_cql_walker.py \
  --device cuda:0 \
  --n_steps 200000 \
  --n_steps_per_epoch 10000 \
  --n_eval_episodes 20 \
  --base_seed 0 \
  --num_seeds 3
```

The current grid covers:

- conservative weight;
- alpha threshold;
- action-sample count;
- batch size;
- backup mode; and
- multiple seeds.

This grid can be expensive. Estimate the total number of runs and GPU time before launching it.

## Critic-count extension

After the main sweep creates `outputs/walker_cql_sweep/best_run.json`, run:

```bash
python extend_cql_walker_critic_sweep.py \
  --device cuda:0 \
  --n_steps 300000 \
  --n_steps_per_epoch 10000 \
  --n_eval_episodes 20 \
  --seed 0
```

This extension tests critic ensembles of 2, 4, and 8 around the best baseline configuration.

### Observed result

Increasing critic count did **not** reliably improve policy performance. This is a useful negative result: the dominant limitation was not necessarily critic under-ensembling. Potential bottlenecks include conservatism calibration, dataset support, optimization horizon, actor learning, and evaluation variance.

## Behavior-cloning baseline

`train_bc_v2.py` is an optional supervised baseline. It expects:

```text
outputs/bc_X.npy
outputs/bc_Y.npy
```

The current repository does not include the preprocessing script that creates those arrays. The BC file is retained as supporting experimental code, but it is not part of the default CQL execution path.

## Outputs

Generated files are written under `outputs/` and `d3rlpy_logs/`, including:

- epoch checkpoints;
- `best_model.d3` and `final_model.d3`;
- exported policy files;
- `training_history.json`;
- sweep and extension CSV files; and
- best-run JSON summaries.

Large datasets, models, logs, and videos are excluded from Git by default.

## Experimental interpretation

The project supports the following conclusions:

1. stable optimization metrics do not guarantee strong closed-loop behavior;
2. checkpointed environment evaluation is necessary for judging an offline policy;
3. larger critic ensembles are not a guaranteed CQL improvement;
4. dataset integrity and dependency compatibility are part of the RL system, not incidental details; and
5. meaningful comparisons require longer horizons, multiple seeds, and strong baselines.

## Current limitations

- The main implementation currently focuses on Walker2d medium data.
- The 10k-step cloud run is a smoke test, not a meaningful benchmark result.
- No universal improvement over standard CQL is claimed.
- The critic extension produced a negative result rather than a consistent gain.
- Multi-task, multi-dataset, and multi-seed result tables remain future work.
- The BC preprocessing pipeline is not included.
- Headless video rendering depends on host OpenGL configuration.

## Recommended next experiments

The highest-value continuation is not another arbitrary CQL modification. It is a controlled comparison among:

- Behavior Cloning;
- CQL;
- IQL; and
- TD3+BC.

Run each method with matched datasets, evaluation episodes, training budgets, and multiple seeds. This would turn the current implementation into a broader offline-RL benchmark and make the conclusions more useful for real decision-making systems.

## Responsible project claim

This repository demonstrates practical offline-RL engineering, experiment design, evaluation, and failure analysis. It does not claim a novel state-of-the-art algorithm or a guaranteed improvement to CQL.
