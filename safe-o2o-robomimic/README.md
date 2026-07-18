# Safe Offline-to-Online RL for Vision-Based Robotic Manipulation

A research-oriented RoboMimic prototype for **behavior-cloning-preserving offline policy improvement** from image and robot-state observations.

The implementation combines a supervised behavior-cloning prior, a deterministic actor, a critic ensemble, behavior regularization, and uncertainty regularization. The long-term project direction is offline-to-online adaptation, but the code and results in this folder validate the **offline stage only**. No online environment fine-tuning result is claimed.

> **Project status:** working research prototype on RoboMimic Lift, with an unsuccessful extension to PickPlaceCan. On Lift, the conservative actor improved success from **68% for the matched BC prior to 76% over 50 evaluation episodes**. On Can, both single-camera and two-camera feedforward BC achieved **0/20 success**, showing that the current architecture does not transfer automatically to a harder long-horizon task.

## Why this problem matters

Offline robot-learning systems are trained from fixed demonstrations, but a learned actor can exploit errors in an imperfect value function and move outside the support of the data. This creates a practical tension:

- pure behavior cloning is stable but may inherit demonstration errors and compounding error;
- unconstrained actor-critic optimization can produce high predicted Q-values for unsupported actions;
- deployment requires a measurable policy-improvement mechanism without discarding the safe behavioral prior.

This prototype studies a conservative compromise: initialize from BC, preserve the BC representation, update the actor slowly, penalize distance from the BC action, and penalize critic disagreement.

## System design

For an observation containing one or more RGB views and low-dimensional robot state:

1. A shared CNN encodes each camera view.
2. A robot-state MLP encodes proprioceptive and object features.
3. The encoded features are concatenated into a multimodal latent state.
4. A BC policy is trained by supervised regression to demonstration actions.
5. A separate RL encoder and actor are initialized from the BC networks.
6. A critic ensemble is trained with Bellman targets.
7. The actor maximizes predicted value while remaining close to BC and avoiding high critic disagreement.

The implemented actor objective is:

```text
L_actor = -E[Q_mean(s, pi(s))]
          + lambda_bc * MSE(pi(s), pi_bc(s))
          + lambda_unc * E[Std(Q_1, ..., Q_K)]
```

The conservative Lift configuration used:

```text
lambda_bc        = 20
actor_lr         = 2e-5
actor_update_freq= 4 critic steps
freeze_rl_encoder= true
BC epochs        = 40
RL epochs        = 10
```

Freezing the RL encoder after copying it from BC was important. Earlier versions allowed RL critic updates to change the visual representation, which invalidated the relationship between the BC prior and the actor.

## Safety mechanism

The evaluator optionally applies a lightweight inference-time safety filter. It compares:

- critic-ensemble disagreement for the proposed action; and
- Euclidean distance between the actor action and BC action.

If either threshold is exceeded, the action is blended toward BC. This mechanism is included as an experimental control, not as a certified robot-safety system. The current thresholds were not validated on physical hardware.

## Project structure

| File | Purpose |
|---|---|
| `safe_o2o_robomimic.py` | Dataset loader, multimodal encoder, BC training, conservative actor-critic training, checkpointing, and safety-filter classes |
| `eval_and_record.py` | Multi-camera RoboSuite evaluation for BC or actor policies, with rollout video recording |
| `scripts/train_lift.sh` | Reproduce the conservative Lift training configuration |
| `scripts/eval_lift.sh` | Evaluate Lift BC or actor over 50 episodes |
| `scripts/train_can_two_camera.sh` | Train the two-camera Can experiment |
| `scripts/eval_can_two_camera.sh` | Evaluate the two-camera Can checkpoint |
| `requirements.txt` | Python dependencies from the tested local environment |

## Validated results

| Task | Observation setup | Policy | Episodes | Success rate | Interpretation |
|---|---|---:|---:|---:|---|
| Lift PH | agent view + robot/object state | BC prior | 50 | 68% | Matched baseline from the conservative checkpoint |
| Lift PH | agent view + robot/object state | conservative actor | 50 | 76% | +8 percentage points over BC |
| Can PH | agent view + robot/object state | BC prior | 20 | 0% | Feedforward BC failed on the harder task |
| Can PH | agent + wrist views + robot/object state | BC prior | 20 | 0% | Additional view alone did not solve the long-horizon problem |

These are local experimental results, not a benchmark-wide claim. They were not averaged over multiple training seeds. The Can failure is retained because it identifies a real limitation rather than presenting only a successful task.

## Requirements

The tested environment was approximately:

- Linux
- Python 3.10
- PyTorch 2.10 on CPU
- RoboMimic 0.5.0
- RoboSuite 1.5.2
- MuJoCo 3.6.0

A GPU is recommended for practical image-model training, but the validated runs were performed in a CPU environment.

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the packaged RoboMimic installation is unavailable or incompatible, install the official repository in editable mode and keep the versions aligned with RoboSuite and MuJoCo.

## Dataset preparation

The trainer expects a RoboMimic HDF5 file containing image observations and `next_obs`. Download an official RoboMimic dataset and convert states to image observations with RoboMimic's `dataset_states_to_obs.py` script.

Example two-camera Can conversion from inside a RoboMimic checkout:

```bash
python robomimic/scripts/dataset_states_to_obs.py \
  --dataset ~/robomimic_datasets/can/ph/demo_v15.hdf5 \
  --output_name image_two_cam.hdf5 \
  --done_mode 2 \
  --camera_names agentview robot0_eye_in_hand \
  --camera_height 84 \
  --camera_width 84
```

Datasets are intentionally excluded from Git.

## Run Lift

Train:

```bash
bash scripts/train_lift.sh \
  ~/robomimic_datasets/lift/ph/image.hdf5 \
  checkpoints/lift_safe_o2o.pt
```

Evaluate the BC prior:

```bash
bash scripts/eval_lift.sh checkpoints/lift_safe_o2o.pt bc videos/lift_bc.mp4
```

Evaluate the actor:

```bash
bash scripts/eval_lift.sh checkpoints/lift_safe_o2o.pt actor videos/lift_actor.mp4
```

## Run the two-camera Can experiment

Train:

```bash
bash scripts/train_can_two_camera.sh \
  ~/robomimic_datasets/can/ph/image_two_cam.hdf5 \
  checkpoints/can_two_camera_safe_o2o.pt
```

Evaluate BC:

```bash
bash scripts/eval_can_two_camera.sh \
  checkpoints/can_two_camera_safe_o2o.pt \
  bc \
  videos/can_two_camera_bc.mp4
```

## Engineering lessons

### 1. Stable losses are not evidence of control quality

The Can BC loss decreased smoothly to a small value, yet closed-loop success remained zero. Supervised action error measures fit to the demonstration distribution, not recovery from compounding rollout errors.

### 2. Environment fidelity is part of the algorithm

The first Can evaluator instantiated `Lift`, causing a 35-dimensional rollout state to be normalized with a 39-dimensional Can checkpoint. A later evaluator used only one camera while the checkpoint expected two. Both failures were implementation-level mismatches, not model-learning conclusions.

### 3. Preserve a trusted baseline during policy improvement

A shared encoder allowed critic learning to change features used by the BC policy. Separating BC and RL encoders, copying BC initialization, and freezing the conservative RL encoder made the comparison internally consistent.

### 4. More sensors do not automatically fix temporal credit and phase ambiguity

Adding the wrist camera did not improve Can BC success. The likely next bottleneck is not only visibility; it is the feedforward policy's inability to model task phase, action history, recovery, and long-horizon sequencing.

## Limitations

- The implementation is a research prototype, not a production robot-learning stack.
- The offline actor-critic update is intentionally simple and is not a full implementation of CQL, IQL, TD3+BC, or another standard benchmark algorithm.
- Results are from one training seed and finite evaluation samples.
- Reward sparsity and critic extrapolation remain concerns.
- The safety filter is heuristic and uncalibrated.
- No online data collection or online fine-tuning result is included.
- Can likely requires sequence modeling, stronger image augmentation, a recurrent or transformer policy, and matched multi-seed evaluation.
- The in-memory dataset loader is simple but not optimized for very large HDF5 datasets.

## Recommended next experiments

The highest-value extension is a controlled comparison on Lift and Can among:

1. feedforward BC;
2. BC-RNN or a transformer policy with observation history;
3. TD3+BC or IQL with the same encoder and evaluation protocol;
4. single-camera versus agent-plus-wrist observations; and
5. multiple seeds with confidence intervals.

For Can, the immediate priority is **BC-RNN before additional actor-critic tuning**. A nonzero and stable BC baseline should be established before claiming RL improvement.

## Responsible project claim

This project demonstrates practical multimodal robot-learning implementation, conservative offline policy improvement, closed-loop evaluation, debugging of environment/data mismatches, and honest failure analysis. It does **not** claim a novel state-of-the-art offline-to-online RL algorithm, certified safety, or successful online adaptation.
