import json
import os
import random
from pathlib import Path
from typing import Any

import d3rlpy
import numpy as np
import torch
from d3rlpy.algos import CQLConfig
from d3rlpy.preprocessing import MinMaxActionScaler, StandardObservationScaler

DATASET_ID = "mujoco/walker2d/medium-v0"
D4RL_REF_MIN = 1.629008
D4RL_REF_MAX = 4592.3


def resolve_dataset_root(project_root: Path) -> None:
    project_dataset_root = project_root / "datasets"
    default_dataset_root = Path.home() / ".minari" / "datasets"

    if (project_dataset_root / "mujoco" / "walker2d" / "medium-v0").exists():
        os.environ["MINARI_DATASETS_PATH"] = str(project_dataset_root)
        print("Using project-local dataset root:", project_dataset_root)
    else:
        os.environ["MINARI_DATASETS_PATH"] = str(default_dataset_root)
        print("Using default Minari dataset root:", default_dataset_root)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    d3rlpy.seed(seed)


def d4rl_normalized_score(mean_return: float) -> float:
    return 100.0 * (mean_return - D4RL_REF_MIN) / (D4RL_REF_MAX - D4RL_REF_MIN)


def evaluate_model(algo: Any, env: Any, n_eval_episodes: int = 20, seed: int = 0) -> dict[str, float]:
    episode_returns = []
    episode_lengths = []

    for ep_idx in range(n_eval_episodes):
        reset_seed = seed + ep_idx
        try:
            obs, info = env.reset(seed=reset_seed)
        except TypeError:
            obs, info = env.reset()

        done = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        while not (done or truncated):
            action = algo.predict(np.expand_dims(obs, axis=0))[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

        episode_returns.append(total_reward)
        episode_lengths.append(step_count)

    episode_returns = np.asarray(episode_returns, dtype=np.float64)
    episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    mean_return = float(episode_returns.mean())

    return {
        "mean_return": mean_return,
        "std_return": float(episode_returns.std()),
        "min_return": float(episode_returns.min()),
        "max_return": float(episode_returns.max()),
        "mean_length": float(episode_lengths.mean()),
        "std_length": float(episode_lengths.std()),
        "normalized_score": float(d4rl_normalized_score(mean_return)),
    }


def build_cql(
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
    temp_learning_rate: float,
    alpha_learning_rate: float,
    batch_size: int,
    gamma: float,
    tau: float,
    n_critics: int,
    initial_temperature: float,
    initial_alpha: float,
    alpha_threshold: float,
    conservative_weight: float,
    n_action_samples: int,
    soft_q_backup: bool,
    max_q_backup: bool,
    device: str | bool,
):
    if soft_q_backup and max_q_backup:
        raise ValueError("soft_q_backup and max_q_backup cannot both be True.")

    return CQLConfig(
        observation_scaler=StandardObservationScaler(),
        action_scaler=MinMaxActionScaler(),
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        temp_learning_rate=temp_learning_rate,
        alpha_learning_rate=alpha_learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        n_critics=n_critics,
        initial_temperature=initial_temperature,
        initial_alpha=initial_alpha,
        alpha_threshold=alpha_threshold,
        conservative_weight=conservative_weight,
        n_action_samples=n_action_samples,
        soft_q_backup=soft_q_backup,
        max_q_backup=max_q_backup,
    ).create(device=device)


def train_with_periodic_eval(
    *,
    algo: Any,
    dataset: Any,
    env: Any,
    run_dir: Path,
    total_steps: int,
    steps_per_epoch: int,
    n_eval_episodes: int,
    eval_seed: int,
    fit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fit_kwargs = dict(fit_kwargs or {})
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_metrics = None
    best_epoch = -1
    best_score = -np.inf
    epochs = total_steps // steps_per_epoch

    for epoch in range(1, epochs + 1):
        print("\n" + "-" * 80)
        print(f"Epoch chunk {epoch}/{epochs} | steps={steps_per_epoch}")
        print("-" * 80)

        algo.fit(
            dataset,
            n_steps=steps_per_epoch,
            n_steps_per_epoch=steps_per_epoch,
            with_timestamp=False,
            save_interval=10**9,
            show_progress=True,
            **fit_kwargs,
        )

        metrics = evaluate_model(
            algo,
            env,
            n_eval_episodes=n_eval_episodes,
            seed=eval_seed + 1000 * epoch,
        )
        metrics["epoch"] = epoch
        metrics["total_steps"] = epoch * steps_per_epoch
        history.append(metrics)

        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch:03d}.d3"
        algo.save(str(checkpoint_path))

        print("Eval metrics:", json.dumps(metrics, indent=2))

        if metrics["normalized_score"] > best_score:
            best_score = metrics["normalized_score"]
            best_epoch = epoch
            best_metrics = metrics
            algo.save(str(run_dir / "best_model.d3"))
            algo.save_policy(str(run_dir / "best_policy.pt"))

    if best_metrics is None:
        raise RuntimeError("Training loop produced no evaluation metrics.")

    algo.save(str(run_dir / "final_model.d3"))
    algo.save_policy(str(run_dir / "final_policy.pt"))

    summary = {
        "best_epoch": best_epoch,
        "best_total_steps": best_epoch * steps_per_epoch,
        "best_metrics": best_metrics,
        "history": history,
    }
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary
