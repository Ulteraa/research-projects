from pathlib import Path
import csv
import itertools
import json

from d3rlpy.datasets import get_minari

from walker_cql_utils import (
    DATASET_ID,
    build_cql,
    resolve_dataset_root,
    set_global_seed,
    train_with_periodic_eval,
)


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Hyperparameter sweep for stronger Walker2d CQL baseline.")
    p.add_argument("--device", default=False)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--n_steps", type=int, default=200_000)
    p.add_argument("--n_steps_per_epoch", type=int, default=10_000)
    p.add_argument("--n_eval_episodes", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_steps % args.n_steps_per_epoch != 0:
        raise ValueError("--n_steps must be divisible by --n_steps_per_epoch")

    project_root = Path(__file__).resolve().parent
    resolve_dataset_root(project_root)

    out_root = project_root / "outputs" / "walker_cql_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    results_csv = out_root / "walker_cql_sweep_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name",
            "seed",
            "conservative_weight",
            "alpha_threshold",
            "n_steps",
            "n_critics",
            "n_action_samples",
            "soft_q_backup",
            "max_q_backup",
            "actor_lr",
            "critic_lr",
            "batch_size",
            "best_epoch",
            "mean_return",
            "std_return",
            "normalized_score",
            "mean_length",
        ])

    print("Loading dataset/environment...")
    dataset, env = get_minari(DATASET_ID, tuple_observation=False)
    print("Dataset size:", dataset.size())
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    conservative_weights = [0.5, 1.0, 2.5, 5.0]
    alpha_thresholds = [1.0, 5.0, 10.0]
    n_action_samples_list = [10, 20]
    actor_lrs = [1e-4]
    critic_lrs = [3e-4]
    batch_sizes = [256, 512]
    backup_modes = [
        {"soft_q_backup": False, "max_q_backup": False},
        {"soft_q_backup": True, "max_q_backup": False},
    ]
    n_critics = 2

    best_run = None
    best_score = float("-inf")

    grid = itertools.product(
        conservative_weights,
        alpha_thresholds,
        n_action_samples_list,
        actor_lrs,
        critic_lrs,
        batch_sizes,
        backup_modes,
        range(args.num_seeds),
    )

    for cw, at, n_action_samples, actor_lr, critic_lr, batch_size, backup_mode, seed_offset in grid:
        seed = args.base_seed + seed_offset
        set_global_seed(seed)

        run_name = (
            f"seed_{seed}_cw_{cw}_at_{at}_nas_{n_action_samples}"
            f"_bs_{batch_size}_soft_{int(backup_mode['soft_q_backup'])}"
            f"_max_{int(backup_mode['max_q_backup'])}_steps_{args.n_steps}"
        )
        run_dir = out_root / run_name
        print("\n" + "=" * 100)
        print(f"Starting run: {run_name}")
        print("=" * 100)

        algo = build_cql(
            actor_learning_rate=actor_lr,
            critic_learning_rate=critic_lr,
            temp_learning_rate=1e-4,
            alpha_learning_rate=1e-4,
            batch_size=batch_size,
            gamma=0.99,
            tau=0.005,
            n_critics=n_critics,
            initial_temperature=1.0,
            initial_alpha=1.0,
            alpha_threshold=at,
            conservative_weight=cw,
            n_action_samples=n_action_samples,
            soft_q_backup=backup_mode["soft_q_backup"],
            max_q_backup=backup_mode["max_q_backup"],
            device=args.device,
        )

        summary = train_with_periodic_eval(
            algo=algo,
            dataset=dataset,
            env=env,
            run_dir=run_dir,
            total_steps=args.n_steps,
            steps_per_epoch=args.n_steps_per_epoch,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed=seed,
            fit_kwargs={"experiment_name": run_name},
        )
        metrics = summary["best_metrics"]

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_name,
                seed,
                cw,
                at,
                args.n_steps,
                n_critics,
                n_action_samples,
                backup_mode["soft_q_backup"],
                backup_mode["max_q_backup"],
                actor_lr,
                critic_lr,
                batch_size,
                summary["best_epoch"],
                metrics["mean_return"],
                metrics["std_return"],
                metrics["normalized_score"],
                metrics["mean_length"],
            ])

        if metrics["normalized_score"] > best_score:
            best_score = metrics["normalized_score"]
            best_run = {
                "run_name": run_name,
                "seed": seed,
                "conservative_weight": cw,
                "alpha_threshold": at,
                "n_steps": args.n_steps,
                "n_critics": n_critics,
                "n_action_samples": n_action_samples,
                "soft_q_backup": backup_mode["soft_q_backup"],
                "max_q_backup": backup_mode["max_q_backup"],
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "batch_size": batch_size,
                "summary": summary,
            }

    env.close()

    if best_run is not None:
        with open(out_root / "best_run.json", "w") as f:
            json.dump(best_run, f, indent=2)
        print("\nBest run:")
        print(json.dumps(best_run, indent=2))


if __name__ == "__main__":
    main()
