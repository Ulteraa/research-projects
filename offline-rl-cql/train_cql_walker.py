from pathlib import Path

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

    p = argparse.ArgumentParser(description="Train a stronger CQL baseline on Walker2d.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=False)
    p.add_argument("--n_steps", type=int, default=200_000)
    p.add_argument("--n_steps_per_epoch", type=int, default=10_000)
    p.add_argument("--n_eval_episodes", type=int, default=20)
    p.add_argument("--actor_lr", type=float, default=1e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)
    p.add_argument("--temp_lr", type=float, default=1e-4)
    p.add_argument("--alpha_lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--n_critics", type=int, default=2)
    p.add_argument("--initial_temperature", type=float, default=1.0)
    p.add_argument("--initial_alpha", type=float, default=1.0)
    p.add_argument("--alpha_threshold", type=float, default=5.0)
    p.add_argument("--conservative_weight", type=float, default=1.0)
    p.add_argument("--n_action_samples", type=int, default=20)
    p.add_argument("--soft_q_backup", action="store_true")
    p.add_argument("--max_q_backup", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_steps % args.n_steps_per_epoch != 0:
        raise ValueError("--n_steps must be divisible by --n_steps_per_epoch")

    project_root = Path(__file__).resolve().parent
    resolve_dataset_root(project_root)
    set_global_seed(args.seed)

    print("Loading Minari dataset through d3rlpy...")
    dataset, env = get_minari(DATASET_ID, tuple_observation=False)
    print("Dataset and environment loaded.")
    print("Dataset size:", dataset.size())
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    algo = build_cql(
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        temp_learning_rate=args.temp_lr,
        alpha_learning_rate=args.alpha_lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        n_critics=args.n_critics,
        initial_temperature=args.initial_temperature,
        initial_alpha=args.initial_alpha,
        alpha_threshold=args.alpha_threshold,
        conservative_weight=args.conservative_weight,
        n_action_samples=args.n_action_samples,
        soft_q_backup=args.soft_q_backup,
        max_q_backup=args.max_q_backup,
        device=args.device,
    )

    run_name = (
        f"seed_{args.seed}_cw_{args.conservative_weight}_at_{args.alpha_threshold}"
        f"_steps_{args.n_steps}_crit_{args.n_critics}"
    )
    out_dir = project_root / "outputs" / "walker_cql_runs" / run_name

    summary = train_with_periodic_eval(
        algo=algo,
        dataset=dataset,
        env=env,
        run_dir=out_dir,
        total_steps=args.n_steps,
        steps_per_epoch=args.n_steps_per_epoch,
        n_eval_episodes=args.n_eval_episodes,
        eval_seed=args.seed,
        fit_kwargs={"experiment_name": run_name},
    )

    env.close()

    print("\nSaved run directory:", out_dir)
    print("Best epoch:", summary["best_epoch"])
    print("Best metrics:", summary["best_metrics"])


if __name__ == "__main__":
    main()
