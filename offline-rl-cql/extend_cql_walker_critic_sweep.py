from pathlib import Path
import csv
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

    p = argparse.ArgumentParser(description="Critic-count extension sweep around the best baseline config.")
    p.add_argument("--device", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=300_000)
    p.add_argument("--n_steps_per_epoch", type=int, default=10_000)
    p.add_argument("--n_eval_episodes", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_steps % args.n_steps_per_epoch != 0:
        raise ValueError("--n_steps must be divisible by --n_steps_per_epoch")

    project_root = Path(__file__).resolve().parent
    resolve_dataset_root(project_root)
    set_global_seed(args.seed)

    base_root = project_root / "outputs" / "walker_cql_sweep"
    best_run_path = base_root / "best_run.json"
    if not best_run_path.exists():
        raise FileNotFoundError(
            f"Could not find best_run.json at {best_run_path}. Run sweep_cql_walker.py first."
        )

    with open(best_run_path, "r") as f:
        best_run = json.load(f)

    out_root = project_root / "outputs" / "walker_cql_critic_extension"
    out_root.mkdir(parents=True, exist_ok=True)

    results_csv = out_root / "walker_cql_critic_extension_results.csv"
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
            "batch_size",
            "best_epoch",
            "mean_return",
            "std_return",
            "normalized_score",
            "mean_length",
        ])

    print("Loading dataset/environment...")
    dataset, env = get_minari(DATASET_ID, tuple_observation=False)

    n_critics_list = [2, 4, 8]
    best_ext = None
    best_score = float("-inf")

    for n_critics in n_critics_list:
        run_name = (
            f"seed_{args.seed}_cw_{best_run['conservative_weight']}"
            f"_at_{best_run['alpha_threshold']}_ncrit_{n_critics}"
            f"_steps_{args.n_steps}"
        )
        run_dir = out_root / run_name
        print("\n" + "=" * 80)
        print(f"Starting extension run: {run_name}")
        print("=" * 80)

        algo = build_cql(
            actor_learning_rate=float(best_run["actor_lr"]),
            critic_learning_rate=float(best_run["critic_lr"]),
            temp_learning_rate=1e-4,
            alpha_learning_rate=1e-4,
            batch_size=int(best_run["batch_size"]),
            gamma=0.99,
            tau=0.005,
            n_critics=n_critics,
            initial_temperature=1.0,
            initial_alpha=1.0,
            alpha_threshold=float(best_run["alpha_threshold"]),
            conservative_weight=float(best_run["conservative_weight"]),
            n_action_samples=int(best_run["n_action_samples"]),
            soft_q_backup=bool(best_run["soft_q_backup"]),
            max_q_backup=bool(best_run["max_q_backup"]),
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
            eval_seed=args.seed,
            fit_kwargs={"experiment_name": run_name},
        )
        metrics = summary["best_metrics"]

        with open(results_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_name,
                args.seed,
                best_run["conservative_weight"],
                best_run["alpha_threshold"],
                args.n_steps,
                n_critics,
                best_run["n_action_samples"],
                best_run["soft_q_backup"],
                best_run["max_q_backup"],
                best_run["batch_size"],
                summary["best_epoch"],
                metrics["mean_return"],
                metrics["std_return"],
                metrics["normalized_score"],
                metrics["mean_length"],
            ])

        if metrics["normalized_score"] > best_score:
            best_score = metrics["normalized_score"]
            best_ext = {
                "run_name": run_name,
                "seed": args.seed,
                "n_steps": args.n_steps,
                "n_critics": n_critics,
                "best_metrics": metrics,
                "baseline_best_run": best_run["run_name"],
                "summary": summary,
            }

    env.close()

    if best_ext is not None:
        with open(out_root / "best_extension_run.json", "w") as f:
            json.dump(best_ext, f, indent=2)
        print("\nBest extension run:")
        print(json.dumps(best_ext, indent=2))


if __name__ == "__main__":
    main()
