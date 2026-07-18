from pathlib import Path
import argparse
import json

import d3rlpy
from d3rlpy.datasets import get_minari

from walker_cql_utils import DATASET_ID, evaluate_model, resolve_dataset_root


def default_model_path(project_root: Path) -> Path:
    best_run_json = project_root / "outputs" / "walker_cql_sweep" / "best_run.json"
    if best_run_json.exists():
        with open(best_run_json, "r") as f:
            best_run = json.load(f)
        candidate = project_root / "outputs" / "walker_cql_sweep" / best_run["run_name"] / "best_model.d3"
        if candidate.exists():
            return candidate
    return project_root / "outputs" / "walker_cql_runs" / "best_model.d3"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained Walker2d CQL model.")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--n_eval_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    resolve_dataset_root(project_root)

    model_path = Path(args.model_path) if args.model_path else default_model_path(project_root)

    print("Model path:", model_path)
    print("Exists:", model_path.exists())
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading dataset/environment...")
    dataset, env = get_minari(DATASET_ID, tuple_observation=False)

    print("Loading trained CQL model...")
    algo = d3rlpy.load_learnable(str(model_path), device=False)
    metrics = evaluate_model(algo, env, n_eval_episodes=args.n_eval_episodes, seed=args.seed)

    print("\nRollout summary")
    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main()
