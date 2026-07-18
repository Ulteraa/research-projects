from pathlib import Path
import argparse
import json

import d3rlpy
import imageio.v2 as imageio
import minari
import numpy as np

from walker_cql_utils import DATASET_ID, d4rl_normalized_score, resolve_dataset_root


def add_pause_frames(frames, n_pause=12):
    if len(frames) == 0:
        return
    last = frames[-1]
    for _ in range(n_pause):
        frames.append(last)


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
    p = argparse.ArgumentParser(description="Record evaluation video for a trained Walker2d CQL model.")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--output_video", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=3)
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    resolve_dataset_root(project_root)

    model_path = Path(args.model_path) if args.model_path else default_model_path(project_root)
    output_video = Path(args.output_video) if args.output_video else project_root / "outputs" / "walker_cql_demo_slow.mp4"

    print("Model path:", model_path)
    print("Exists:", model_path.exists())
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Loading environment with rgb_array rendering...")
    ds = minari.load_dataset(DATASET_ID)
    env = ds.recover_environment(render_mode="rgb_array")

    print("Loading trained CQL model...")
    algo = d3rlpy.load_learnable(str(model_path), device=False)

    frames = []
    episode_returns = []
    episode_lengths = []

    for ep_idx in range(args.n_episodes):
        try:
            obs, info = env.reset(seed=args.seed + ep_idx)
        except TypeError:
            obs, info = env.reset()

        done = False
        truncated = False
        total_reward = 0.0
        step_count = 0

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not (done or truncated):
            action = algo.predict(np.expand_dims(obs, axis=0))[0]
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)

        episode_returns.append(total_reward)
        episode_lengths.append(step_count)
        print(f"Episode {ep_idx + 1:02d} | return={total_reward:.4f} | length={step_count}")
        add_pause_frames(frames, n_pause=12)

    env.close()

    if len(frames) == 0:
        raise RuntimeError("No frames were rendered.")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_video, frames, fps=args.fps)

    mean_return = float(np.mean(episode_returns))
    print("\nSaved video to:", output_video)
    print(f"Num episodes recorded: {args.n_episodes}")
    print(f"Mean return: {mean_return:.4f}")
    print(f"Normalized score: {d4rl_normalized_score(mean_return):.4f}")
    print(f"Std return: {np.std(episode_returns):.4f}")
    print(f"Mean length: {np.mean(episode_lengths):.4f}")
    print(f"Total frames: {len(frames)}")
    print(f"FPS: {args.fps}")


if __name__ == "__main__":
    main()
