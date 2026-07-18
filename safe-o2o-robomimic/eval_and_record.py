"""Evaluate BC or actor checkpoints in a RoboSuite task and save rollout video."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import robosuite as suite
import torch

from safe_o2o_robomimic import (
    CriticEnsemble,
    DeterministicPolicy,
    ObservationEncoder,
    SafetyConfig,
    SafetyFilter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video_path", default="rollout.mp4")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--env_name", default="Lift")
    parser.add_argument("--camera_names", nargs="+", default=["agentview"])
    parser.add_argument("--camera_height", type=int, default=84)
    parser.add_argument("--camera_width", type=int, default=84)
    parser.add_argument("--policy", choices=["actor", "bc"], default="actor")
    parser.add_argument("--use_safety_filter", action="store_true")
    parser.add_argument("--uncert_threshold", type=float, default=0.5)
    parser.add_argument("--bc_dist_threshold", type=float, default=0.5)
    parser.add_argument("--blend_alpha", type=float, default=0.25)
    parser.add_argument("--image_keys", nargs="+", default=["agentview_image"])
    parser.add_argument(
        "--robot_state_keys",
        nargs="+",
        default=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_env(args: argparse.Namespace):
    return suite.make(
        args.env_name,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
        camera_names=args.camera_names,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )


def process_image(obs: Dict, key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Missing image key '{key}'. Available keys: {list(obs.keys())}")
    image = np.asarray(obs[key], dtype=np.float32)
    if image.max() > 1.0:
        image /= 255.0
    return np.transpose(image, (2, 0, 1))


def process_robot_state(
    obs: Dict,
    keys: List[str],
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> np.ndarray:
    parts = []
    for key in keys:
        if key not in obs:
            raise KeyError(f"Missing state key '{key}'. Available keys: {list(obs.keys())}")
        parts.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
    state = np.concatenate(parts, axis=0).astype(np.float32)
    if state.shape != state_mean.shape:
        raise ValueError(
            f"Rollout state has shape {state.shape}, but checkpoint expects {state_mean.shape}. "
            "Check --env_name and --robot_state_keys."
        )
    return ((state - state_mean) / state_std).astype(np.float32)


@torch.no_grad()
def select_action(
    obs: Dict,
    *,
    bc_encoder: ObservationEncoder,
    bc_policy: DeterministicPolicy,
    rl_encoder: ObservationEncoder,
    actor: DeterministicPolicy,
    critics: CriticEnsemble,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    image_keys: List[str],
    robot_state_keys: List[str],
    device: torch.device,
    policy_name: str,
    safety_filter: Optional[SafetyFilter],
):
    images = np.stack([process_image(obs, key) for key in image_keys], axis=0)
    robot_state = process_robot_state(obs, robot_state_keys, state_mean, state_std)
    images_tensor = torch.as_tensor(images, dtype=torch.float32, device=device).unsqueeze(0)
    robot_tensor = torch.as_tensor(robot_state, dtype=torch.float32, device=device).unsqueeze(0)

    if policy_name == "bc":
        action = bc_policy(bc_encoder(images_tensor, robot_tensor))
        return action.squeeze(0).cpu().numpy(), {}

    rl_latent = rl_encoder(images_tensor, robot_tensor)
    proposed_action = actor(rl_latent)
    if safety_filter is None:
        return proposed_action.squeeze(0).cpu().numpy(), {}

    bc_action = bc_policy(bc_encoder(images_tensor, robot_tensor))
    q_values = critics(rl_latent, proposed_action)
    action, info = safety_filter.filter_action(proposed_action, bc_action, q_values)
    return action.squeeze(0).cpu().numpy(), info


def main() -> None:
    args = parse_args()
    if len(args.camera_names) != len(args.image_keys):
        raise ValueError("--camera_names and --image_keys must describe the same number of views.")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    state_mean = np.asarray(checkpoint["state_norm_mean"], dtype=np.float32)
    state_std = np.asarray(checkpoint["state_norm_std"], dtype=np.float32)

    num_views = len(args.image_keys)
    robot_state_dim = len(state_mean)
    action_dim = int(train_args.get("action_dim", 7))
    bc_encoder = ObservationEncoder(num_views, robot_state_dim).to(device)
    rl_encoder = ObservationEncoder(num_views, robot_state_dim).to(device)
    obs_dim = bc_encoder.output_dim
    bc_policy = DeterministicPolicy(
        obs_dim, action_dim, action_limit=train_args.get("action_limit", 1.0)
    ).to(device)
    actor = DeterministicPolicy(
        obs_dim, action_dim, action_limit=train_args.get("action_limit", 1.0)
    ).to(device)
    critics = CriticEnsemble(
        train_args.get("num_critics", 2), obs_dim, action_dim
    ).to(device)

    bc_encoder.load_state_dict(checkpoint["bc_encoder"])
    bc_policy.load_state_dict(checkpoint["bc_policy"])
    rl_encoder.load_state_dict(checkpoint["rl_encoder"])
    actor.load_state_dict(checkpoint["actor"])
    critics.load_state_dict(checkpoint["critics"])
    for module in (bc_encoder, bc_policy, rl_encoder, actor, critics):
        module.eval()

    safety_filter = None
    if args.use_safety_filter and args.policy == "actor":
        safety_filter = SafetyFilter(
            SafetyConfig(
                uncert_threshold=args.uncert_threshold,
                bc_dist_threshold=args.bc_dist_threshold,
                blend_alpha=args.blend_alpha,
            )
        )

    env = build_env(args)
    video_path = Path(args.video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    render_camera = args.camera_names[0]

    frames = []
    episode_returns = []
    episode_lengths = []
    successes = []

    print(
        f"Evaluating env={args.env_name} policy={args.policy} "
        f"cameras={args.camera_names} safety_filter={safety_filter is not None}"
    )

    try:
        for episode in range(args.num_episodes):
            obs = env.reset()
            episode_return = 0.0
            success = False
            length = 0

            for _ in range(args.max_steps):
                frame = env.sim.render(
                    width=args.camera_width,
                    height=args.camera_height,
                    camera_name=render_camera,
                )[::-1]
                frames.append(frame)

                action, _ = select_action(
                    obs,
                    bc_encoder=bc_encoder,
                    bc_policy=bc_policy,
                    rl_encoder=rl_encoder,
                    actor=actor,
                    critics=critics,
                    state_mean=state_mean,
                    state_std=state_std,
                    image_keys=args.image_keys,
                    robot_state_keys=args.robot_state_keys,
                    device=device,
                    policy_name=args.policy,
                    safety_filter=safety_filter,
                )
                obs, reward, done, _ = env.step(action)
                episode_return += float(reward)
                length += 1

                if hasattr(env, "_check_success") and env._check_success():
                    success = True
                    break
                if done:
                    break

            if frames:
                frames.extend([frames[-1]] * 10)
            episode_returns.append(episode_return)
            episode_lengths.append(length)
            successes.append(success)
            print(
                f"episode={episode + 1}/{args.num_episodes} "
                f"return={episode_return:.4f} length={length} success={int(success)}"
            )
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No frames were generated.")
    imageio.mimwrite(video_path, frames, fps=args.fps, macro_block_size=1)
    print(f"Saved video to {video_path}")
    print(f"mean_return={np.mean(episode_returns):.4f}")
    print(f"mean_length={np.mean(episode_lengths):.2f}")
    print(f"success_rate={np.mean(np.asarray(successes, dtype=np.float32)):.4f}")


if __name__ == "__main__":
    main()
