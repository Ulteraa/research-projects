"""BC-preserving offline actor-critic training for RoboMimic image datasets.

The implementation intentionally separates the behavior-cloning prior from the
RL encoder and actor. After BC pretraining, the RL policy is initialized from
BC and can be constrained with a strong behavior-cloning penalty and critic-
ensemble disagreement penalty.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunningNorm:
    def __init__(self, eps: float = 1e-6):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, x: np.ndarray) -> None:
        self.mean = x.mean(axis=0).astype(np.float32)
        self.std = x.std(axis=0).astype(np.float32)
        self.std = np.where(self.std < self.eps, 1.0, self.std).astype(np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("RunningNorm must be fit before transform.")
        return (x - self.mean) / self.std


class RoboMimicOfflineDataset(Dataset):
    """Loads transitions from a RoboMimic HDF5 observation dataset.

    The code expects each demonstration to contain ``obs``, ``next_obs``,
    ``actions``, ``rewards`` and ``dones``. Images are converted to CHW float
    tensors in [0, 1], while the concatenated low-dimensional state is
    standardized over the selected demonstrations.
    """

    def __init__(
        self,
        hdf5_path: str,
        image_keys: Sequence[str],
        robot_state_keys: Sequence[str],
        reward_scale: float = 1.0,
        max_demos: Optional[int] = None,
        normalize_robot_state: bool = True,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.image_keys = list(image_keys)
        self.robot_state_keys = list(robot_state_keys)
        self.reward_scale = reward_scale
        self.normalize_robot_state = normalize_robot_state
        self.samples: List[Dict[str, np.ndarray]] = []
        self.state_norm = RunningNorm()
        self._load(max_demos=max_demos)

    def _load(self, max_demos: Optional[int]) -> None:
        raw_samples: List[Dict[str, np.ndarray]] = []
        all_robot_states: List[np.ndarray] = []

        with h5py.File(self.hdf5_path, "r") as file:
            data_group = file["data"]
            demo_keys = sorted(data_group.keys())
            if max_demos is not None:
                demo_keys = demo_keys[:max_demos]

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                actions = demo["actions"][:].astype(np.float32)
                rewards = (demo["rewards"][:] * self.reward_scale).astype(np.float32)
                dones = demo["dones"][:].astype(np.float32)

                obs_group = demo["obs"]
                next_obs_group = demo["next_obs"]

                image_arrays = [obs_group[key][:] for key in self.image_keys]
                next_image_arrays = [next_obs_group[key][:] for key in self.image_keys]
                robot_arrays = [obs_group[key][:] for key in self.robot_state_keys]
                next_robot_arrays = [next_obs_group[key][:] for key in self.robot_state_keys]

                robot_state = np.concatenate(robot_arrays, axis=-1).astype(np.float32)
                next_robot_state = np.concatenate(next_robot_arrays, axis=-1).astype(np.float32)
                all_robot_states.append(robot_state)

                for index in range(actions.shape[0]):
                    raw_samples.append(
                        {
                            "images": [array[index] for array in image_arrays],
                            "robot_state": robot_state[index],
                            "action": actions[index],
                            "reward": rewards[index],
                            "done": dones[index],
                            "next_images": [array[index] for array in next_image_arrays],
                            "next_robot_state": next_robot_state[index],
                        }
                    )

        if not raw_samples:
            raise ValueError(f"No transitions found in dataset: {self.hdf5_path}")

        if self.normalize_robot_state:
            self.state_norm.fit(np.concatenate(all_robot_states, axis=0))
        else:
            state_dim = raw_samples[0]["robot_state"].shape[-1]
            self.state_norm.mean = np.zeros(state_dim, dtype=np.float32)
            self.state_norm.std = np.ones(state_dim, dtype=np.float32)

        for sample in raw_samples:
            robot_state = sample["robot_state"]
            next_robot_state = sample["next_robot_state"]
            if self.normalize_robot_state:
                robot_state = self.state_norm.transform(robot_state[None])[0].astype(np.float32)
                next_robot_state = self.state_norm.transform(next_robot_state[None])[0].astype(np.float32)

            self.samples.append(
                {
                    "images": [self._process_image(image) for image in sample["images"]],
                    "robot_state": robot_state,
                    "action": sample["action"],
                    "reward": sample["reward"],
                    "done": sample["done"],
                    "next_images": [self._process_image(image) for image in sample["next_images"]],
                    "next_robot_state": next_robot_state,
                }
            )

    @staticmethod
    def _process_image(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return np.transpose(image, (2, 0, 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "images": torch.tensor(np.stack(sample["images"], axis=0), dtype=torch.float32),
            "robot_state": torch.tensor(sample["robot_state"], dtype=torch.float32),
            "action": torch.tensor(sample["action"], dtype=torch.float32),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "done": torch.tensor(sample["done"], dtype=torch.float32),
            "next_images": torch.tensor(np.stack(sample["next_images"], axis=0), dtype=torch.float32),
            "next_robot_state": torch.tensor(sample["next_robot_state"], dtype=torch.float32),
        }


class ConvImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)


class MultiViewImageEncoder(nn.Module):
    def __init__(self, num_views: int, image_out_dim: int = 256, fuse: str = "concat"):
        super().__init__()
        if fuse not in {"concat", "mean"}:
            raise ValueError(f"Unsupported fusion method: {fuse}")
        self.num_views = num_views
        self.fuse = fuse
        self.single_view_encoder = ConvImageEncoder(out_dim=image_out_dim)
        self.output_dim = image_out_dim * num_views if fuse == "concat" else image_out_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, views, channels, height, width = images.shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, received {views}.")
        features = self.single_view_encoder(
            images.reshape(batch * views, channels, height, width)
        ).reshape(batch, views, -1)
        return features.reshape(batch, -1) if self.fuse == "concat" else features.mean(dim=1)


class RobotStateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 128), out_dim: int = 128):
        super().__init__()
        dims = [input_dim, *hidden_dims, out_dim]
        layers: List[nn.Module] = []
        for index in range(len(dims) - 2):
            layers.extend([nn.Linear(dims[index], dims[index + 1]), nn.ReLU(inplace=True)])
        layers.extend([nn.Linear(dims[-2], dims[-1]), nn.LayerNorm(dims[-1]), nn.ReLU(inplace=True)])
        self.net = nn.Sequential(*layers)
        self.output_dim = out_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        num_views: int,
        robot_state_dim: int,
        image_latent_dim: int = 256,
        robot_latent_dim: int = 128,
        fuse: str = "concat",
    ):
        super().__init__()
        self.image_encoder = MultiViewImageEncoder(num_views, image_latent_dim, fuse)
        self.robot_encoder = RobotStateEncoder(robot_state_dim, out_dim=robot_latent_dim)
        self.output_dim = self.image_encoder.output_dim + self.robot_encoder.output_dim

    def forward(self, images: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.image_encoder(images), self.robot_encoder(robot_state)], dim=-1
        )


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        layers: List[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        action_limit: float = 1.0,
    ):
        super().__init__()
        self.body = MLP([obs_dim, *hidden_dims, action_dim])
        self.action_limit = action_limit

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.action_limit * torch.tanh(self.body(latent))


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = MLP([obs_dim + action_dim, *hidden_dims, 1])

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1)).squeeze(-1)


class CriticEnsemble(nn.Module):
    def __init__(self, num_critics: int, obs_dim: int, action_dim: int):
        super().__init__()
        if num_critics < 2:
            raise ValueError("At least two critics are required for disagreement estimation.")
        self.critics = nn.ModuleList(
            [QCritic(obs_dim, action_dim) for _ in range(num_critics)]
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([critic(latent, action) for critic in self.critics], dim=0)


@dataclass
class SafetyConfig:
    uncert_threshold: float = 0.5
    bc_dist_threshold: float = 0.5
    blend_alpha: float = 0.25


class SafetyFilter(nn.Module):
    """Blend a proposed action toward BC when it is uncertain or off-prior."""

    def __init__(self, config: SafetyConfig):
        super().__init__()
        self.config = config

    def filter_action(
        self,
        proposed_action: torch.Tensor,
        bc_action: torch.Tensor,
        q_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q_std = q_values.std(dim=0, unbiased=False)
        bc_dist = torch.norm(proposed_action - bc_action, dim=-1)
        unsafe = (q_std > self.config.uncert_threshold) | (
            bc_dist > self.config.bc_dist_threshold
        )
        alpha = self.config.blend_alpha
        safe_action = torch.where(
            unsafe.unsqueeze(-1),
            alpha * proposed_action + (1.0 - alpha) * bc_action,
            proposed_action,
        )
        return safe_action, {"q_std": q_std, "bc_dist": bc_dist, "unsafe": unsafe.float()}


class SafeOfflineTrainer:
    def __init__(
        self,
        bc_encoder: ObservationEncoder,
        bc_policy: DeterministicPolicy,
        rl_encoder: ObservationEncoder,
        actor: DeterministicPolicy,
        critics: CriticEnsemble,
        *,
        actor_lr: float,
        critic_lr: float,
        bc_lr: float,
        gamma: float,
        tau: float,
        lambda_bc: float,
        lambda_unc: float,
        actor_update_freq: int,
        freeze_rl_encoder: bool,
        device: torch.device,
    ) -> None:
        self.device = device
        self.bc_encoder = bc_encoder.to(device)
        self.bc_policy = bc_policy.to(device)
        self.rl_encoder = rl_encoder.to(device)
        self.actor = actor.to(device)
        self.critics = critics.to(device)

        self.target_rl_encoder = copy.deepcopy(self.rl_encoder).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critics = copy.deepcopy(self.critics).to(device)
        for module in (self.target_rl_encoder, self.target_actor, self.target_critics):
            module.requires_grad_(False)

        self.bc_opt = torch.optim.Adam(
            list(self.bc_encoder.parameters()) + list(self.bc_policy.parameters()), lr=bc_lr
        )
        critic_parameters = list(self.critics.parameters())
        if not freeze_rl_encoder:
            critic_parameters += list(self.rl_encoder.parameters())
        self.critic_opt = torch.optim.Adam(critic_parameters, lr=critic_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.gamma = gamma
        self.tau = tau
        self.lambda_bc = lambda_bc
        self.lambda_unc = lambda_unc
        self.actor_update_freq = max(1, actor_update_freq)
        self.freeze_rl_encoder = freeze_rl_encoder
        self.update_index = 0

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.data.mul_(1.0 - tau).add_(tau * source_parameter.data)

    def train_bc_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        images = batch["images"].to(self.device)
        robot_state = batch["robot_state"].to(self.device)
        action = batch["action"].to(self.device)

        prediction = self.bc_policy(self.bc_encoder(images, robot_state))
        loss = F.mse_loss(prediction, action)
        self.bc_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.bc_opt.step()
        return {"bc_loss": float(loss.item())}

    def initialize_rl_from_bc(self) -> None:
        self.rl_encoder.load_state_dict(copy.deepcopy(self.bc_encoder.state_dict()))
        self.actor.load_state_dict(copy.deepcopy(self.bc_policy.state_dict()))
        self.target_rl_encoder.load_state_dict(copy.deepcopy(self.rl_encoder.state_dict()))
        self.target_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        self.target_critics.load_state_dict(copy.deepcopy(self.critics.state_dict()))

        self.bc_encoder.requires_grad_(False).eval()
        self.bc_policy.requires_grad_(False).eval()
        if self.freeze_rl_encoder:
            self.rl_encoder.requires_grad_(False).eval()
            self.target_rl_encoder.load_state_dict(self.rl_encoder.state_dict())

    def _encode_current(self, images: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        if self.freeze_rl_encoder:
            with torch.no_grad():
                return self.rl_encoder(images, robot_state)
        return self.rl_encoder(images, robot_state)

    def train_rl_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.update_index += 1
        images = batch["images"].to(self.device)
        robot_state = batch["robot_state"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        done = batch["done"].to(self.device)
        next_images = batch["next_images"].to(self.device)
        next_robot_state = batch["next_robot_state"].to(self.device)

        latent = self._encode_current(images, robot_state)
        q_values = self.critics(latent, action)
        with torch.no_grad():
            next_latent = self.target_rl_encoder(next_images, next_robot_state)
            next_action = self.target_actor(next_latent)
            target_q = self.target_critics(next_latent, next_action).mean(dim=0)
            td_target = reward + self.gamma * (1.0 - done) * target_q

        critic_loss = ((q_values - td_target.unsqueeze(0)) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_metrics = {
            "actor_loss": 0.0,
            "actor_rl_loss": 0.0,
            "actor_bc_loss": 0.0,
            "actor_unc_loss": 0.0,
            "q_mean": 0.0,
            "q_std": 0.0,
            "actor_updated": 0.0,
        }

        if self.update_index % self.actor_update_freq == 0:
            actor_latent = self._encode_current(images, robot_state).detach()
            actor_action = self.actor(actor_latent)
            actor_q_values = self.critics(actor_latent, actor_action)
            q_mean = actor_q_values.mean(dim=0)
            q_std = actor_q_values.std(dim=0, unbiased=False)
            with torch.no_grad():
                bc_action = self.bc_policy(self.bc_encoder(images, robot_state))

            actor_rl_loss = -q_mean.mean()
            actor_bc_loss = F.mse_loss(actor_action, bc_action)
            actor_unc_loss = q_std.mean()
            actor_loss = (
                actor_rl_loss
                + self.lambda_bc * actor_bc_loss
                + self.lambda_unc * actor_unc_loss
            )

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor, self.target_actor, self.tau)

            actor_metrics = {
                "actor_loss": float(actor_loss.item()),
                "actor_rl_loss": float(actor_rl_loss.item()),
                "actor_bc_loss": float(actor_bc_loss.item()),
                "actor_unc_loss": float(actor_unc_loss.item()),
                "q_mean": float(q_mean.mean().item()),
                "q_std": float(q_std.mean().item()),
                "actor_updated": 1.0,
            }

        if not self.freeze_rl_encoder:
            self._soft_update(self.rl_encoder, self.target_rl_encoder, self.tau)
        self._soft_update(self.critics, self.target_critics, self.tau)

        return {"critic_loss": float(critic_loss.item()), **actor_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--image_keys", nargs="+", required=True)
    parser.add_argument("--robot_state_keys", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--bc_epochs", type=int, default=40)
    parser.add_argument("--rl_epochs", type=int, default=10)
    parser.add_argument("--actor_lr", type=float, default=2e-5)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--bc_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--lambda_bc", type=float, default=20.0)
    parser.add_argument("--lambda_unc", type=float, default=0.1)
    parser.add_argument("--num_critics", type=int, default=2)
    parser.add_argument("--actor_update_freq", type=int, default=4)
    parser.add_argument("--action_limit", type=float, default=1.0)
    parser.add_argument("--freeze_rl_encoder", action="store_true")
    parser.add_argument("--checkpoint_path", default="safe_o2o_robomimic_ckpt.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def infer_dims(dataset: RoboMimicOfflineDataset) -> Tuple[int, int, int]:
    sample = dataset[0]
    return (
        int(sample["images"].shape[0]),
        int(sample["robot_state"].shape[-1]),
        int(sample["action"].shape[-1]),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = RoboMimicOfflineDataset(
        hdf5_path=args.dataset,
        image_keys=args.image_keys,
        robot_state_keys=args.robot_state_keys,
        reward_scale=args.reward_scale,
        max_demos=args.max_demos,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    num_views, robot_state_dim, action_dim = infer_dims(dataset)
    bc_encoder = ObservationEncoder(num_views, robot_state_dim)
    rl_encoder = ObservationEncoder(num_views, robot_state_dim)
    obs_dim = bc_encoder.output_dim
    bc_policy = DeterministicPolicy(obs_dim, action_dim, action_limit=args.action_limit)
    actor = DeterministicPolicy(obs_dim, action_dim, action_limit=args.action_limit)
    critics = CriticEnsemble(args.num_critics, obs_dim, action_dim)

    trainer = SafeOfflineTrainer(
        bc_encoder,
        bc_policy,
        rl_encoder,
        actor,
        critics,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        bc_lr=args.bc_lr,
        gamma=args.gamma,
        tau=args.tau,
        lambda_bc=args.lambda_bc,
        lambda_unc=args.lambda_unc,
        actor_update_freq=args.actor_update_freq,
        freeze_rl_encoder=args.freeze_rl_encoder,
        device=device,
    )

    print(f"Loaded dataset with {len(dataset):,} transitions")
    print(
        f"num_views={num_views}, robot_state_dim={robot_state_dim}, "
        f"action_dim={action_dim}, obs_dim={obs_dim}"
    )

    for epoch in range(args.bc_epochs):
        epoch_metrics = [trainer.train_bc_step(batch) for batch in loader]
        loss = float(np.mean([metric["bc_loss"] for metric in epoch_metrics]))
        print(f"[BC] epoch={epoch + 1}/{args.bc_epochs} bc_loss={loss:.6f}")

    trainer.initialize_rl_from_bc()
    print("Initialized RL policy from frozen BC prior")

    for epoch in range(args.rl_epochs):
        epoch_metrics = [trainer.train_rl_step(batch) for batch in loader]
        summary = {
            key: float(np.mean([metric[key] for metric in epoch_metrics]))
            for key in epoch_metrics[0]
        }
        print(
            f"[RL] epoch={epoch + 1}/{args.rl_epochs} "
            f"critic={summary['critic_loss']:.6f} "
            f"actor={summary['actor_loss']:.6f} "
            f"rl={summary['actor_rl_loss']:.6f} "
            f"bc={summary['actor_bc_loss']:.6f} "
            f"unc={summary['actor_unc_loss']:.6f} "
            f"qmean={summary['q_mean']:.6f} "
            f"qstd={summary['q_std']:.6f} "
            f"actor_updates={summary['actor_updated']:.3f}"
        )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "bc_encoder": trainer.bc_encoder.state_dict(),
        "bc_policy": trainer.bc_policy.state_dict(),
        "rl_encoder": trainer.rl_encoder.state_dict(),
        "actor": trainer.actor.state_dict(),
        "critics": trainer.critics.state_dict(),
        "state_norm_mean": dataset.state_norm.mean,
        "state_norm_std": dataset.state_norm.std,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
