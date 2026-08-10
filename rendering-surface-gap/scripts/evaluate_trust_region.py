#!/usr/bin/env python3
"""Audit function-space SDF drift from the accepted trust anchor."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from rootsplat import Config, RootSplat
from rootsplat.artifacts import save_json
from rootsplat.experiment import load_checkpoint, load_config


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=float("nan"), median=float("nan"), p95=float("nan"),
                    maximum=float("nan"))
    return dict(mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-mean-displacement", type=float, default=None)
    parser.add_argument("--max-p95-displacement", type=float, default=None)
    parser.add_argument("--max-gradient-rmse", type=float, default=0.35)
    args = parser.parse_args()

    run = Path(args.run)
    cfg = load_config(run / "config.resolved.yaml")
    model = RootSplat(Config(**cfg["model"]), device=args.device)
    checkpoint = Path(args.checkpoint) if args.checkpoint else \
        run / "checkpoints" / "final.pt"
    state = load_checkpoint(checkpoint, model, map_location=args.device,
                            restore_rng=False)
    anchor = model._trust_state
    if anchor is None:
        raise RuntimeError("Checkpoint has no function-space trust anchor")

    x = anchor["x"]
    with torch.enable_grad():
        value, gradient = model.sdf.s_and_grad(x, create_graph=False)
    delta_value = (value - anchor["s"]).abs()
    delta_gradient = gradient - anchor["grad"]
    gradient_error = delta_gradient.norm(dim=-1)
    # First-order implicit-function displacement estimate |delta s|/||grad s||.
    displacement = delta_value / gradient.norm(dim=-1).clamp_min(1e-6)
    n = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    nr = anchor["grad"] / anchor["grad"].norm(
        dim=-1, keepdim=True).clamp_min(1e-8)
    angle = torch.rad2deg(torch.acos((n * nr).sum(-1).clamp(-1.0, 1.0)))

    scale = float(model.cfg.trust_distance_scale)
    mean_limit = scale if args.max_mean_displacement is None else \
        float(args.max_mean_displacement)
    p95_limit = 2.0 * scale if args.max_p95_displacement is None else \
        float(args.max_p95_displacement)
    ds = statistics(displacement.detach().cpu().numpy())
    gs = statistics(gradient_error.detach().cpu().numpy())
    checks = dict(
        displacement_mean=dict(value=ds["mean"], relation="<=",
                               threshold=mean_limit,
                               passed=bool(ds["mean"] <= mean_limit)),
        displacement_p95=dict(value=ds["p95"], relation="<=",
                              threshold=p95_limit,
                              passed=bool(ds["p95"] <= p95_limit)),
        gradient_rmse=dict(
            value=float(torch.sqrt(delta_gradient.pow(2).sum(-1).mean())),
            relation="<=", threshold=float(args.max_gradient_rmse),
            passed=bool(torch.sqrt(delta_gradient.pow(2).sum(-1).mean()) <=
                        float(args.max_gradient_rmse))))
    failures = [name for name, check in checks.items() if not check["passed"]]
    payload = dict(
        schema="rootsplat.function_space_trust.v1",
        status="pass" if not failures else "fail", failures=failures,
        checkpoint=str(checkpoint), checkpoint_step=int(state["step"]),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        samples=int(len(x)), distance_scale=scale, checks=checks,
        sdf_value_absolute=statistics(delta_value.detach().cpu().numpy()),
        zero_set_displacement_first_order=ds,
        gradient_l2=gs,
        normal_angle_degrees=statistics(angle.detach().cpu().numpy()))
    output = run / "metrics" / "trust_region.json"
    save_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
