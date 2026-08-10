#!/usr/bin/env python3
"""Learned-field oracle-root diagnostics for the visible-root table."""
from pathlib import Path
import argparse

import numpy as np
import torch

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.artifacts import save_json
from rootsplat.experiment import frozen_parameters, load_checkpoint, load_config
from rootsplat.root import newton_step


def bracket_roots(sdf, o, d, t0, radius=.25, samples=33):
    off = torch.linspace(-radius, radius, samples, device=t0.device)
    t = (t0[:, None] + off[None]).clamp_min(1e-4)
    with torch.no_grad():
        s = sdf.s((o[:, None] + t[..., None] * d[:, None]).reshape(-1, 3)).reshape(len(t0), samples)
    change = s[:, :-1] * s[:, 1:] <= 0
    center = samples // 2
    rank = torch.arange(samples - 1, device=t0.device).sub(center).abs()[None].expand_as(change)
    rank = torch.where(change, rank, torch.full_like(rank, samples + 1))
    j = rank.argmin(1); valid = change.any(1)
    a = t.gather(1, j[:, None]).squeeze(1)
    b = t.gather(1, (j + 1)[:, None]).squeeze(1)
    return a, b, valid


def bisect(sdf, o, d, a, b, iterations=50):
    with torch.no_grad():
        fa = sdf.s(o + a[:, None] * d)
        for _ in range(iterations):
            m = .5 * (a + b); fm = sdf.s(o + m[:, None] * d)
            same = torch.sign(fm) == torch.sign(fa)
            a = torch.where(same, m, a); fa = torch.where(same, fm, fa)
            b = torch.where(same, b, m)
    return .5 * (a + b)


def stats(x):
    if len(x) == 0:
        return dict(median=float("nan"), p95=float("nan"), mean=float("nan"))
    return dict(median=float(np.median(x)), p95=float(np.percentile(x, 95)),
                mean=float(np.mean(x)))


def bootstrap_slope(e0, e1, seed=0, B=2000):
    x, y = np.log(e0), np.log(e1)
    slope = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x)); bs.append(np.polyfit(x[i], y[i], 1)[0])
    return slope, [float(v) for v in np.percentile(bs, [2.5, 97.5])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda"); p.add_argument("--rays", type=int, default=4096)
    p.add_argument("--view", type=int, default=-1); p.add_argument("--radius", type=float, default=.25)
    a = p.parse_args(); run = Path(a.run); cfg = load_config(run / "config.resolved.yaml")
    dcfg = cfg["dataset"]
    scene = DTUScene(
        dcfg["scene"], device=a.device,
        downscale=dcfg.get("downscale", .25),
        test_every=dcfg.get("test_every", 8),
        priors_dir=dcfg.get("priors_dir"),
        depth_type=dcfg.get("depth_type", "ray"),
        normal_space=dcfg.get("normal_space", "world"),
        depth_scale=dcfg.get("depth_scale", 1.0),
        require_masks=dcfg.get("require_masks", False),
        require_scale_matrices=dcfg.get("require_scale_matrices", False))
    model = RootSplat(Config(**cfg["model"]), a.device)
    load_checkpoint(a.checkpoint or run / "checkpoints" / "final.pt", model,
                    map_location=a.device); model.eval()
    vid = a.view if a.view >= 0 else (scene.test_ids[0] if scene.test_ids else scene.train_ids[0])
    view = scene.views[vid]
    if view.mask is None:
        fg = torch.arange(view.camera.H * view.camera.W)
        sample_domain = "all_pixels_no_mask"
    else:
        fg = torch.where(view.mask.reshape(-1) > .5)[0]
        sample_domain = "foreground_mask"
    if len(fg) == 0:
        raise RuntimeError(f"Selected view {view.name} has an empty sampling domain")
    take = fg[torch.randperm(len(fg))[:min(a.rays, len(fg))]]
    pixels = view.camera.pixel_grid()[take].to(a.device)
    with frozen_parameters(model), torch.enable_grad():
        _Q, _P, out, _cert, o, d = model.render(view.camera, pixels, create_graph=False)
        t0 = out["t0"]
        lo, hi, bracketed = bracket_roots(model.sdf, o, d, t0, a.radius)
        if not bool(bracketed.any()):
            raise RuntimeError(
                "No learned-field roots were bracketed; inspect the rendered "
                "alpha/depth channels or increase --radius")
        oracle = bisect(model.sdf, o[bracketed], d[bracketed], lo[bracketed], hi[bracketed])
        t0v = t0[bracketed]
        t1, _, _, _ = newton_step(model.sdf, o[bracketed], d[bracketed], t0v,
                                  lam=model.cfg.lam_newton, create_graph=False)
        t2, _, _, _ = newton_step(model.sdf, o[bracketed], d[bracketed], t1,
                                  lam=model.cfg.lam_newton, create_graph=False)
    e0 = (t0v - oracle).abs().detach().cpu().numpy()
    e1 = (t1 - oracle).abs().detach().cpu().numpy()
    e2 = (t2 - oracle).abs().detach().cpu().numpy()
    local = (e0 >= 1e-4) & (e0 <= .05) & (e1 > 1e-10)
    slope, ci = bootstrap_slope(e0[local], e1[local]) if local.sum() >= 20 else (float("nan"), [float("nan")]*2)
    result = dict(view=view.name, sample_domain=sample_domain,
                  requested_rays=int(len(take)),
                  bracketed_rays=int(bracketed.sum()), fail_fraction=float(1-bracketed.float().mean()),
                  proposal=stats(e0), one_newton=stats(e1), two_newton=stats(e2),
                  local_fit_rays=int(local.sum()), slope=slope, bootstrap_ci95=ci)
    save_json(run / "diagnostics" / "root_solver.json", result)
    print(result)


if __name__ == "__main__":
    main()
