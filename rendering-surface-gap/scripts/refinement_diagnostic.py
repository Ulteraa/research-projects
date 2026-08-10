#!/usr/bin/env python3
"""Measure finite-resolution drift under 1, 4, and 16-way subdivision."""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from rootsplat import Config, DTUScene, RootSplat
from rootsplat.artifacts import save_json
from rootsplat.experiment import frozen_parameters, load_checkpoint, load_config
from rootsplat.root import certify


def render_level(model, cam, pixels, level, copied_mass=False):
    model.split_level = torch.full((len(model.F),), int(level), dtype=torch.long,
                                   device=model.device)
    Q = model.build_quadrature(create_graph=False)
    if copied_mass and level:
        Q = dict(Q); Q["m"] = Q["m"] * (4 ** level)
    o, d = cam.rays(pixels)
    v = cam.center - Q["mu"]; v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    color = model.app(Q["mu"], Q["n"], v)
    out = model.raster(Q, o, d, colors=color, camera=cam, pixels=pixels)
    cert = certify(model.sdf, o, d, out, out["sig_ray"], lam=model.cfg.lam_newton,
                   create_graph=False)
    return {k: x.detach().cpu().numpy() for k, x in
            dict(ell=out["ell_total"], alpha=out["alpha"], rgb=out["rgb"],
                 depth=cert["depth"], valid=cert["valid"]).items()}


def drift(base, value, foreground):
    eps = 1e-8
    rel_ell = np.mean(np.abs(value["ell"][foreground] - base["ell"][foreground]) /
                      (base["ell"][foreground] + eps))
    unsat = foreground & (base["alpha"] >= 0.05) & (base["alpha"] <= 0.95)
    da = float(np.mean(np.abs(value["alpha"][unsat] - base["alpha"][unsat]))) \
        if unsat.any() else float("nan")
    valid = foreground & base["valid"] & value["valid"]
    dd = float(np.mean(np.abs(value["depth"][valid] - base["depth"][valid]))) \
        if valid.any() else float("nan")
    di = float(np.mean(np.abs(value["rgb"] - base["rgb"])))
    return dict(relative_optical_thickness=float(rel_ell), unsaturated_alpha=da,
                depth_l1=dd, rgb_l1=di)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--view", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    a = p.parse_args()
    run = Path(a.run); cfg = load_config(run / "config.resolved.yaml")
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
                    map_location=a.device)
    model.eval(); model.refresh_topology(scene.train_cameras)
    vid = a.view if a.view >= 0 else (scene.test_ids[0] if scene.test_ids else scene.train_ids[0])
    view = scene.views[vid]
    pix = view.camera.pixel_grid()
    if a.stride > 1:
        uv = pix.reshape(view.camera.H, view.camera.W, 2)
        pix = uv[::a.stride, ::a.stride].reshape(-1, 2)
    mask = None if view.mask is None else view.mask.numpy()
    result = {}
    with frozen_parameters(model), torch.enable_grad():
        base = render_level(model, view.camera, pix, 0)
        foreground = base["alpha"] > .05 if mask is None else \
            mask[::a.stride, ::a.stride].reshape(-1) > .5
        for scheme, copied in (("area_mass", False), ("copied_parent_mass", True)):
            result[scheme] = {"1": dict(relative_optical_thickness=0.0,
                                          unsaturated_alpha=0.0, depth_l1=0.0,
                                          rgb_l1=0.0)}
            for level in (1, 2):
                value = render_level(model, view.camera, pix, level, copied)
                result[scheme][str(4 ** level)] = drift(base, value, foreground)
    result["view"] = view.name
    result["evaluation_domain"] = "rendered_coverage" if mask is None else \
        "foreground_mask"
    save_json(run / "diagnostics" / "refinement.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for scheme, style in (("area_mass", "o-"), ("copied_parent_mass", "s--")):
        K = np.array([1, 4, 16])
        for ax, metric, title in zip(axes,
                ("relative_optical_thickness", "rgb_l1", "depth_l1"),
                (r"Relative $\Delta\ell$", r"$\Delta$RGB", r"$\Delta$depth")):
            ax.plot(K, [result[scheme][str(k)][metric] for k in K], style,
                    label=scheme.replace("_", " "))
            ax.set_xscale("log", base=4); ax.set_xticks(K, labels=K); ax.set_title(title)
            ax.grid(alpha=.25)
    axes[0].legend(fontsize=8); fig.tight_layout()
    out = run / "figures" / "refinement_drift.png"; out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220); plt.close(fig)
    print(f"Wrote {run / 'diagnostics' / 'refinement.json'}")


if __name__ == "__main__":
    main()
