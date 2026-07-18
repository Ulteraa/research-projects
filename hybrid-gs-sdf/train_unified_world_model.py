#!/usr/bin/env python3
"""
V1 joint trainer for Gaussian Splatting + image-conditioned SDF.

Designed to live inside the official graphdeco-inria/gaussian-splatting repo root.

What this file does:
- Keeps the official GS branch explicit: Scene + GaussianModel + render()
- Adds a trainable multiview encoder that produces a scene latent z_scene
- Adds an SDF decoder conditioned on z_scene
- Supervises the SDF branch from a CO3D point cloud (surface points + approximate outside samples)
- Couples GS and SDF through a zero-level-set consistency loss on Gaussian centers

What this file does NOT do yet:
- It does not directly feature-condition the official GS renderer internals
- It does not include a diffusion branch yet
- It does not require CO3D depths / masks at runtime (pointcloud supervision is enough for v1)

Recommended usage:
  python train_unified_world_model.py \
      -s /path/to/scene \
      -m ./output/unified_v1 \
      --pointcloud_path /path/to/pointcloud.ply

Dependencies beyond the official GS env:
  pip install plyfile
"""

import os
import sys
import uuid
import random
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from plyfile import PlyData
except ImportError as e:
    raise ImportError("Please install plyfile: pip install plyfile") from e

# ---- Official GS imports ----
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state, get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
class MultiViewEncoder(nn.Module):
    """Simple multiview encoder.

    Input:  V x 3 x H x W
    Output: z_scene (1 x D), per-view embeddings (V x D)
    """
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # images: [V, 3, H, W]
        feats = self.net(images).flatten(1)             # [V, 256]
        per_view = self.norm(self.proj(feats))          # [V, D]
        z_scene = per_view.mean(dim=0, keepdim=True)    # [1, D]
        return z_scene, per_view


class SDFDecoder(nn.Module):
    """Latent-conditioned SDF MLP."""
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 256, n_layers: int = 6):
        super().__init__()
        layers = []
        in_dim = latent_dim + 3
        for i in range(n_layers - 1):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.SiLU(inplace=True)]
        layers += [nn.Linear(hidden_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z_scene: torch.Tensor) -> torch.Tensor:
        # x: [N, 3], z_scene: [1, D] or [N, D]
        if z_scene.ndim == 2 and z_scene.shape[0] == 1:
            z = z_scene.expand(x.shape[0], -1)
        elif z_scene.ndim == 2 and z_scene.shape[0] == x.shape[0]:
            z = z_scene
        else:
            raise ValueError(f"Unexpected z_scene shape {tuple(z_scene.shape)} for x {tuple(x.shape)}")
        return self.mlp(torch.cat([x, z], dim=-1))


# -----------------------------------------------------------------------------
# Point cloud helpers
# -----------------------------------------------------------------------------
def load_pointcloud_xyz(pointcloud_path: str, device: str = "cuda") -> torch.Tensor:
    ply = PlyData.read(pointcloud_path)
    v = ply["vertex"]
    xyz = torch.stack([
        torch.tensor(v["x"], dtype=torch.float32),
        torch.tensor(v["y"], dtype=torch.float32),
        torch.tensor(v["z"], dtype=torch.float32),
    ], dim=1)
    return xyz.to(device)


def point_bbox(points: torch.Tensor, expand: float = 0.10) -> Tuple[torch.Tensor, torch.Tensor]:
    pmin = points.min(dim=0).values
    pmax = points.max(dim=0).values
    center = 0.5 * (pmin + pmax)
    half = 0.5 * (pmax - pmin)
    half = half * (1.0 + expand)
    return center - half, center + half


def sample_surface_points(surface_xyz: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.randint(0, surface_xyz.shape[0], (n,), device=surface_xyz.device)
    return surface_xyz[idx]


def nearest_distance(query: torch.Tensor, ref: torch.Tensor, ref_chunk: int = 16384) -> torch.Tensor:
    # query: [N, 3], ref: [M, 3]
    best = None
    for start in range(0, ref.shape[0], ref_chunk):
        chunk = ref[start:start + ref_chunk]
        d = torch.cdist(query, chunk)  # [N, c]
        dmin = d.min(dim=1).values
        best = dmin if best is None else torch.minimum(best, dmin)
    return best


def sample_outside_points(
    surface_xyz: torch.Tensor,
    n: int,
    margin: float,
    bbox_expand: float,
    oversample: int = 8,
) -> torch.Tensor:
    device = surface_xyz.device
    bmin, bmax = point_bbox(surface_xyz, expand=bbox_expand)
    need = n
    out_chunks = []
    # Use only a random subset of the surface cloud for NN rejection to limit cost.
    ref_subset = sample_surface_points(surface_xyz, min(surface_xyz.shape[0], 16384))
    for _ in range(12):
        cand = torch.rand((need * oversample, 3), device=device) * (bmax - bmin)[None] + bmin[None]
        dmin = nearest_distance(cand, ref_subset)
        good = cand[dmin > margin]
        if good.numel() > 0:
            take = min(need, good.shape[0])
            out_chunks.append(good[:take])
            need -= take
        if need <= 0:
            break
    if need > 0:
        # Fallback: take the farthest points we can find.
        cand = torch.rand((need * oversample, 3), device=device) * (bmax - bmin)[None] + bmin[None]
        dmin = nearest_distance(cand, ref_subset)
        topk = torch.topk(dmin, k=min(need, cand.shape[0]), largest=True).indices
        out_chunks.append(cand[topk])
    return torch.cat(out_chunks, dim=0)[:n]


# -----------------------------------------------------------------------------
# Camera / multiview helpers
# -----------------------------------------------------------------------------
def sample_train_cameras(scene: Scene, n_views: int) -> List:
    cams = scene.getTrainCameras().copy()
    if len(cams) == 0:
        raise RuntimeError("No training cameras found.")
    if len(cams) <= n_views:
        return random.sample(cams, len(cams))
    return random.sample(cams, n_views)


def make_encoder_batch(cameras: List, out_hw: int = 224) -> torch.Tensor:
    imgs = []
    for cam in cameras:
        img = cam.original_image
        if cam.alpha_mask is not None:
            img = img * cam.alpha_mask
        img = F.interpolate(img.unsqueeze(0), size=(out_hw, out_hw), mode="bilinear", align_corners=False)
        imgs.append(img.squeeze(0))
    return torch.stack(imgs, dim=0)


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def gs_photo_loss(pred: torch.Tensor, gt: torch.Tensor, lambda_dssim: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    ll1 = l1_loss(pred, gt)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0))
    else:
        ssim_value = ssim(pred, gt)
    loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)
    return loss, {"l1": float(ll1.item()), "ssim": float(ssim_value.item())}


def surface_zero_loss(sdf_decoder: SDFDecoder, surface_pts: torch.Tensor, z_scene: torch.Tensor) -> torch.Tensor:
    pred = sdf_decoder(surface_pts, z_scene)
    return pred.abs().mean()


def outside_positive_loss(
    sdf_decoder: SDFDecoder,
    outside_pts: torch.Tensor,
    z_scene: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    pred = sdf_decoder(outside_pts, z_scene)
    return F.softplus(margin - pred).mean()


def eikonal_loss(sdf_decoder: SDFDecoder, query_pts: torch.Tensor, z_scene: torch.Tensor) -> torch.Tensor:
    x = query_pts.clone().detach().requires_grad_(True)
    y = sdf_decoder(x, z_scene)
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def gaussian_surface_consistency_loss(
    sdf_decoder: SDFDecoder,
    gaussians: GaussianModel,
    z_scene: torch.Tensor,
    n_gauss: int = 4096,
) -> torch.Tensor:
    xyz = gaussians.get_xyz
    if xyz.shape[0] == 0:
        return torch.tensor(0.0, device=z_scene.device)
    if xyz.shape[0] > n_gauss:
        idx = torch.randint(0, xyz.shape[0], (n_gauss,), device=xyz.device)
        pts = xyz[idx]
    else:
        pts = xyz
    pred = sdf_decoder(pts, z_scene)
    return pred.abs().mean()


# -----------------------------------------------------------------------------
# Logging / saving
# -----------------------------------------------------------------------------
def prepare_output_folder(args) -> None:
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())[:10]
        args.model_path = os.path.join("./output", unique_str)
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def save_unified_checkpoint(path: str, iteration: int, encoder: nn.Module, sdf_decoder: nn.Module, opt_main: torch.optim.Optimizer):
    ckpt = {
        "iteration": iteration,
        "encoder": encoder.state_dict(),
        "sdf_decoder": sdf_decoder.state_dict(),
        "opt_main": opt_main.state_dict(),
    }
    torch.save(ckpt, path)


@torch.no_grad()
def evaluate_subset(scene: Scene, gaussians: GaussianModel, pipe, dataset, iteration: int, n_views: int = 5) -> Dict[str, float]:
    cams = scene.getTrainCameras()
    if len(cams) == 0:
        return {"l1": float("nan"), "psnr": float("nan")}
    sel = [cams[idx % len(cams)] for idx in range(0, n_views)]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    total_l1 = 0.0
    total_psnr = 0.0
    for cam in sel:
        out = render(cam, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        pred = torch.clamp(out["render"], 0.0, 1.0)
        gt = torch.clamp(cam.original_image.to("cuda"), 0.0, 1.0)
        total_l1 += l1_loss(pred, gt).mean().double()
        total_psnr += psnr(pred, gt).mean().double()

    total_l1 /= len(sel)
    total_psnr /= len(sel)
    print(f"\n[ITER {iteration}] Eval(train subset): L1 {total_l1} PSNR {total_psnr}")
    return {"l1": float(total_l1), "psnr": float(total_psnr)}


# -----------------------------------------------------------------------------
# Main training
# -----------------------------------------------------------------------------
def training(dataset, opt, pipe, args):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit("Trying to use sparse_adam but diff_gaussian_rasterization is not installed.")

    prepare_output_folder(args)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    encoder = MultiViewEncoder(latent_dim=args.latent_dim).cuda()
    sdf_decoder = SDFDecoder(latent_dim=args.latent_dim, hidden_dim=args.sdf_hidden_dim, n_layers=args.sdf_layers).cuda()

    # Load optional point cloud supervision.
    surface_xyz = None
    if args.pointcloud_path is not None and os.path.isfile(args.pointcloud_path):
        surface_xyz = load_pointcloud_xyz(args.pointcloud_path, device="cuda")
        print(f"Loaded pointcloud for SDF supervision: {args.pointcloud_path} ({surface_xyz.shape[0]} points)")
    else:
        print("[Warning] pointcloud_path not provided or not found. V1 trainer will fall back to Gaussian-center consistency only.")

    opt_main = torch.optim.Adam(
        list(encoder.parameters()) + list(sdf_decoder.parameters()),
        lr=args.lr_main,
        betas=(0.9, 0.99),
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss = 0.0
    ema_gs = 0.0
    ema_sdf = 0.0

    # Mirrors official GS scheduling.
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE
    progress = tqdm(range(1, opt.iterations + 1), desc="Unified training")

    for iteration in progress:
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # -----------------------------
        # 1) shared encoder latent
        # -----------------------------
        if iteration >= args.sdf_start_iter:
            enc_cams = sample_train_cameras(scene, args.encoder_views)
            enc_imgs = make_encoder_batch(enc_cams, out_hw=args.encoder_resolution)
            z_scene, _ = encoder(enc_imgs)
        else:
            z_scene = None

        # -----------------------------
        # 2) standard GS camera sample + render
        # -----------------------------
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randrange(len(viewpoint_stack)))

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )
        pred_rgb = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            pred_rgb = pred_rgb * viewpoint_cam.alpha_mask.cuda()

        gt_rgb = viewpoint_cam.original_image.cuda()
        loss_gs, gs_stats = gs_photo_loss(pred_rgb, gt_rgb, opt.lambda_dssim)
        loss = loss_gs

        # -----------------------------
        # 3) SDF branch + coupling
        # -----------------------------
        loss_sdf_surface = torch.tensor(0.0, device="cuda")
        loss_sdf_outside = torch.tensor(0.0, device="cuda")
        loss_eik = torch.tensor(0.0, device="cuda")
        loss_gauss_surface = torch.tensor(0.0, device="cuda")

        if iteration >= args.sdf_start_iter and z_scene is not None:
            if surface_xyz is not None:
                surf_pts = sample_surface_points(surface_xyz, args.surface_samples)
                loss_sdf_surface = surface_zero_loss(sdf_decoder, surf_pts, z_scene)

                out_pts = sample_outside_points(
                    surface_xyz,
                    n=args.outside_samples,
                    margin=args.outside_margin,
                    bbox_expand=args.bbox_expand,
                )
                loss_sdf_outside = outside_positive_loss(sdf_decoder, out_pts, z_scene, margin=args.outside_margin)

                eik_query = torch.cat([
                    surf_pts + 0.01 * torch.randn_like(surf_pts),
                    out_pts,
                ], dim=0)
                loss_eik = eikonal_loss(sdf_decoder, eik_query, z_scene)

            if iteration >= args.consistency_start_iter:
                loss_gauss_surface = gaussian_surface_consistency_loss(
                    sdf_decoder, gaussians, z_scene, n_gauss=args.gaussian_surface_samples
                )

            loss = (
                loss
                + args.lambda_sdf_surface * loss_sdf_surface
                + args.lambda_sdf_outside * loss_sdf_outside
                + args.lambda_eikonal * loss_eik
                + args.lambda_gaussian_surface * loss_gauss_surface
            )

        # -----------------------------
        # 4) backward / step
        # -----------------------------
        opt_main.zero_grad(set_to_none=True)
        gaussians.optimizer.zero_grad(set_to_none=True)
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # Keep official GS optimizer steps.
        gaussians.exposure_optimizer.step()
        if use_sparse_adam:
            visible = radii > 0
            gaussians.optimizer.step(visible, radii.shape[0])
        else:
            gaussians.optimizer.step()
        opt_main.step()

        # -----------------------------
        # 5) official GS densification / pruning
        # -----------------------------
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_points, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

        # -----------------------------
        # 6) logging / save
        # -----------------------------
        ema_loss = 0.4 * float(loss.item()) + 0.6 * ema_loss
        ema_gs = 0.4 * float(loss_gs.item()) + 0.6 * ema_gs
        sdf_total = float(
            (args.lambda_sdf_surface * loss_sdf_surface
             + args.lambda_sdf_outside * loss_sdf_outside
             + args.lambda_eikonal * loss_eik
             + args.lambda_gaussian_surface * loss_gauss_surface).item()
        )
        ema_sdf = 0.4 * sdf_total + 0.6 * ema_sdf

        if iteration % 10 == 0:
            progress.set_postfix({
                "Loss": f"{ema_loss:.6f}",
                "GS": f"{ema_gs:.6f}",
                "SDF": f"{ema_sdf:.6f}",
                "Pts": gaussians.get_xyz.shape[0],
            })

        if iteration in args.test_iterations:
            evaluate_subset(scene, gaussians, pipe, dataset, iteration, n_views=args.eval_views)

        if iteration in args.save_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians + unified checkpoint")
            scene.save(iteration)
            ckpt_path = os.path.join(dataset.model_path, f"unified_ckpt_{iteration}.pth")
            save_unified_checkpoint(ckpt_path, iteration, encoder, sdf_decoder, opt_main)

    progress.close()
    print("\nUnified training complete.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Unified V1 trainer: GS + image-conditioned SDF")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # V1 trainer args
    parser.add_argument("--pointcloud_path", type=str, default=None,
                        help="CO3D pointcloud.ply used for SDF supervision")
    parser.add_argument("--encoder_views", type=int, default=4,
                        help="How many train views to feed the multiview encoder per step")
    parser.add_argument("--encoder_resolution", type=int, default=224,
                        help="Square resize for encoder inputs")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--sdf_hidden_dim", type=int, default=256)
    parser.add_argument("--sdf_layers", type=int, default=6)

    parser.add_argument("--surface_samples", type=int, default=2048,
                        help="Number of point-cloud surface samples per step")
    parser.add_argument("--outside_samples", type=int, default=2048,
                        help="Number of approximate outside samples per step")
    parser.add_argument("--gaussian_surface_samples", type=int, default=4096,
                        help="How many Gaussian centers to use for SDF zero-level consistency")
    parser.add_argument("--outside_margin", type=float, default=0.03,
                        help="Distance threshold used to reject 'outside' samples away from the surface cloud")
    parser.add_argument("--bbox_expand", type=float, default=0.20,
                        help="Expansion ratio for sampling random outside points in the surface bbox")

    parser.add_argument("--sdf_start_iter", type=int, default=1000,
                        help="Iteration to start SDF supervision")
    parser.add_argument("--consistency_start_iter", type=int, default=3000,
                        help="Iteration to start GS<->SDF Gaussian center consistency")

    parser.add_argument("--lr_main", type=float, default=1e-4)
    parser.add_argument("--lambda_sdf_surface", type=float, default=1.0)
    parser.add_argument("--lambda_sdf_outside", type=float, default=0.2)
    parser.add_argument("--lambda_eikonal", type=float, default=0.05)
    parser.add_argument("--lambda_gaussian_surface", type=float, default=0.5)

    parser.add_argument("--eval_views", type=int, default=5)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    return parser, lp, op, pp


if __name__ == "__main__":
    parser, lp, op, pp = build_parser()
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    training(dataset, opt, pipe, args)


