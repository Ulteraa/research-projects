#!/usr/bin/env python3
"""Create a tiny IDR-layout sphere scene for installation/GPU smoke tests."""
from pathlib import Path
import argparse

import numpy as np
from PIL import Image


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def look_at(center):
    forward = normalize(-center[None])[0]
    up = np.array([0., 1., 0.])
    right = normalize(np.cross(up, forward)[None])[0]
    up_camera = np.cross(forward, right)
    R = np.stack([right, up_camera, forward], 0)
    return R, -R @ center


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/synthetic_sphere")
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--resolution", type=int, default=64)
    a = p.parse_args(); root = Path(a.output)
    (root / "image").mkdir(parents=True, exist_ok=True)
    (root / "mask").mkdir(parents=True, exist_ok=True)
    H = W = a.resolution; focal = .9 * W
    K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], np.float32)
    yy, xx = np.meshgrid(np.arange(H) + .5, np.arange(W) + .5, indexing="ij")
    hom = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3)
    cameras = {}
    for i in range(a.views):
        az = 2 * np.pi * i / a.views; el = .18 * np.sin(2 * az)
        C = np.array([1.7*np.cos(az)*np.cos(el), 1.7*np.sin(el),
                      1.7*np.sin(az)*np.cos(el)])
        R, t = look_at(C)
        dcam = hom @ np.linalg.inv(K).T
        d = normalize(dcam @ R)
        b = d @ C; c = C @ C - .5**2; disc = b*b - c
        hit = disc >= 0; depth = -b - np.sqrt(np.maximum(disc, 0))
        hit &= depth > 0
        point = C[None] + depth[:, None] * d
        color = np.ones((H*W, 3), np.float32)
        color[hit] = .15 + .8 * ((point[hit] / .5 + 1) * .5)
        Image.fromarray(np.round(color.reshape(H, W, 3)*255).astype(np.uint8)).save(
            root / "image" / f"{i:03d}.png")
        Image.fromarray((hit.reshape(H, W)*255).astype(np.uint8)).save(
            root / "mask" / f"{i:03d}.png")
        P = K @ np.concatenate([R, t[:, None]], 1)
        world = np.eye(4, dtype=np.float32); world[:3, :4] = P
        cameras[f"world_mat_{i}"] = world
        cameras[f"scale_mat_{i}"] = np.eye(4, dtype=np.float32)
    np.savez(root / "cameras_sphere.npz", **cameras)
    print(root)


if __name__ == "__main__":
    main()
