#!/usr/bin/env python3
"""Density-unbiased surface metrics for a predicted and reference mesh/point cloud.

This complements, but does not rename itself as, the official DTU evaluator.
Use the benchmark's released crop/visibility evaluator for the headline DTU row.
"""
from pathlib import Path
import argparse

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from rootsplat.artifacts import save_json


def sample_surface(obj, n, rng):
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if isinstance(obj, trimesh.Trimesh) and len(obj.faces):
        # Implement area-weighted sampling locally so results are seeded and
        # identical across trimesh releases (older versions do not accept the
        # ``seed`` argument used by newer versions).
        tri = np.asarray(obj.triangles)
        area = np.asarray(obj.area_faces, dtype=np.float64)
        valid = np.isfinite(area) & (area > 0)
        if not valid.any():
            raise ValueError("Prediction/reference mesh has no positive-area faces")
        face_pool = np.flatnonzero(valid)
        fid = rng.choice(face_pool, size=n, p=area[valid] / area[valid].sum())
        uv = rng.random((n, 2))
        flip = uv.sum(axis=1) > 1.0
        uv[flip] = 1.0 - uv[flip]
        chosen = tri[fid]
        pts = chosen[:, 0] + uv[:, :1] * (chosen[:, 1] - chosen[:, 0]) \
              + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
        normals = obj.face_normals[fid]
        return pts, normals
    pts = np.asarray(obj.vertices)
    idx = rng.integers(0, len(pts), n)
    normals = np.asarray(obj.vertex_normals)[idx] if hasattr(obj, "vertex_normals") else None
    return pts[idx], normals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prediction", required=True); p.add_argument("--ground-truth", required=True)
    p.add_argument("--output", required=True); p.add_argument("--samples", type=int, default=2_000_000)
    p.add_argument("--thresholds", type=float, nargs="+", default=[.5, 1., 2.])
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply both geometries before evaluating (e.g. normalized units to mm)")
    p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    pred = trimesh.load(a.prediction, process=False); gt = trimesh.load(a.ground_truth, process=False)
    P, PN = sample_surface(pred, a.samples, rng); G, GN = sample_surface(gt, a.samples, rng)
    P, G = P * a.scale, G * a.scale
    tg, tp = cKDTree(G), cKDTree(P)
    dpg, ipg = tg.query(P, workers=-1); dgp, igp = tp.query(G, workers=-1)
    result = dict(accuracy=float(dpg.mean()), completion=float(dgp.mean()),
                  chamfer_l1=float(.5 * (dpg.mean() + dgp.mean())), thresholds={})
    for t in a.thresholds:
        precision, recall = float((dpg < t).mean()), float((dgp < t).mean())
        result["thresholds"][str(t)] = dict(precision=precision, recall=recall,
            fscore=float(2*precision*recall/max(precision+recall, 1e-12)))
    if PN is not None and GN is not None:
        nc1 = np.abs((PN * GN[ipg]).sum(-1)).mean()
        nc2 = np.abs((GN * PN[igp]).sum(-1)).mean()
        result["normal_consistency"] = float(.5 * (nc1 + nc2))
    save_json(a.output, result); print(result)


if __name__ == "__main__":
    main()
