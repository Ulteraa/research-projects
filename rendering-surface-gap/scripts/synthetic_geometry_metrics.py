#!/usr/bin/env python3
"""Analytic geometry evaluation for the generated radius-R sphere scene."""
from pathlib import Path
import argparse
import json
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import trimesh

from rootsplat.artifacts import save_json
from rootsplat.evaluation import topology_statistics


def sample_surface(vertices, faces, count, rng):
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    area = 0.5 * twice_area
    face = rng.choice(len(faces), size=count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    su = np.sqrt(u)
    point = ((1.0 - su)[:, None] * tri[face, 0]
             + (su * (1.0 - v))[:, None] * tri[face, 1]
             + (su * v)[:, None] * tri[face, 2])
    normal = cross[face] / np.maximum(twice_area[face, None], 1e-30)
    return point, normal, area


def fibonacci_sphere(count, radius):
    i = np.arange(count, dtype=np.float64)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    y = 1.0 - 2.0 * (i + 0.5) / count
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = 2.0 * np.pi * i / golden
    return radius * np.stack([r * np.cos(theta), y, r * np.sin(theta)], 1)


def percentiles(x):
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)),
                p95=float(np.percentile(x, 95)),
                p99=float(np.percentile(x, 99)), max=float(np.max(x)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--samples", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=(0.0025, 0.005, 0.01, 0.02, 0.05))
    a = p.parse_args()

    mesh = trimesh.load(a.mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if not len(V) or not len(F):
        raise RuntimeError("Mesh is empty")

    rng = np.random.default_rng(a.seed)
    pred, face_normal, face_area = sample_surface(V, F, a.samples, rng)
    gt = fibonacci_sphere(a.samples, a.radius)
    radius_pred = np.linalg.norm(pred, axis=1)
    radial_normal = pred / np.maximum(radius_pred[:, None], 1e-30)
    cosine = np.clip((face_normal * radial_normal).sum(1), -1.0, 1.0)
    normal_oriented = np.degrees(np.arccos(cosine))
    normal_unoriented = np.degrees(np.arccos(np.abs(cosine)))

    pred_tree, gt_tree = cKDTree(pred), cKDTree(gt)
    pred_to_gt = gt_tree.query(pred, workers=-1)[0]
    gt_to_pred = pred_tree.query(gt, workers=-1)[0]
    fscore = {}
    for threshold in a.thresholds:
        precision = float(np.mean(pred_to_gt < threshold))
        recall = float(np.mean(gt_to_pred < threshold))
        f = 2.0 * precision * recall / max(precision + recall, 1e-12)
        fscore[f"{threshold:g}"] = dict(precision=precision, recall=recall,
                                        fscore=f)

    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    edge = np.sort(directed, axis=1)
    unique_edge = np.unique(edge, axis=0)
    graph = coo_matrix((np.ones(2 * len(unique_edge)),
                        (np.r_[unique_edge[:, 0], unique_edge[:, 1]],
                         np.r_[unique_edge[:, 1], unique_edge[:, 0]])),
                       shape=(len(V), len(V)))
    components = int(connected_components(graph, directed=False)[0])
    tri = V[F]
    signed_volume = float(np.einsum(
        "ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    topo = topology_statistics(F)
    topo["vertices"] = int(len(V))
    topo["connected_components"] = components

    radial_signed = radius_pred - a.radius
    payload = dict(
        mesh=str(Path(a.mesh)), radius=float(a.radius), samples=int(a.samples),
        topology=topo,
        surface_area=dict(predicted=float(face_area.sum()),
                          ground_truth=float(4.0 * math.pi * a.radius ** 2),
                          ratio=float(face_area.sum() /
                                      (4.0 * math.pi * a.radius ** 2))),
        volume=dict(predicted_signed=signed_volume,
                    ground_truth=float(4.0 * math.pi * a.radius ** 3 / 3.0),
                    absolute_ratio=float(abs(signed_volume) /
                                         (4.0 * math.pi * a.radius ** 3 / 3.0))),
        centroid=V.mean(0).tolist(), bounds=[V.min(0).tolist(), V.max(0).tolist()],
        radial_signed_mean=float(radial_signed.mean()),
        radial_absolute=percentiles(np.abs(radial_signed)),
        normal_oriented_degrees=percentiles(normal_oriented),
        normal_unoriented_degrees=percentiles(normal_unoriented),
        outward_normal_fraction=float(np.mean(cosine > 0)),
        chamfer_l1_symmetric=float(0.5 * (pred_to_gt.mean() + gt_to_pred.mean())),
        pred_to_gt=percentiles(pred_to_gt), gt_to_pred=percentiles(gt_to_pred),
        thresholds=fscore)
    save_json(a.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
