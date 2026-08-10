#!/usr/bin/env python3
"""Aggregate scene/seed run folders without treating pixels as independent."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np
import yaml


def flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    return out


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def bootstrap(values, seed=0, B=10000):
    x = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B): means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True,
                   help="Run directory; repeat for every scene and seed")
    p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(); rows = []
    for run_name in a.run:
        run = Path(run_name)
        cfg = yaml.safe_load((run / "config.resolved.yaml").read_text())
        manifest = read_json(run / "manifest.json")
        row = dict(run=str(run), scene=manifest.get("scene", cfg["dataset"].get("scene")),
                   variant=cfg["experiment"].get("variant", "unknown"),
                   seed=int(cfg["experiment"].get("seed", 0)))
        for label, path in (("nvs", run / "metrics" / "test.json"),
                            ("topology", run / "metrics" / "topology.json"),
                            ("geometry", run / "metrics" / "uniform_surface.json"),
                            ("root", run / "diagnostics" / "root_solver.json"),
                            ("refine", run / "diagnostics" / "refinement.json")):
            row.update(flatten(read_json(path), label))
        rows.append(row)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.csv", "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    aggregate = {}
    for variant in sorted({r["variant"] for r in rows}):
        vr = [r for r in rows if r["variant"] == variant]
        scenes = sorted({r["scene"] for r in vr})
        metrics = {}
        numeric = sorted(k for k in keys if k not in ("run", "scene", "variant", "seed")
                         and any(isinstance(r.get(k), (int, float)) for r in vr))
        for key in numeric:
            # Average repeated seeds inside each scene, then treat scenes as the
            # independent statistical units.
            scene_values = []
            for scene in scenes:
                v = [r[key] for r in vr if r["scene"] == scene and key in r and np.isfinite(r[key])]
                if v: scene_values.append(float(np.mean(v)))
            if scene_values:
                metrics[key] = dict(mean=float(np.mean(scene_values)),
                                    median=float(np.median(scene_values)),
                                    ci95=bootstrap(scene_values, a.seed),
                                    scenes=len(scene_values))
        aggregate[variant] = metrics
    (out / "aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True)+"\n")
    print(out)


if __name__ == "__main__":
    main()

