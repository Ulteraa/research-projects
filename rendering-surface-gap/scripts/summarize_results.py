#!/usr/bin/env python3
"""Validate the audited metric snapshot and regenerate compact result tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "metrics_snapshot.json"
DEFAULT_CSV = ROOT / "results" / "summary.csv"
DEFAULT_MARKDOWN = ROOT / "results" / "summary.md"

BASELINES = ("PGSR", "MILo", "TSGS")
DIAGNOSTICS = ("RayOT", "GaugeSplat", "TraceSplat", "TSGS_first_surface", "VP0")
METRICS = ("psnr", "ssim", "lpips", "foreground_psnr", "chamfer_mm")


def load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate(snapshot: dict[str, Any]) -> None:
    scope = snapshot["scope"]
    if scope["dataset"] != "DTU" or scope["scan"] != 24:
        raise ValueError("This artifact is scoped only to DTU Scan 24")
    if scope["train_views"] != 42 or scope["test_views"] != [0, 8, 16, 24, 32, 40, 48]:
        raise ValueError("Unexpected train/test protocol")

    baselines = snapshot["published_baselines"]
    diagnostics = snapshot["project_diagnostics"]
    if tuple(baselines) != BASELINES:
        raise ValueError(f"Unexpected baseline order or identity: {tuple(baselines)}")
    if tuple(diagnostics) != DIAGNOSTICS:
        raise ValueError(f"Unexpected diagnostic order or identity: {tuple(diagnostics)}")

    for group in (baselines, diagnostics):
        for method, values in group.items():
            for key, value in values.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"Non-finite value: {method}.{key}")

    for name in ("RayOT", "GaugeSplat", "TraceSplat", "TSGS_first_surface"):
        values = diagnostics[name]
        parent = baselines[values["parent"]]
        observed = (parent["chamfer_mm"] - values["chamfer_mm"]) / parent["chamfer_mm"]
        if not math.isclose(observed, values["relative_chamfer_improvement"], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Relative Chamfer mismatch for {name}")

    if diagnostics["RayOT"]["gate"] != "reject_image":
        raise ValueError("RayOT must remain image-gate rejected")
    for name in ("GaugeSplat", "TraceSplat"):
        if diagnostics[name]["gate"] != "reject_geometry":
            raise ValueError(f"{name} must remain geometry-gate rejected")
    if diagnostics["TSGS_first_surface"]["image_metrics_changed"] is not False:
        raise ValueError("First-surface extraction must not claim changed image metrics")
    if diagnostics["VP0"]["gate"] != "do_not_scale":
        raise ValueError("VP0 gate changed unexpectedly")


def rows(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, values in snapshot["published_baselines"].items():
        role = "third-party reproduction" if method == "TSGS" else "third-party control"
        decision = "geometry reproduction failed" if method == "TSGS" else "reference"
        output.append({"method": method, "role": role, "parent": "", **values, "decision": decision})

    for method, values in snapshot["project_diagnostics"].items():
        row = {
            "method": method,
            "role": "post-hoc extraction diagnostic" if method == "TSGS_first_surface" else "project diagnostic",
            "parent": values.get("parent", ""),
            "decision": values.get("gate", values.get("status", "diagnostic")),
        }
        for metric in METRICS:
            row[metric] = values.get(metric, "unchanged" if method == "TSGS_first_surface" and metric != "chamfer_mm" else "")
        row["relative_chamfer_improvement"] = values.get("relative_chamfer_improvement", "")
        output.append(row)
    return output


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields = (
        "method", "role", "parent", "psnr", "ssim", "lpips",
        "foreground_psnr", "chamfer_mm", "relative_chamfer_improvement", "decision",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, data: list[dict[str, Any]]) -> None:
    lines = [
        "# Audited DTU Scan 24 summary",
        "",
        "| Method | Role | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FG PSNR ↑ | Chamfer (mm) ↓ | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in data:
        lines.append(
            "| {method} | {role} | {psnr} | {ssim} | {lpips} | {fg} | {chamfer} | {decision} |".format(
                method=row["method"],
                role=row["role"],
                psnr=fmt(row.get("psnr", ""), 3),
                ssim=fmt(row.get("ssim", ""), 4),
                lpips=fmt(row.get("lpips", ""), 4),
                fg=fmt(row.get("foreground_psnr", ""), 3),
                chamfer=fmt(row.get("chamfer_mm", ""), 3),
                decision=row["decision"],
            )
        )
    lines.extend([
        "",
        "PGSR, MILo, and TSGS are third-party methods. The local TSGS run failed its official geometry-reproduction gate. All other rows are project diagnostics.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--check", action="store_true", help="validate only; do not rewrite tables")
    args = parser.parse_args()

    snapshot = load_snapshot(args.input)
    validate(snapshot)
    if not args.check:
        data = rows(snapshot)
        write_csv(DEFAULT_CSV, data)
        write_markdown(DEFAULT_MARKDOWN, data)
        print(f"Wrote {DEFAULT_CSV.relative_to(ROOT)} and {DEFAULT_MARKDOWN.relative_to(ROOT)}")
    print("Metric snapshot validation: PASS")


if __name__ == "__main__":
    main()
