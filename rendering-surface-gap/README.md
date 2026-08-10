# The Rendering–Surface Gap in Gaussian Splatting

An audited single-scene study of a practical failure mode in geometry-aware Gaussian splatting: a representation can preserve or improve novel-view rendering while its extracted surface remains inaccurate, and a geometry intervention can improve one surface direction while worsening another.

This repository artifact packages the measured results, paper, representative renders, and small standard-library scripts used to rebuild and validate the summary tables. It does **not** claim a new state-of-the-art method.

## What is ours

- the rendering–surface-gap analysis and controlled evaluation protocol;
- the paired image/geometry acceptance gates;
- the RayOT, GaugeSplat, TraceSplat, VP0, and first-surface diagnostic experiments;
- the result audit, provenance snapshot, tables, and technical report.

PGSR, MILo, and TSGS are published third-party methods used as local control runs. Their implementations and names remain the work of their respective authors. The TSGS run produced the best image metrics in this artifact but did **not** reproduce the authors' reported Scan-24 geometry target: the archived evaluator records 2.752 mm measured Chamfer versus 0.391 mm reported.

## Verified DTU Scan 24 results

All image metrics use the same seven held-out views: 0, 8, 16, 24, 32, 40, and 48. Geometry is reported in millimetres under the staged DTU evaluation used by the archived runs.

| Method | Role | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Foreground PSNR ↑ | Chamfer ↓ |
|---|---|---:|---:|---:|---:|---:|
| PGSR | third-party control | 20.258 | 0.7210 | 0.2372 | 21.620 | 3.287 |
| MILo | third-party control | 20.565 | 0.7383 | 0.2158 | 20.787 | **2.431** |
| TSGS | third-party reproduction | **23.480** | **0.9149** | **0.0987** | **27.574** | 2.752 |
| RayOT | project diagnostic on PGSR | 20.071 | 0.7054 | 0.2527 | 21.293 | 3.111 |
| GaugeSplat | project diagnostic on PGSR | 20.257 | 0.7209 | 0.2372 | 21.619 | 3.281 |
| TraceSplat | project diagnostic on MILo | 20.546 | 0.7381 | 0.2158 | 20.777 | 2.426 |
| TSGS first surface | post-hoc extraction diagnostic | unchanged | unchanged | unchanged | unchanged | 2.646 |

The strongest renderer in this experiment is the local TSGS reproduction. The strongest measured third-party geometry control is MILo. The TSGS first-surface extraction improves TSGS Chamfer by 3.86%, but it is a post-hoc extraction diagnostic: data-to-surface worsens while surface-to-data improves. It is not presented as a new trained method.

The project interventions did not satisfy both gates:

- RayOT improved PGSR Chamfer by 5.36% but degraded every held-out image metric.
- GaugeSplat changed PGSR geometry by only 0.17% and was rejected by the geometry gate.
- TraceSplat changed MILo geometry by only 0.19% and was rejected by the geometry gate.
- VP0 reduced its median moment defect by only 0.078%, rolled back 63.6% of blocks, and produced no valid mesh score.

## Repository layout

```text
rendering-surface-gap/
├── README.md
├── PROVENANCE.md
├── paper/
│   ├── rendering_surface_gap_dtu_scan24.pdf
│   └── rendering_surface_gap_source.zip
├── results/
│   ├── metrics_snapshot.json
│   ├── summary.csv
│   ├── summary.md
│   └── raw/                 # selected archived JSON reports
├── media/
│   └── representative PNG renders and comparisons
├── scripts/
│   └── summarize_results.py
└── tests/
    └── test_metrics.py
```

## Rebuild and validate the tables

Only Python 3.9+ and the standard library are required.

```bash
cd rendering-surface-gap
python scripts/summarize_results.py
python -m unittest discover -s tests -v
```

`summarize_results.py` recomputes the relative Chamfer changes, checks finite values and method identities, and regenerates `results/summary.csv` and `results/summary.md` deterministically.

## Scope and limitations

- This is a **single DTU Scan 24 diagnostic**, not a benchmark-wide comparison.
- The runs use 42 training views and seven held-out views.
- Third-party training code, model weights, DTU data, and licenses are not redistributed here.
- The TSGS image result is valid for this run, but its official geometry reproduction gate failed.
- The archive contains result-level reproducibility and report source; it is not a unified training environment for all third-party baselines.
- The negative interventions are evidence about this setup, not a general impossibility result.

## Project page

The visual project page is published at [ulteraa.github.io/projects/rendering-surface-gap.html](https://ulteraa.github.io/projects/rendering-surface-gap.html).
