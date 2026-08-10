# Audited DTU Scan 24 summary

| Method | Role | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FG PSNR ↑ | Chamfer (mm) ↓ | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| PGSR | third-party control | 20.258 | 0.7210 | 0.2372 | 21.620 | 3.287 | reference |
| MILo | third-party control | 20.565 | 0.7383 | 0.2158 | 20.787 | 2.431 | reference |
| TSGS | third-party reproduction | 23.480 | 0.9149 | 0.0987 | 27.574 | 2.752 | geometry reproduction failed |
| RayOT | project diagnostic | 20.071 | 0.7054 | 0.2527 | 21.293 | 3.111 | reject_image |
| GaugeSplat | project diagnostic | 20.257 | 0.7209 | 0.2372 | 21.619 | 3.281 | reject_geometry |
| TraceSplat | project diagnostic | 20.546 | 0.7381 | 0.2158 | 20.777 | 2.426 | reject_geometry |
| TSGS_first_surface | post-hoc extraction diagnostic | unchanged | unchanged | unchanged | unchanged | 2.646 | post_hoc_extraction_diagnostic |
| VP0 | project diagnostic |  |  |  |  |  | do_not_scale |

PGSR, MILo, and TSGS are third-party methods. The local TSGS run failed its official geometry-reproduction gate. All other rows are project diagnostics.
