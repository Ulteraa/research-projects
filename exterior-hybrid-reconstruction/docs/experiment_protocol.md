# Milestone 1 Experiment Protocol

Compare the same images under:

| Path | Learned initialization | Bundle adjustment | Dense MVS |
|---|---:|---:|---:|
| COLMAP | No | Yes | Optional |
| VGGT feed-forward | Yes | No | No |
| VGGT + BA | Yes | Yes | No |

Record registered images, point count, observations per image, track length, reprojection error, runtime, GPU memory, and qualitative failures. Representations differ, so point count alone must not be interpreted as quality.
