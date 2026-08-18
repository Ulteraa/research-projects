# Experiment record

The project used nuScenes `v1.0-mini` for rapid debugging and controlled architectural ablation.

| Variant | Epochs | mAP | NDS | Evidence status |
|---|---:|---:|---:|---|
| Joint camera–LiDAR BEVFusion baseline (`ConvFuser`) | 6 | 0.2578 | 0.2520 | Recorded experiment summary; raw log not currently included |
| `ReliabilityAwareFuser` | 6 | 0.2392 | 0.2415 | Recorded experiment summary; raw log not currently included |
| `ReliabilityAwareFuser` + `ObjectRefineTransFusionHead` | 4 checkpoint visualized | — | — | Qualitative point-cloud/box visualization available; final quantitative log not currently included |

## Interpretation

The first reliability-aware fusion experiment did not outperform the joint baseline in the recorded six-epoch run. That negative result motivated moving the main geometric intervention into the detection head, where local fused-BEV context is used to refine coarse proposals.

No claim is made that the custom model beats BEVFusion generally, achieves state-of-the-art results, or matches full nuScenes leaderboard evaluation.

To complete the quantitative record, add the original MMEngine logs or `vis_data/scalars.json` files from the corresponding `work_dirs` directories.
