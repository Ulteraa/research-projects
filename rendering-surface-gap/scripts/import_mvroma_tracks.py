#!/usr/bin/env python3
"""Import saved MV-RoMa dense 1-to-N predictions into RootSplat tracks.

Each input NPZ must contain ``source_view`` (scalar), ``target_views`` (T,),
``flow`` (T,2,H,W) in MV-RoMa's normalized [-1,1] target coordinates, and
``certainty`` (T,1,H,W) as pre-sigmoid logits.  Saving this small interface at
the end of the official MV-RoMa demo keeps its code and weights outside the
RootSplat repository.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from rootsplat.tracks import save_tracks


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--sigma-min-pixels", type=float, default=0.5)
    parser.add_argument("--sigma-range-pixels", type=float, default=3.0)
    parser.add_argument("--max-tracks", type=int, default=500_000)
    args = parser.parse_args()
    if args.stride <= 0 or not 0 <= args.confidence <= 1:
        raise ValueError("stride must be positive and confidence must lie in [0,1]")

    track_id, view_id, uv01, confidence, sigma01 = [], [], [], [], []
    local_track = 0
    files = sorted(Path(args.input_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError("No MV-RoMa NPZ predictions were found")
    for path in files:
        z = np.load(path, allow_pickle=False)
        source = int(np.asarray(z["source_view"]).item())
        targets = np.asarray(z["target_views"], dtype=np.int64)
        flow = np.asarray(z["flow"], dtype=np.float32)
        certainty = np.asarray(z["certainty"], dtype=np.float32)
        if flow.ndim == 5 and flow.shape[0] == 1:
            flow = flow[0]
        if certainty.ndim == 5 and certainty.shape[0] == 1:
            certainty = certainty[0]
        if flow.ndim != 4 or flow.shape[1] != 2 or flow.shape[0] != len(targets):
            raise ValueError(f"Invalid MV-RoMa flow shape in {path}")
        if certainty.ndim == 3:
            certainty = certainty[:, None]
        if certainty.shape[:2] != (len(targets), 1) or \
                certainty.shape[-2:] != flow.shape[-2:]:
            raise ValueError(f"Invalid MV-RoMa certainty shape in {path}")
        T, _two, H, W = flow.shape
        for y in range(args.stride // 2, H, args.stride):
            for x in range(args.stride // 2, W, args.stride):
                score = sigmoid(certainty[:, 0, y, x])
                valid = (score >= args.confidence) & \
                    np.isfinite(flow[:, :, y, x]).all(-1)
                target01 = 0.5 * (flow[:, :, y, x] + 1.0)
                valid &= (target01 >= 0).all(-1) & (target01 <= 1).all(-1)
                if not valid.any():
                    continue
                selected = np.flatnonzero(valid)
                source_conf = float(score[selected].mean())
                track_id.append(local_track); view_id.append(source)
                uv01.append(((x + .5) / W, (y + .5) / H))
                confidence.append(source_conf)
                source_sigma = args.sigma_min_pixels + \
                    (1.0 - source_conf) * args.sigma_range_pixels
                sigma01.append((source_sigma / W, source_sigma / H))
                for target in selected:
                    c = float(score[target])
                    sigma = args.sigma_min_pixels + (1.0 - c) * args.sigma_range_pixels
                    track_id.append(local_track); view_id.append(int(targets[target]))
                    uv01.append(tuple(target01[target]))
                    confidence.append(c); sigma01.append((sigma / W, sigma / H))
                local_track += 1
                if args.max_tracks > 0 and local_track >= args.max_tracks:
                    break
            if args.max_tracks > 0 and local_track >= args.max_tracks:
                break
        if args.max_tracks > 0 and local_track >= args.max_tracks:
            break
    if not local_track:
        raise RuntimeError("No MV-RoMa tracks passed the import thresholds")
    save_tracks(args.output, track_id, view_id, uv01, confidence, sigma01,
                metadata=dict(source="MV-RoMa", stride=args.stride,
                              confidence_threshold=args.confidence,
                              prediction_files=len(files)))
    report = dict(schema="rootsplat.mvroma_import.v1", files=len(files),
                  tracks=local_track, observations=len(track_id),
                  output=str(Path(args.output)))
    Path(args.output).with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
