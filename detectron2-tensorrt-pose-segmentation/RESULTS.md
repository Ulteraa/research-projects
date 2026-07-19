# Results and evidence status

## Available evidence

The accompanying report shows qualitative examples of joint object detection,
instance masks, and keypoint output before and after TensorRT conversion. It
also includes training curves for the underlying model. These figures support
the feasibility of carrying an additional keypoint branch through the
conversion pipeline.

## Evidence not provided

The supplied report and archive do not contain a reproducible table with:

- box AP, mask AP, or keypoint OKS/AP before and after conversion;
- end-to-end PyTorch and TensorRT latency;
- throughput, peak GPU memory, engine size, or warm-up protocol;
- GPU, CUDA, cuDNN, TensorRT, PyTorch, and driver versions;
- a controlled FP32/FP16 comparison or repeated trials.

Accordingly, this repository does not claim that TensorRT preserves accuracy or
achieves a particular speedup for the joint model. Numeric results published
for NVIDIA's box-and-mask reference sample are not results for this added
keypoint branch.

## Minimum validation needed for a quantitative claim

Use the same validation split, preprocessing, score thresholds, maximum
detections, and batch size for the original checkpoint and TensorRT engine.
Report:

| Metric | PyTorch FP32 | TensorRT FP32 | TensorRT FP16 |
|---|---:|---:|---:|
| Box AP | — | — | — |
| Mask AP | — | — | — |
| Keypoint OKS/AP | — | — | — |
| Median latency (ms/image) | — | — | — |
| P95 latency (ms/image) | — | — | — |
| Peak GPU memory (MiB) | — | — | — |

Include hardware/software versions, at least 100 warmed-up timing iterations,
whether transfers and preprocessing are timed, and the exact checkpoint hash.
