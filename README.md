# Research Projects

Selected research prototypes and implementations in computer vision, 3D vision, generative modeling, and machine learning.

These projects emphasize hands-on system design and implementation. Each folder documents its validation status, requirements, usage, and limitations.

## Projects

### [Hybrid Geometry-Aware Gaussian Splatting](hybrid-gs-sdf)

A joint training prototype combining the official 3D Gaussian Splatting pipeline with an image-conditioned signed distance function (SDF) branch. It explores auxiliary geometric supervision and zero-level-set consistency on Gaussian centers.

**Topics:** Gaussian Splatting · SDF · Novel-view synthesis · Multi-view learning · 3D scene representation

**Status:** Working research prototype with qualitative inspection. Controlled GS-only comparisons and ablation studies remain future work.


### [Offline RL Benchmarking and Stability Analysis with CQL](offline-rl-cql)

A reproducible offline reinforcement learning project on Minari Walker2d, covering CQL training, periodic rollout evaluation, checkpoint selection, hyperparameter sweeps, critic-count ablations, cloud GPU execution, and failure analysis.

**Topics:** Offline RL · CQL · Continuous control · MuJoCo · Minari · d3rlpy · PyTorch

**Status:** Working research prototype. The training and evaluation pipeline runs successfully; increasing critic count did not reliably improve performance, and no universal improvement over standard CQL is claimed.

## About

This repository is a portfolio of research-oriented implementations. Projects may range from completed systems to clearly labeled experimental prototypes. Performance claims are limited to the evidence documented within each project.
