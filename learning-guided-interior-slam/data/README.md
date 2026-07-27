# Data

This repository does not redistribute TUM RGB-D data.

Download these sequences directly from the TUM RGB-D benchmark:

```text
rgbd_dataset_freiburg1_room
rgbd_dataset_freiburg1_desk
```

The validated RunPod layout was:

```text
/workspace/interior-slam/third_party/MASt3R-SLAM/datasets/tum/
├── rgbd_dataset_freiburg1_room/
└── rgbd_dataset_freiburg1_desk/
```

Each extracted sequence should contain the original RGB, depth, association, and ground-truth files supplied by TUM.

MASt3R-SLAM trajectories were stored under:

```text
/workspace/interior-slam/third_party/MASt3R-SLAM/logs/
├── tum_fr1_room/rgbd_dataset_freiburg1_room.txt
└── tum_fr1_desk/rgbd_dataset_freiburg1_desk.txt
```

Dataset terms and attribution remain with the TUM RGB-D benchmark.
