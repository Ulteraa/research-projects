# Framework changes

The overlay targets Ultralytics v8.2.74. Its main changes are:

- `ultralytics/nn/modules/head.py`: adds the unified `PoseSegment` head with
  detection, mask-prototype/mask-coefficient, and keypoint branches.
- `ultralytics/nn/tasks.py`: registers `PoseSegmentModel` and its loss.
- `ultralytics/utils/loss.py`: adds `v8PoseSegmentLoss` for box,
  classification, DFL, mask, keypoint-location, and keypoint-visibility terms.
- `ultralytics/models/yolo/pose_segment/`: adds training, validation, and
  prediction task implementations.
- `ultralytics/utils/metrics.py`: adds joint box, mask, and pose metric
  aggregation.
- `ultralytics/data/`: supports a label record that contains a box, keypoints,
  and an instance polygon for each object.
- `ultralytics/engine/exporter.py`: handles pose metadata and the extra joint
  outputs during export.
- Task/model configuration and dispatch files register `pose_segment` across
  the Ultralytics API.

The smaller edits in the overlay are integration points required to pass the
new task through existing dataset, trainer, validator, plotting, model, and
export code.
