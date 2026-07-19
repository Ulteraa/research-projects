# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseSegmentModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class PoseSegmentTrainer(yolo.detect.DetectionTrainer):
    """
    A trainer class that combines pose estimation and instance segmentation tasks.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize PoseSegmentTrainer with specific configurations."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose_segment"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return PoseSegmentModel initialized with specified config and weights."""
        model = PoseSegmentModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an instance of PoseSegmentValidator for validation of the pose-segment model."""
        self.loss_names = "box_loss", "seg_loss","pose_loss", "cls_loss", "dfl_loss", "kobj"
        return yolo.pose_segment.PoseSegmentValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with pose and segmentation labels."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch.get("masks"),
            kpts=batch.get("keypoints"),
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/validation metrics for pose and segmentation."""
        plot_results(file=self.csv, segment=True, pose=True, on_plot=self.on_plot)


# class SegmentationTrainer(yolo.detect.DetectionTrainer):
#     """
#     A class extending the DetectionTrainer class for training based on a segmentation model.
#
#     Example:
#         ```python
#         from ultralytics.models.yolo.segment import SegmentationTrainer
#
#         args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
#         trainer = SegmentationTrainer(overrides=args)
#         trainer.train()
#         ```
#     """
#
#     def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
#         """Initialize a SegmentationTrainer object with given arguments."""
#         if overrides is None:
#             overrides = {}
#         overrides["task"] = "segment"
#         super().__init__(cfg, overrides, _callbacks)
#
#     def get_model(self, cfg=None, weights=None, verbose=True):
#         """Return SegmentationModel initialized with specified config and weights."""
#         model = SegmentationModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1)
#         if weights:
#             model.load(weights)
#
#         return model
#
#     def get_validator(self):
#         """Return an instance of SegmentationValidator for validation of YOLO model."""
#         self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
#         return yolo.segment.SegmentationValidator(
#             self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
#         )
#
#     def plot_training_samples(self, batch, ni):
#         """Creates a plot of training sample images with labels and box coordinates."""
#         plot_images(
#             batch["img"],
#             batch["batch_idx"],
#             batch["cls"].squeeze(-1),
#             batch["bboxes"],
#             masks=batch["masks"],
#             paths=batch["im_file"],
#             fname=self.save_dir / f"train_batch{ni}.jpg",
#             on_plot=self.on_plot,
#         )
#
#     def plot_metrics(self):
#         """Plots training/val metrics."""
#         plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
