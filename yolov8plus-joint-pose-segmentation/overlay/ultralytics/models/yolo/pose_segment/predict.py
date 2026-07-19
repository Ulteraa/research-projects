# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops


class PoseSegmentPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a model that performs both pose estimation and segmentation.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.pose_segment import PoseSegmentPredictor

        args = dict(model='yolov8n-pose-seg.pt', source=ASSETS)
        predictor = PoseSegmentPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes PoseSegmentPredictor with provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose_segment"
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def postprocess(self, preds, img, orig_imgs):
        # for conversion I did follwing changes
        if len(preds)==3:
            a, b, preds_pose =  preds
            preds_seg =(a,b)
        else:
            preds_seg, preds_pose = preds

        # it Was like this for training
        # preds_seg, preds_pose = preds
        """Apply non-maximum suppression and return detections with high confidence scores."""
        p_pose = ops.non_max_suppression(
            preds_pose,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            multi_label=True,
        )

        p_seg = ops.non_max_suppression(
            preds_seg [0],
            self.args.conf,
            self.args.iou,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            multi_label=True,

        )
        # preds_seg = preds[0]
        proto = preds_seg[1][-1] if isinstance(preds_seg[1], tuple) else preds_seg[1]
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        # """Return detection, segmentation, and pose estimation results for a given input image or list of images."""
        # p = ops.non_max_suppression(
        #     preds[0],
        #     self.args.conf,
        #     self.args.iou,
        #     agnostic=self.args.agnostic_nms,
        #     max_det=self.args.max_det,
        #     classes=self.args.classes,
        #     nc=len(self.model.names),
        # )
        #

        results = []
        #proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # Segmentation proto
        #keypoints = preds[2] if len(preds) > 2 else None  # Keypoint predictions
        # proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
        # for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
        for i, (pred_p, pred_s, orig_img, img_path) in enumerate(zip(p_pose, p_seg, orig_imgs, self.batch[0])):
            if  len(pred_p) or  len(pred_s):
                if not len(pred_p):
                    pred_kpts = None
                else:
                    ####################################### keypoint prediction

                    pred_p[:, :4] = ops.scale_boxes(img.shape[2:], pred_p[:, :4], orig_img.shape).round()
                    # !!!!! important Note !!!!! this line of the code should change to following when dealing with model conversion
                    pred_kpts = pred_p[:, 6:].view(len(pred_p), *self.model.kpt_shape) if len(pred_p) else pred_p[:, 6:]
                    # uncomment follwing two lines for conversion and comment the above line but for the training it is the other way around
                    # pred_kpts = pred_p[:, 6:].view(len(pred_p), *self.model.model.yaml['kpt_shape']) if len(pred_p) else pred_p[:, 6:]
                    pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)



                    # results.append(
                    #     Results(orig_img, path=img_path, names=self.model.names, boxes=pred_p[:, :6], keypoints=pred_kpts)
                    # )
                if not len(pred_s):
                    masks = None
                else:
                    ####################################### mask prediction
                    masks = ops.process_mask(proto[i], pred_s[:, 6:], pred_s[:, :4], img.shape[2:], upsample=True)  # HWC
                    #pred_s[:, :4] = ops.scale_boxes(img.shape[2:], pred_s[:, :4], orig_img.shape)




                results.append(
                        Results(orig_img, path=img_path, names=self.model.names, boxes=pred_p[:, :6],  keypoints=pred_kpts, masks=masks))

            #     # Scale bounding boxes
            #     pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
            #
            #     # Process segmentation masks
            #     masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:],
            #                              upsample=True) if proto is not None else None
            #
            #     # Process keypoints
            #     pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if keypoints is not None else None
            #     if pred_kpts is not None:
            #         pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
            #
            # results.append(
            #     Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks,
            #             keypoints=pred_kpts)
            # )
        return results

# # Ultralytics YOLO 🚀, AGPL-3.0 license
#
# from ultralytics.engine.results import Results
# from ultralytics.models.yolo.detect.predict import DetectionPredictor
# from ultralytics.utils import DEFAULT_CFG, ops
#
#
# class SegmentationPredictor(DetectionPredictor):
#     """
#     A class extending the DetectionPredictor class for prediction based on a segmentation model.
#
#     Example:
#         ```python
#         from ultralytics.utils import ASSETS
#         from ultralytics.models.yolo.segment import SegmentationPredictor
#
#         args = dict(model='yolov8n-seg.pt', source=ASSETS)
#         predictor = SegmentationPredictor(overrides=args)
#         predictor.predict_cli()
#         ```
#     """
#
#     def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
#         """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
#         super().__init__(cfg, overrides, _callbacks)
#         self.args.task = "segment"
#
#     def postprocess(self, preds, img, orig_imgs):
#         """Applies non-max suppression and processes detections for each image in an input batch."""
#         p = ops.non_max_suppression(
#             preds[0],
#             self.args.conf,
#             self.args.iou,
#             agnostic=self.args.agnostic_nms,
#             max_det=self.args.max_det,
#             nc=len(self.model.names),
#             classes=self.args.classes,
#         )
#
#         if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
#             orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
#
#         results = []
#         proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]  # tuple if PyTorch model or array if exported
#         for i, (pred, orig_img, img_path) in enumerate(zip(p, orig_imgs, self.batch[0])):
#             if not len(pred):  # save empty boxes
#                 masks = None
#             elif self.args.retina_masks:
#                 pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
#                 masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
#             else:
#                 masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
#                 pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
#             results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks))
#         return results
