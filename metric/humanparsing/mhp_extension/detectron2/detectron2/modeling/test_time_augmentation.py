# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
from contextlib import contextmanager
from itertools import count
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.structures import Instances

from .meta_arch import GeneralizedRCNN
from .postprocessing import detector_postprocess
from .roi_heads.fast_rcnn import fast_rcnn_inference_single_image

__all__ = ["DatasetMapperTTA", "GeneralizedRCNNWithTTA"]


class DatasetMapperTTA:
    """
    Implement test-time augmentation for detection data.
    It is a callable which takes a dataset dict from a detection dataset,
    and returns a list of dataset dicts where the images
    are augmented from the input image by the transformations defined in the config.
    This is used for test-time augmentation.
    """

    def __init__(self, cfg):
        self.min_sizes = cfg.TEST.AUG.MIN_SIZES
        self.max_size = cfg.TEST.AUG.MAX_SIZE
        self.flip = cfg.TEST.AUG.FLIP
        self.image_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a detection dataset dict

        Returns:
            list[dict]:
                a list of dataset dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
        """
        ret = []
        if "image" not in dataset_dict:
            numpy_image = read_image(dataset_dict["file_name"], self.image_format)
        else:
            numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy().astype("uint8")
        for min_size in self.min_sizes:
            image = np.copy(numpy_image)
            tfm = ResizeShortestEdge(min_size, self.max_size).get_transform(image)
            resized = tfm.apply_image(image)
            resized = torch.as_tensor(resized.transpose(2, 0, 1).astype("float32"))

            dic = copy.deepcopy(dataset_dict)
            dic["horiz_flip"] = False
            dic["image"] = resized
            ret.append(dic)

            if self.flip:
                dic = copy.deepcopy(dataset_dict)
                dic["horiz_flip"] = True
                dic["image"] = torch.flip(resized, dims=[2])
                ret.append(dic)
        return ret


class GeneralizedRCNNWithTTA(nn.Module):
    """
    A GeneralizedRCNN with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        assert isinstance(
            model, GeneralizedRCNN
        ), "TTA is only supported on GeneralizedRCNN. Got a model of type {}".format(type(model))
        self.cfg = cfg.clone()
        assert not self.cfg.MODEL.KEYPOINT_ON, "TTA for keypoint is not supported yet"
        assert (
            not self.cfg.MODEL.LOAD_PROPOSALS
        ), "TTA for pre-computed proposals is not supported yet"

        self.model = model

        if tta_mapper is None:
            tta_mapper = DatasetMapperTTA(cfg)
        self.tta_mapper = tta_mapper
        self.batch_size = batch_size

    @contextmanager
    def _turn_off_roi_heads(self, attrs):
        """
        Open a context where some heads in `model.roi_heads` are temporarily turned off.
        Args:
            attr (list[str]): the attribute in `model.roi_heads` which can be used
                to turn off a specific head, e.g., "mask_on", "keypoint_on".
        """
        roi_heads = self.model.roi_heads
        old = {}
        for attr in attrs:
            try:
                old[attr] = getattr(roi_heads, attr)
            except AttributeError:
                # The head may not be implemented in certain ROIHeads
                pass

        if len(old.keys()) == 0:
            yield
        else:
            for attr in old.keys():
                setattr(roi_heads, attr, False)
            yield
            for attr in old.keys():
                setattr(roi_heads, attr, old[attr])

    def _batch_inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=do_postprocess,
                    )
                )
                inputs, instances = [], []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """
        return [self._inference_one_image(x) for x in batched_inputs]

    def _detector_postprocess(self, outputs, aug_vars):
        return detector_postprocess(outputs, aug_vars["height"], aug_vars["width"])

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict

        Returns:
            dict: one output dict
        """

        augmented_inputs, aug_vars = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        with self._turn_off_roi_heads(["mask_on", "keypoint_on"]):
            # temporarily disable roi heads
            all_boxes, all_scores, all_classes = self._get_augmented_boxes(
                augmented_inputs, aug_vars
            )
        merged_instances = self._merge_detections(
            all_boxes, all_scores, all_classes, (aug_vars["height"], aug_vars["width"])
        )

        if self.cfg.MODEL.MASK_ON:
            # Use the detected boxes to obtain new fields
            augmented_instances = self._rescale_detected_boxes(
                augmented_inputs, merged_instances, aug_vars
            )
            # run forward on the detected boxes
            outputs = self._batch_inference(
                augmented_inputs, augmented_instances, do_postprocess=False
            )
            # Delete now useless variables to avoid being out of memory
            del augmented_inputs, augmented_instances, merged_instances
            # average the predictions
            outputs[0].pred_masks = self._reduce_pred_masks(outputs, aug_vars)
            # postprocess
            output = self._detector_postprocess(outputs[0], aug_vars)
            return {"instances": output}
        else:
            return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)

        do_hflip = [k.pop("horiz_flip", False) for k in augmented_inputs]
        heights = [k["height"] for k in augmented_inputs]
        widths = [k["width"] for k in augmented_inputs]
        assert (
            len(set(heights)) == 1 and len(set(widths)) == 1
        ), "Augmented version of the inputs should have the same original resolution!"
        height = heights[0]
        width = widths[0]
        aug_vars = {"height": height, "width": width, "do_hflip": do_hflip}

        return augmented_inputs, aug_vars

    def _get_augmented_boxes(self, augmented_inputs, aug_vars):
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs, do_postprocess=False)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for idx, output in enumerate(outputs):
            rescaled_output = self._detector_postprocess(output, aug_vars)
            pred_boxes = rescaled_output.pred_boxes.tensor
            if aug_vars["do_hflip"][idx]:
                pred_boxes[:, [0, 2]] = aug_vars["width"] - pred_boxes[:, [2, 0]]
            all_boxes.append(pred_boxes)
            all_scores.extend(rescaled_output.scores)
            all_classes.extend(rescaled_output.pred_classes)
        all_boxes = torch.cat(all_boxes, dim=0).cpu()
        return all_boxes, all_scores, all_classes

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        # select from the union of all results
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )

        return merged_instances

    def _rescale_detected_boxes(self, augmented_inputs, merged_instances, aug_vars):
        augmented_instances = []
        for idx, input in enumerate(augmented_inputs):
            actual_height, actual_width = input["image"].shape[1:3]
            scale_x = actual_width * 1.0 / aug_vars["width"]
            scale_y = actual_height * 1.0 / aug_vars["height"]
            pred_boxes = merged_instances.pred_boxes.clone()
            pred_boxes.tensor[:, 0::2] *= scale_x
            pred_boxes.tensor[:, 1::2] *= scale_y
            if aug_vars["do_hflip"][idx]:
                pred_boxes.tensor[:, [0, 2]] = actual_width - pred_boxes.tensor[:, [2, 0]]

            aug_instances = Instances(
                image_size=(actual_height, actual_width),
                pred_boxes=pred_boxes,
                pred_classes=merged_instances.pred_classes,
                scores=merged_instances.scores,
            )
            augmented_instances.append(aug_instances)
        return augmented_instances

    def _reduce_pred_masks(self, outputs, aug_vars):
        for idx, output in enumerate(outputs):
            if aug_vars["do_hflip"][idx]:
                output.pred_masks = output.pred_masks.flip(dims=[3])
        all_pred_masks = torch.stack([o.pred_masks for o in outputs], dim=0)
        avg_pred_masks = torch.mean(all_pred_masks, dim=0)
        return avg_pred_masks
