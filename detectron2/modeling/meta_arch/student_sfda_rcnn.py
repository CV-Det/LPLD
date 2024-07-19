# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, pairwise_iou, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.layers import cat

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

import torch.nn.functional as F

__all__ = ["student_sfda_RCNN"]

@META_ARCH_REGISTRY.register()
class student_sfda_RCNN(nn.Module):
    """
    student_sfda_RCNN R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], cfg=None, model_teacher=None,
                t_features=None, t_proposals=None, t_results=None, mode="test", ablation=1):
        
        if not self.training and mode == "test":
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs, mode)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, t_results) #t_results for gt_instances
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        _, detector_losses = self.roi_heads(images, features, proposals, t_results)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # Extract features from the student and teacher models
        s_box_features = self.roi_heads._shared_roi_transform([features['res4']], [t_proposals[0].proposal_boxes]) #s_box_features = 300,2048,7,7
        s_box_features_mean = s_box_features.mean(dim=[2, 3])
        s_box_features_norm = F.normalize(s_box_features_mean, dim=1)
        s_roih_logits = self.roi_heads.box_predictor(s_box_features_mean)
        
        t_box_features = model_teacher.roi_heads._shared_roi_transform([t_features['res4']], [t_proposals[0].proposal_boxes])
        t_box_features_mean = t_box_features.mean(dim=[2, 3])
        t_box_features_norm = F.normalize(t_box_features_mean, dim=1)
        t_roih_logits = model_teacher.roi_heads.box_predictor(t_box_features_mean)

        # Compute the cosine similarity between the student and teacher features
        c_similarity = F.cosine_similarity(s_box_features_norm.detach(), t_box_features_norm.detach(), dim=1)
        t_roih_classes = t_roih_logits[0].argmax(dim=1)

        # Compute the LPL loss
        t_proposal_boxes = cat([p.proposal_boxes.tensor for p in t_proposals], dim=0)
        t_boxes = model_teacher.roi_heads.box_predictor.box2box_transform.apply_deltas(t_roih_logits[1], t_proposal_boxes)
        t_box = torch.zeros(t_proposal_boxes.shape[0], 4).cuda()

        for index, cl in enumerate(t_roih_classes):
            if cl == self.roi_heads.num_classes:
                t_box[index] = t_proposal_boxes[index]
            else:
                t_box[index] =  t_boxes[index, 4*cl:4*cl+4]

        t_box_boxes = Boxes(t_box)
        t_result_boxes = Boxes(t_results[0].gt_boxes.tensor)
        iou_matrix = pairwise_iou(t_box_boxes, t_result_boxes)

        if iou_matrix.shape[1] == 0:
            return losses

        t_indices = torch.nonzero(torch.max(iou_matrix, dim=1).values <= 0.4).flatten()
        if t_indices.nelement() == 0:
            return losses

        t_softmax_w_bg = F.softmax(t_roih_logits[0][t_indices], dim=1)
        t_softmax_wo_bg = F.softmax(t_roih_logits[0][t_indices][:, :-1], dim=1)
        wo_bg_max = torch.max(t_softmax_wo_bg, dim=1)[0]
        w_bg_prob = t_softmax_w_bg[:, -1]
        t_indices_filtered = t_indices[(wo_bg_max >= 0.9) & (w_bg_prob <= 0.99)]

        if t_indices_filtered.nelement() == 0:
            return losses

        losses["lpl_kl"] = self._kl_divergence(s_roih_logits[0][t_indices_filtered],
                                               t_roih_logits[0][t_indices_filtered][:, :-1],
                                               weight = 1-c_similarity[t_indices_filtered])

        return losses

    def _kl_divergence(self, student_logits, teacher_logits, weight=None):
        teacher_probs = F.softmax(teacher_logits, dim=1)
        student_probs = F.softmax(student_logits, dim=1)
        teacher_probs = torch.cat([teacher_probs, 1e-10*torch.ones(teacher_probs.size(0), 1).cuda()], dim=1)
        KL_loss = teacher_probs * (teacher_probs.log() - student_probs.log())
        if weight is not None:
            KL_loss = (KL_loss.sum(1)*weight).sum()/student_probs.size(0)
        else:
            KL_loss = KL_loss.sum()/student_probs.size(0)

        return KL_loss/10

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _= self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return student_sfda_RCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]], mode = "test"):
        """
        Normalize, pad and batch the input images.
        """
        if mode == "train":
            images = [x["image_strong"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        elif mode == "test":
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results