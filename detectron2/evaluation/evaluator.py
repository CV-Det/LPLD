# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
import os
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.file_io import PathManager
import xml.etree.ElementTree as ET
import cv2

import pdb

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of both bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)
    return iou

def draw_predictions_with_gt(filename, predictions):
    img = cv2.imread(filename)
    if "visible" in filename:
        ann_file_path = filename.replace("JPEGImages", "Annotations").replace("visible/", "").replace(".jpg", ".xml")
    elif "lwir" in filename:
        ann_file_path = filename.replace("JPEGImages", "Annotations").replace("lwir/", "").replace(".jpg", ".xml")
    else:
        ann_file_path = filename.replace("JPEGImages", "Annotations").replace(".jpg", ".xml")

    with PathManager.open(ann_file_path) as f:
        tree = ET.parse(f)

    gt_boxes = []
    if "kaist" in filename:
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls != "person":
                continue

            gt_bbox = obj.find("bndbox")
            gt_bbox = [float(gt_bbox.find(x).text) for x in ["x", "y", "w", "h"]]
            gt_bbox[2], gt_bbox[3] = gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]
            gt_bbox[0] -= 1.0
            gt_bbox[1] -= 1.0
            gt_bbox = [int(coord) for coord in gt_bbox]
            gt_boxes.append(gt_bbox)
            #cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)

    elif "flir" in filename:
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls in ["rider", "train"]:
                continue

            gt_bbox = obj.find("bndbox")
            gt_bbox = [
                int(gt_bbox.find("xmin").text),
                int(gt_bbox.find("ymin").text),
                int(gt_bbox.find("xmax").text),
                int(gt_bbox.find("ymax").text),
            ]
            gt_boxes.append(gt_bbox)
            #cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)

    elif "watercolor" in filename:
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in ["bicycle", "bird", "car", "cat", "dog", "person"]:
                continue

            gt_bbox = obj.find("bndbox")
            gt_bbox = [
                int(gt_bbox.find("xmin").text),
                int(gt_bbox.find("ymin").text),
                int(gt_bbox.find("xmax").text),
                int(gt_bbox.find("ymax").text),
            ]
            gt_boxes.append(gt_bbox)
            #cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)

    elif "clipart" in filename:
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in ( "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                            "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
                            "pottedplant", "sheep", "sofa", "train", "tvmonitor"):
                continue

            gt_bbox = obj.find("bndbox")
            gt_bbox = [
                int(gt_bbox.find("xmin").text),
                int(gt_bbox.find("ymin").text),
                int(gt_bbox.find("xmax").text),
                int(gt_bbox.find("ymax").text),
            ]
            gt_boxes.append(gt_bbox)

    else:
        for obj in tree.findall("object"):
            cls = obj.find("name").text 
            if cls not in ( 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'):
                continue

            gt_bbox = obj.find("bndbox")
            gt_bbox = [
                int(gt_bbox.find("xmin").text),
                int(gt_bbox.find("ymin").text),
                int(gt_bbox.find("xmax").text),
                int(gt_bbox.find("ymax").text),
            ]
            gt_boxes.append(gt_bbox)
            #cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), 2)

    draw_box = []
    num_objects = len(predictions["instances"])
    for gt_bbox in gt_boxes:
        max_idx = -1
        max_iou = 0
        for i in range(num_objects):
            pred_bbox = predictions["instances"].pred_boxes.tensor[i].cpu().numpy()
            pred_bbox = [int(coord) for coord in pred_bbox]

            iou = calculate_iou(gt_bbox, pred_bbox)
            if iou > 0.5:
                if iou > max_iou:
                    max_iou = iou
                    max_idx = i
            

        if max_idx != -1:
            pred_bbox = predictions["instances"].pred_boxes.tensor[max_idx].cpu().numpy()
            pred_bbox = [int(coord) for coord in pred_bbox]
            # cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
            draw_box.append(pred_bbox)

    # for i in range(num_objects):
    #     pred_bbox = predictions["instances"].pred_boxes.tensor[i].cpu().numpy()
    #     pred_bbox = [int(coord) for coord in pred_bbox]

    #     # Check IoU with all ground truth boxes
    #     for gt_bbox in gt_boxes:
    #         iou = calculate_iou(gt_bbox, pred_bbox)

    #         # Draw the prediction box if IoU is above 50%
    #         if iou > 0.5:
    #             cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
    #             break  # Break to avoid drawing the same prediction box multiple times

    #cv2.imwrite(os.path.join(output_dir, filename.split("/")[-1]), img)
    return img, draw_box

def inference_on_dataset(
    model, data_loader, evaluator, draw=False
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        imgs, draw_boxes = [], []
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if draw:
                img, draw_box = draw_predictions_with_gt(inputs[0]["file_name"], outputs[0])
                imgs.append(img)
                draw_boxes.append(draw_box)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
            #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
            #     log_every_n_seconds(
            #         logging.INFO,
            #         (
            #             f"Inference done {idx + 1}/{total}. "
            #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
            #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
            #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
            #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
            #             f"ETA={eta}"
            #         ),
            #         n=5,
            #     )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    
    if not draw:
        return results

    return results, imgs, draw_boxes

def inference_on_corruption_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            base_dict = {k: v for k, v in inputs[0].items() if "image" not in k}
            base_dict["image_id"] = inputs[0]["image_id"]
            for severity in range(4,5):
                corrupt_inputs = base_dict.copy()
                corrupt_inputs["image"] = inputs[0]["image_" + str(severity)]
                corrupt_inputs = [corrupt_inputs]
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(corrupt_inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(corrupt_inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                #     log_every_n_seconds(
                #         logging.INFO,
                #         (
                #             f"Inference done {idx + 1}/{total}. "
                #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
                #             f"ETA={eta}"
                #         ),
                #         n=5,
                #     )
                start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    # logger.info(
    #     "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_time_str, total_time / (total - num_warmup), num_devices
    #     )
    # )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # logger.info(
    #     "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
    #         total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
    #     )
    # )
    
    ###### results = OrderedDict([('bbox', {'AP': 0.454, 'AP50': 0.454, 'AP75': 0.454})])
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
