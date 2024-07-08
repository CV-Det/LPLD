
import logging
import os
import cv2
from collections import OrderedDict
import torch
import torchvision
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.modeling import build_model

logger = logging.getLogger("detectron2")

def visualize(args, model):
    img_path = args.img_path
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    if height < width:
        img = cv2.resize(img, (600*width//height, 600))
    else:
        img = cv2.resize(img, (600, 600*height//width))

    img_tensor = torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1)
    img_list = [{
        "height" : img_tensor.shape[1],
        "width" : img_tensor.shape[2],
        "image" : img_tensor
    }]
    outputs = model(img_list)
    for idx in range(len(outputs[0]["instances"])):
        pred_box = outputs[0]["instances"][idx].pred_boxes.tensor.detach().cpu()
        pred_score = outputs[0]["instances"][idx].scores.item()
        if pred_score >= 0.5:
            bbox = pred_box.numpy().astype(int)[0]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    cv2.imwrite("output.jpg", img)
    return

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(args.model_dir)
    logger.info("Trained model has been sucessfully loaded")
    model.eval()
    return visualize(args, model)

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--img_path", type=str, help="sample path to visualize")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
