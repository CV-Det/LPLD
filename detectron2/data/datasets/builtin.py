# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes_org import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

from .clipart import register_clipart
from .watercolor import register_watercolor
from .cityscape import register_cityscape
from .sim10k import register_sim10k
from .kitti import register_kitti
from .cityscape_car import register_cityscape_car
from .kaist import register_kaist_viz, register_kaist_tr
from .kaist import register_kaist_viz_person, register_kaist_tr_person
from .flir import register_flir
from .bdd100k import register_bdd

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_minival_100": ("coco/val2014", "coco/annotations/instances_minival2014_100.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2014_minival_100": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014_100.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

# ==== Predefined splits for Clipart ===========
def register_all_clipart(root):
    SPLITS = [
        ("clipart_train", "clipart", "train"),
        ("clipart_test", "clipart", "test"),
    ]
    for name, dirname, split in SPLITS:
        register_clipart(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "clipart"

# ==== Predefined splits for Watercolor ===========
def register_all_watercolor(root):
    SPLITS = [
        ("watercolor_train", "watercolor", "train"),
        ("watercolor_test", "watercolor", "test"),
    ]
    for name, dirname, split in SPLITS:
        register_watercolor(name, os.path.join(root, dirname), split)
        MetadataCatalog.get(name).evaluator_type = "watercolor"

# ==== Predefined splits for Cityscapes ===========
def register_all_cityscape(root):
    SPLITS = [
        ("cityscape_2007_train_s", "cityscape/VOC2007", "train_s"),
        ("cityscape_2007_test_s", "cityscape/VOC2007", "test_s"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_cityscape(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "cityscape"

# ==== Predefined splits for Kaist viz ===========
def register_all_kaist_viz(root):
    SPLITS = [
        ("kaist_train_viz", "kaist", "train-all-20"),
        ("kaist_test_viz", "kaist", "test-all-20"),
    ]
    for name, dirname, split in SPLITS:
        register_kaist_viz(name, os.path.join(root, dirname), split, 2007)
        MetadataCatalog.get(name).evaluator_type = "kaist_viz"

# ==== Predefined splits for Kaist tr ===========
def register_all_kaist_tr(root):
    SPLITS = [
        ("kaist_train_tr", "kaist", "train-all-20"),
        ("kaist_test_tr", "kaist", "test-all-20"),
    ]
    for name, dirname, split in SPLITS:
        register_kaist_tr(name, os.path.join(root, dirname), split, 2007)
        MetadataCatalog.get(name).evaluator_type = "kaist_tr"

# ==== Predefined splits for Kaist viz ===========
def register_all_kaist_viz_person(root):
    SPLITS = [
        ("kaist_train_viz_person", "kaist", "train-all-20"),
        ("kaist_test_viz_person", "kaist", "test-all-20"),
    ]
    for name, dirname, split in SPLITS:
        register_kaist_viz_person(name, os.path.join(root, dirname), split, 2007)
        MetadataCatalog.get(name).evaluator_type = "kaist_viz_person"

# ==== Predefined splits for Kaist tr ===========
def register_all_kaist_tr_person(root):
    SPLITS = [
        ("kaist_train_tr_person", "kaist", "train_all"),
        ("kaist_test_tr_person", "kaist", "test_all"),
    ]
    for name, dirname, split in SPLITS:
        register_kaist_tr_person(name, os.path.join(root, dirname), split, 2007)
        MetadataCatalog.get(name).evaluator_type = "kaist_tr_person"

# ==== Predefined splits for Foggy ===========
def register_all_foggy(root):
    SPLITS = [
        ("cityscape_2007_train_t", "cityscape/VOC2007", "train_t"),
        ("cityscape_2007_test_t", "cityscape/VOC2007", "test_t"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_cityscape(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "foggy"

# ==== Predefined splits for Sim10k ===========
def register_all_sim10k(root):
    SPLITS = [
        ("sim10k_2012_train_s", "sim10k/VOC2012", "val"),
        ("sim10k_2012_test_s", "sim10k/VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_sim10k(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "sim10k"

# ==== Predefined splits for Kitti ===========
def register_all_kitti(root):
    SPLITS = [
        ("kitti_2012_train_s", "kitti/VOC2012", "trainval"),
        ("kitti_2012_test_s", "kitti/VOC2012", "trainval"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_kitti(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "kitti"

# ==== Predefined splits for Cityscapes ===========
def register_all_cityscape_car(root):
    SPLITS = [
        ("cityscape_car_2007_train_s", "cityscape/VOC2007", "train_s"),
        ("cityscape_car_2007_test_s", "cityscape/VOC2007", "test_s"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_cityscape_car(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "cityscape_car"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

def register_all_bdd(root):
    SPLITS = [
        ("bdd100k_2007_train_t", "bdd/VOC2007", "train"),
        ("bdd100k_2007_test_t", "bdd/VOC2007", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_bdd(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "bdd100k"

def register_all_flir(root):
    SPLITS = [
        ("flir_train", "FLIR_VOC", "train_5"),
        ("flir_test", "FLIR_VOC", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_flir(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "flir"

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./dataset`.
    _root = os.getenv("DETECTRON2_DATASETS", "dataset")
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_cityscapes(_root)
    register_all_cityscapes_panoptic(_root)
    register_all_pascal_voc(_root)
    register_all_ade20k(_root)

    register_all_clipart(_root)
    register_all_watercolor(_root)
    register_all_cityscape(_root)
    register_all_foggy(_root)
    register_all_sim10k(_root)
    register_all_kitti(_root)
    register_all_cityscape_car(_root)

    register_all_kaist_viz(_root)
    register_all_kaist_tr(_root)
    register_all_kaist_tr_person(_root)
    register_all_kaist_viz_person(_root)

    register_all_bdd(_root)
    register_all_flir(_root)
