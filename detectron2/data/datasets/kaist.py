import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

__all__ = ["load_kaist_viz_instances", "register_kaist_viz",
           "load_kaist_ir_instances", "register_kaist_ir"]

CLASS_NAMES = ( 'person', 'people', 'cyclist', 'person?')
CLASS_NAMES_PERSON = ('person', 'background')

def load_kaist_viz_instances(dirname: str, split: str,
                             class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Kaist Viz detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        viz_fileid = "/".join(fileid.split("/")[:-1]) + "/visible/" + fileid.split("/")[-1]
        jpeg_file = os.path.join(dirname, "JPEGImages", viz_fileid+".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": viz_fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls == "person?" or cls not in class_names:
                continue

            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["x", "y", "w", "h"]]
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def load_kaist_tr_instances(dirname: str, split: str,
                             class_names: Union[List[str], Tuple[str, ...]]):
    """
    Load Kaist Viz detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    # Needs to read many small annotation files. Makes sense at local

    annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + ".xml")
        tr_fileid = "/".join(fileid.split("/")[:-1]) + "/lwir/" + fileid.split("/")[-1]
        jpeg_file = os.path.join(dirname, "JPEGImages", tr_fileid+".jpg")

        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)

        r = {
            "file_name": jpeg_file,
            "image_id": tr_fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls == "person?" or cls not in class_names:
                continue

            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["x", "y", "w", "h"]]
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": class_names.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def register_kaist_viz(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_kaist_viz_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_kaist_tr(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda: load_kaist_tr_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_kaist_viz_person(name, dirname, split, year, class_names=CLASS_NAMES_PERSON):
    DatasetCatalog.register(name, lambda: load_kaist_viz_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )

def register_kaist_tr_person(name, dirname, split, year, class_names=CLASS_NAMES_PERSON):
    DatasetCatalog.register(name, lambda: load_kaist_tr_instances(dirname, split, class_names))
    MetadataCatalog.get(name).set(
        thing_classes=list(class_names), dirname=dirname, year=year, split=split
    )