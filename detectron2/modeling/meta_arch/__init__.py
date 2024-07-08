# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

from .panoptic_fpn import PanopticFPN

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .student_sfda_rcnn import student_sfda_RCNN
from .teacher_sfda_rcnn import teacher_sfda_RCNN
from .student_sfda_rcnn_jy import student_sfda_RCNN_jy
from .teacher_sfda_rcnn_jy import teacher_sfda_RCNN_jy
from .student_sfda_rcnn_irg import student_sfda_RCNN_IRG
from .teacher_sfda_rcnn_irg import teacher_sfda_RCNN_IRG
from .single_sfda_rcnn import single_sfda_RCNN
from .retinanet import RetinaNet
from .semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head

__all__ = list(globals().keys())
