# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.nn as nn
import copy
import torch
from typing import Union, List, Dict, Any, cast
from detectron2.modeling.backbone import (
    Backbone,
    BACKBONE_REGISTRY
)
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool, LastLevelP6P7
from detectron2.layers import get_norm


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                norm = get_norm("BN", v)
                layers += [conv2d, norm, nn.ReLU(inplace=True)]
            else:
                norm = get_norm("FrozenBN", v)
                layers += [conv2d, norm, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class vgg_backbone(Backbone):
    def __init__(self, cfg):
        super().__init__()
        self.vgg = make_layers(cfgs['vgg16'], batch_norm=False)
        self._initialize_weights()
        _out_feature_channels = [64, 128, 256, 512, 512]
        _out_feature_strides = [2, 4, 8, 16, 32]

        self.stages = [nn.Sequential(*list(self.vgg._modules.values())[0:7]),\
                    nn.Sequential(*list(self.vgg._modules.values())[7:14]),\
                    nn.Sequential(*list(self.vgg._modules.values())[14:24]),\
                    nn.Sequential(*list(self.vgg._modules.values())[24:34]),\
                    nn.Sequential(*list(self.vgg._modules.values())[34:]),]
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        self._stage_names = []

        for i, stage in enumerate(self.stages):
            name = "vgg{}".format(i)
            self.add_module(name, stage)
            self._stage_names.append(name)
            self._out_feature_channels[name] = _out_feature_channels[i]
            self._out_feature_strides[name] = _out_feature_strides[i]

        self._out_features = self._stage_names

        del self.vgg

    def forward(self, x):
        features = {}
        for name, stage in zip(self._stage_names, self.stages):
            x = stage(x)
            features[name] = x
        return features

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_backbone(cfg, _):
    return vgg_backbone(cfg)

@BACKBONE_REGISTRY.register() #already register in baseline model
def build_vgg_fpn_backbone(cfg, _):
    bottom_up = vgg_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
    )

    return backbone