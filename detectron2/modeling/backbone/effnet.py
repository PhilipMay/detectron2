import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY


class EffNet(Backbone):
    def __init__(self):
        self._eff_model = EfficientNet.from_pretrained("efficientnet-b1", advprop=True)
        super(EffNet, self).__init__()

    def forward(self, x):
        return self._eff_model.extract_features(x)

    def output_shape(self):
        return

    def freeze(self, freeze_at=0):
        return self


@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg, input_shape):
    """
    Create a ResNet instance from config.

    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    logger = logging.getLogger(__name__)
    logger.info("build_resnet_backbone input_shape: {}".format(input_shape))

    return EffNet()
