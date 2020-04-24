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
        super(EffNet, self).__init__()
        self._eff_model = EfficientNet.from_pretrained("efficientnet-b1", advprop=True)

    def forward(self, x):
        outputs = {}
        outputs['res4'] = self._eff_model.extract_features(x)
        return outputs

    def output_shape(self):
        result = {
            'res4': ShapeSpec(
                channels=1024, stride=16
            )
        }

        logger = logging.getLogger(__name__)
        logger.info("build_resnet_backbone output_shape: {}".format(result))

        return result
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
