import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn.functional as F
from torch import nn
from .model import EfficientNet

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

logger = logging.getLogger(__name__)


class EffNet(Backbone):
    def __init__(self):
        super(EffNet, self).__init__()
        #logger.info("EfficientNet.__init__")
        self._eff_model = EfficientNet.from_pretrained("efficientnet-b6", advprop=True)
        #for idx, block in enumerate(self._eff_model._blocks):
            #logger.info("##################################")
            #logger.info("EfficientNet extract_features: {} - {}".format(idx, block))
            #logger.info("EfficientNet extract_features: {}".format(block._block_args))

    def forward(self, x):
        outputs = self._eff_model.fpn_forward(x)
        return outputs

    def output_shape(self):
        result = {
            'res2': ShapeSpec(
                channels=32, stride=4
            ),
            'res3': ShapeSpec(
                channels=56, stride=8
            ),
            'res4': ShapeSpec(
                channels=160, stride=16
            ),
            'res5': ShapeSpec(
                channels=448, stride=32
            ),
        }

        #logger.info("EffNet output_shape: {}".format(result))

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

    logger.info("build_resnet_backbone input_shape: {}".format(input_shape))
    assert input_shape.channels == 3

    return EffNet()
