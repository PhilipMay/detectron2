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


def fpn_forward(self, inputs):
    """ Returns output of the final convolution layer """

    logger.info('### EffNet Block inputs {}'.format(inputs.size()))

    outputs = {}

    # Stem
    x = self._swish(self._bn0(self._conv_stem(inputs)))

    logger.info('### EffNet Block stem {}'.format(x.size()))

    # Blocks
    for idx, block in enumerate(self._blocks):
        drop_connect_rate = self._global_params.drop_connect_rate
        if drop_connect_rate:
            drop_connect_rate *= float(idx) / len(self._blocks)
        x = block(x, drop_connect_rate=drop_connect_rate)

        logger.info('### EffNet Block {} - {} - {}'.format(idx, x.size(), block._block_args))

        if idx == 31:
            outputs['res5'] = x
        elif idx == 21:
            outputs['res4'] = x
        elif idx == 9:
            outputs['res3'] = x
        elif idx == 5:
            outputs['res2'] = x

    # Head
    # x = self._swish(self._bn1(self._conv_head(x)))

    return outputs


class EffNet(Backbone):
    def __init__(self):
        super(EffNet, self).__init__()
        #logger.info("EfficientNet.__init__")
        self._eff_model = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)
        self._eff_model.fpn_forward = fpn_forward
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
    logger.info("build_resnet_backbone input_shape: {}".format(input_shape))
    assert input_shape.channels == 3

    return EffNet()
