from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from detectron2.layers import ShapeSpec

import torch.nn as nn

from deeplab3.modeling.backbone import build_backbone as deeplab_build_backbone
from deeplab3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

__all__ = [
    "DeeplabBackbone"
    "build_deeplab_backbone",
]

class DeeplabBackbone(Backbone):

    def __init__(self, cfg):
        super().__init__()

        if cfg.MODEL.DEEPLAB.BACKBONE == 'drn':
            output_stride = 8
        else:
            output_stride = 16

        if cfg.MODEL.DEEPLAB.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.model = deeplab_build_backbone(cfg.MODEL.DEEPLAB.BACKBONE, output_stride, BatchNorm, cfg.INPUT.USE_DEPTH, use_deeplab_format=False)

    def forward(self, input):
        return self.model(input)

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self.model._out_feature_channels[name], stride=self.model._out_feature_strides[name]
            )
            for name in self.model._out_features
        }


@BACKBONE_REGISTRY.register()
def build_deeplab_backbone(cfg, input_shape):
    return DeeplabBackbone(cfg)
