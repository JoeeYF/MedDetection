

from ..registry import BACKBONES

from .resnet import ResNet
from .resnext import ResNeXt

from .deeplung_bk import DeepLungBK


BACKBONES.register_module(ResNet)
BACKBONES.register_module(ResNeXt)

BACKBONES.register_module(DeepLungBK)






