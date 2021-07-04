

from ..registry import METRICS

from .dice import Dice
from .roc_auc import AUC
from .iou import IOU
from .acc import Acc
from .dist import Dist

METRICS.register_module(Dice)
METRICS.register_module(AUC)
METRICS.register_module(IOU)
METRICS.register_module(Acc)
METRICS.register_module(Dist)