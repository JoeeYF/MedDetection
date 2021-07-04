

from ..registry import LOSSES

from .cross_entropy_loss import CrossEntropyLoss
# from .focal_loss import FocalLoss
from .smooth_l1_loss import SmoothL1Loss
from .fcloss import SigmoidFocalLoss
# from .mm_focal_loss import MMFocalLoss
from .combine_loss import CombineLoss
from .l1loss import BalancedL1Loss
from .iou_loss import IoULoss, GIoULoss, DIoULoss

# LOSSES.register_module(DiceAndCrossEntropyLoss)
LOSSES.register_module(CrossEntropyLoss)
# LOSSES.register_module(FocalLoss)
LOSSES.register_module(SigmoidFocalLoss)
LOSSES.register_module(SmoothL1Loss)
# LOSSES.register_module(MMFocalLoss)
LOSSES.register_module(CombineLoss)
LOSSES.register_module(BalancedL1Loss)
LOSSES.register_module(IoULoss)
LOSSES.register_module(GIoULoss)
LOSSES.register_module(DIoULoss)
