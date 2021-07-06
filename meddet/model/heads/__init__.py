

from ..registry import HEADS

from meddet.model.heads.dense_heads.retina_head import RetinaHead
from meddet.model.heads.dense_heads.rpn_head import RPNHead
from meddet.model.heads.dense_heads.deeplung_head import DeepLungHead
from meddet.model.heads.dense_heads.centernet_head import CenterNetHead
from meddet.model.heads.roi_heads.bbox_heads import ConvFCBBoxHead, DoubleBBoxHead
from meddet.model.heads.roi_heads.base_roi_head import ROIHead

HEADS.register_module(RetinaHead)
HEADS.register_module(DeepLungHead)
HEADS.register_module(RPNHead)
HEADS.register_module(CenterNetHead)
HEADS.register_module(ConvFCBBoxHead)
HEADS.register_module(DoubleBBoxHead)
HEADS.register_module(ROIHead)