

from ..registry import NECKS

from .fpn import FPN
from .dec import Decoder

NECKS.register_module(FPN)
NECKS.register_module(Decoder)