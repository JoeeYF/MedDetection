

from ..registry import DETECTORS

from .retina_net import RetinaNet
from .faster_rcnn import FasterRCNN
from .deeplung import DeepLung
from .deeplung_sim import DeepLungSim

from .det_monitor import DetMonitor
from .centernet import CenterNet
DETECTORS.register_module(RetinaNet)
DETECTORS.register_module(FasterRCNN)
DETECTORS.register_module(DeepLung)
DETECTORS.register_module(DeepLungSim)
DETECTORS.register_module(CenterNet)
