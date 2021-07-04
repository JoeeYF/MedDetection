

from meddet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
METRICS = Registry('metric')