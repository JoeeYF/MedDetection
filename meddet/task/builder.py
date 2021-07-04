

from torch import nn
import inspect

from meddet.utils import build_from_cfg
from meddet.model import BACKBONES, HEADS, LOSSES, NECKS, METRICS
from meddet.task.registry import DETECTORS


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_metric(cfg):
    return build(cfg, METRICS)


def build_detector(cfg):
    return build(cfg, DETECTORS)