 

# -*- coding:utf-8 -*-

from meddet.model.heads.utils.util_anchors import AnchorGenerator
from meddet.model.heads.utils.util_assigners import IoUAssigner, MaxIoUAssigner, DistAssigner
from meddet.model.heads.utils.util_bboxes import DeltaBBoxCoder
from meddet.model.heads.utils.util_samplers import RandomSampler, OHEMSampler, HNEMSampler
from meddet.model.heads.utils.util_extractors import SingleRoIExtractor, GenericRoIExtractor

ANCHORS = dict(
    AnchorGenerator=AnchorGenerator
)
CODERS = dict(
    DeltaBBoxCoder=DeltaBBoxCoder,
)
ASSIGNERS = dict(
    IoUAssigner=IoUAssigner,
    MaxIoUAssigner=MaxIoUAssigner,
    DistAssigner=DistAssigner,
)
SAMPLERS = dict(
    RandomSampler=RandomSampler,
    OHEMSampler=OHEMSampler,
    HNEMSampler=HNEMSampler
)

EXTRACTORS = dict(
    SingleRoIExtractor=SingleRoIExtractor,
    GenericRoIExtractor=GenericRoIExtractor
)


def build_from_cfg(cfg: dict, registry: dict):
    """
    Add assigner here
    Args:
        cfg: config dict
        registry:

    Returns: obj
    """
    assert isinstance(cfg, dict) and 'type' in cfg.keys()

    obj_type = cfg.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the ASSIGNERS'.format(obj_type))
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))

    try:
        return obj_cls(**cfg)
    except Exception as e:
        raise Exception(f"error while build class: {obj_type}\n" + str(e))


def build_anchor_generator(cfg):
    return build_from_cfg(cfg, ANCHORS)


def build_coder(cfg):
    return build_from_cfg(cfg, CODERS)


def build_assigner(cfg):
    return build_from_cfg(cfg, ASSIGNERS)


def build_sampler(cfg):
    return build_from_cfg(cfg, SAMPLERS)


def build_extractor(cfg):
    return build_from_cfg(cfg, EXTRACTORS)
