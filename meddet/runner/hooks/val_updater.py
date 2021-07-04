# Copyright (c) Open-MMLab. All rights reserved.
from __future__ import division
import math

from .hook import HOOKS, Hook


class ValUpdaterHook(Hook):
    def __init__(self,
                 start,
                 **kwargs):
        self.start = start

    def get_latitude(self, runner):
        raise NotImplementedError

    def before_train_epoch(self, runner):
        runner.data_loader.dataset.setLatitude(self.get_latitude(runner))


@HOOKS.register_module
class FixedValUpdaterHook(ValUpdaterHook):

    def __init__(self, **kwargs):
        super(FixedValUpdaterHook, self).__init__(**kwargs)

    def get_latitude(self, runner):
        latitude = self.start
        return min(latitude, 1.0)


@HOOKS.register_module
class StepValUpdaterHook(ValUpdaterHook):

    def __init__(self, step, delta=0.1, **kwargs):
        assert isinstance(step, (list, int))
        if isinstance(step, list):
            for s in step:
                assert isinstance(s, int) and s > 0
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.delta = delta
        super(StepValUpdaterHook, self).__init__(**kwargs)

    def get_latitude(self, runner):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return self.start + self.delta * progress // self.step

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        latitude = self.start + self.delta * exp
        return min(latitude, 1.0)


@HOOKS.register_module
class PolyValUpdaterHook(ValUpdaterHook):

    def __init__(self, power=2., **kwargs):
        self.power = power
        super(PolyValUpdaterHook, self).__init__(**kwargs)

    def get_latitude(self, runner):
        progress = runner.epoch
        max_progress = runner.max_epochs
        latitude = self.start + (progress / max_progress) ** self.power
        return min(latitude, 1.0)


@HOOKS.register_module
class CosineValUpdaterHook(ValUpdaterHook):

    def __init__(self, target_lr=1.0, warm=0.9, **kwargs):
        self.target_lr = target_lr
        self.warm = warm
        super(CosineValUpdaterHook, self).__init__(**kwargs)

    def get_latitude(self, runner):
        progress = runner.epoch
        max_progress = runner.max_epochs
        if progress >= self.warm * max_progress:
            latitude = 1
        else:
            latitude = self.target_lr + 0.5 * (self.start - self.target_lr) * \
                       (1 + math.cos(math.pi * (progress / (self.warm * max_progress))))
        return min(latitude, 1)
