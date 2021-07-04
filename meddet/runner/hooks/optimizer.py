# Copyright (c) Open-MMLab. All rights reserved.
from torch.nn.utils import clip_grad

from .hook import HOOKS, Hook


@HOOKS.register_module
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        if runner.float16:
            runner.optimizer.zero_grad()
            runner.scaler.scale(runner.outputs['loss']).backward()
            if self.grad_clip is not None:
                runner.scaler.unscale_(runner.optimizer)
                self.clip_grads(runner.model.parameters())
            runner.scaler.step(runner.optimizer)
            runner.scaler.update()
        else:
            runner.optimizer.zero_grad()
            runner.outputs['loss'].backward()
            if self.grad_clip is not None:
                self.clip_grads(runner.model.parameters())
            runner.optimizer.step()


