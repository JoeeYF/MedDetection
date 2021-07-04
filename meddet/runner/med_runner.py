

import torch
from torch.cuda.amp import autocast, GradScaler
import os.path as osp
import os
import time
import traceback
import numpy as np
import random

from . import hooks
from ..utils import LogBuffer, obj_from_dict
from .hooks import Hook, get_priority, LrUpdaterHook, lr_updater, CheckpointHook, IterTimerHook, OptimizerHook, ValUpdaterHook, val_updater
from .checkpoint import load_checkpoint, save_checkpoint


class MedRunner(object):
    """A training helper for PyTorch.
    """

    def __init__(self,
                 model,
                 monitor,
                 optimizer,
                 work_dir,
                 logger,
                 timestamp,
                 float16=True):

        assert isinstance(model, torch.nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(work_dir, str)

        self.model = model
        self.monitor = monitor
        self.optimizer = optimizer
        self.work_dir = osp.abspath(work_dir)
        self.timestamp = timestamp
        self.logger = logger
        self.float16 =float16
        os.makedirs(self.work_dir, exist_ok=True)
        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = 0, 1
        # self._rank, self._world_size = get_dist_info()

        self.scaler = GradScaler()
        self.mode = None
        self.data_loader = None
        self.log_buffer = LogBuffer()
        self.outputs = None  # a dict of losses or metrics
        self._hooks = []
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    @property
    def model_name(self):
        """str: Name of the model, usually the module class name."""
        return self._model_name

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    @property
    def inner_iter(self):
        """int: Iteration in an epoch."""
        return self._inner_iter

    @property
    def max_epochs(self):
        """int: Maximum training epochs."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Maximum training iterations."""
        return self._max_iters

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def current_latitude(self):
        return self.data_loader.dataset.getLatitude()

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=False):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        else:
            meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            os.symlink(filename, osp.join(out_dir, 'latest.pth'))

    def register_hook(self, hook, priority='NORMAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority
        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)

    def build_hook(self, args, hook_type=None):
        if isinstance(args, Hook):
            return args
        elif isinstance(args, dict):
            assert issubclass(hook_type, Hook)
            return hook_type(**args)
        else:
            raise TypeError('"args" must be either a Hook object'
                            ' or dict, not {}'.format(type(args)))

    def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def resume(self,
               checkpoint,
               resume_optimizer=True,
               map_location='default'):
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(
                    checkpoint, map_location='cpu')
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info('resumed epoch %d, iter %d', self.epoch, self.iter)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')

        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            data_batch['iter'] = i
            data_batch['epoch'] = self._epoch
            self.call_hook('before_train_iter')
            if self.float16:
                with autocast():
                    outputs, prediction, net_output = self.model(data_batch, True, self.optimizer)
            else:
                outputs, prediction, net_output = self.model(data_batch, True, self.optimizer)

            if self._inner_iter == 0:
                self.monitor.setTrainMode()
                self.monitor.viewData(data_batch, force_save=True)
                self.monitor.viewFeatures(data_batch, net_output, force_save=True)
                # self.monitor.viewResults(data_batch, net_output, force_save=True)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
        pass

    def valid(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'valid'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        self.data_loader.dataset.setLatitude(0)
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            data_batch['iter'] = i
            data_batch['epoch'] = self._epoch
            self.call_hook('before_val_iter')
            try:
                if self.float16:
                    with autocast():
                        with torch.no_grad():
                            outputs, prediction, net_output = self.model(data_batch, False, self.optimizer)
                else:
                    with torch.no_grad():
                        outputs, prediction, net_output = self.model(data_batch, False, self.optimizer)
            except Exception as e:
                traceback.print_exc()
                continue
            if self._inner_iter == 0:
                self.monitor.setValidMode()
                self.monitor.viewData(data_batch, force_save=True)
                self.monitor.viewFeatures(data_batch, net_output, force_save=True)
                self.monitor.viewResults(data_batch, prediction, force_save=True)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')
        pass

    def run(self, data_loaders, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loaders, list)
        # assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)

        self._max_epochs = max_epochs
        # work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        # # self.logger.info('Start running, host: %s, work_dir: %s',
        # #                  get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, max_epochs)
        # LrUpdaterHook, LoggerHook
        self.call_hook('before_run')

        while self.epoch < max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                epochs = epochs - self.epoch % epochs
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run an epoch'.format(mode))
                    epoch_runner = getattr(self, mode)
                elif callable(mode):  # custom train()
                    epoch_runner = mode
                else:
                    raise TypeError('mode in workflow must be a str or '
                                    'callable function, not {}'.format(type(mode)))
                for _ in range(epochs):
                    _seed_ = np.random.get_state()[1][0]
                    np.random.seed(_seed_ + self.epoch * 99)
                    random.seed(_seed_ + self.epoch * 99)
                    # print('init epoch seed', _seed_, '->', np.random.get_state()[1][0])
                    if mode == 'train' and self.epoch >= max_epochs:
                        continue
                    epoch_runner(data_loaders[i], **kwargs)
                    if mode != 'train' and self.epoch >= max_epochs:
                        return

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def register_lr_hooks(self, lr_config):
        if isinstance(lr_config, LrUpdaterHook):
            self.register_hook(lr_config)
        elif isinstance(lr_config, dict):
            assert 'policy' in lr_config
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_aug_hooks(self, augmentation_config):
        if isinstance(augmentation_config, ValUpdaterHook):
            self.register_hook(augmentation_config)
        elif isinstance(augmentation_config, dict):
            assert 'policy' in augmentation_config
            hook_name = augmentation_config['policy'].title() + 'ValUpdaterHook'
            if not hasattr(val_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(val_updater, hook_name)
            self.register_hook(hook_cls(**augmentation_config))
        else:
            raise TypeError('"val_updater" must be either a ValUpdaterHook object'
                            ' or dict, not {}'.format(type(augmentation_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config,
                                checkpoint_config,
                                log_config,
                                augmentation_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        self.register_logger_hooks(log_config)
        if augmentation_config is not None:
            self.register_aug_hooks(augmentation_config)
