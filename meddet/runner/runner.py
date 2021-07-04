

import logging
import torch
import os.path as osp
import os
import time
import math

from . import hooks
from ..utils import get_time_str, LogBuffer, obj_from_dict, get_root_logger
from .hooks import Hook, get_priority, LrUpdaterHook, lr_updater, CheckpointHook, IterTimerHook, OptimizerHook
from .checkpoint import load_checkpoint, save_checkpoint


class Runner(object):
    """A training helper for PyTorch.
    """

    def __init__(self,
                 model,
                 monitor,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 timestamp=None):
        self.model = model
        if optimizer is not None:
            self.optimizer = self.init_optimizer(optimizer)
        else:
            self.optimizer = None
        self.data_loader = None
        # create work_dir
        if isinstance(work_dir, str):
            self.work_dir = osp.abspath(work_dir)
            os.makedirs(self.work_dir, exist_ok=True)
        elif work_dir is None:
            self.work_dir = None
        else:
            raise TypeError('"work_dir" must be a str or None')

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._rank, self._world_size = 0, 8
        # self._rank, self._world_size = get_dist_info()
        self.monitor = monitor
        if timestamp is None:
            self.timestamp = get_time_str()
        else:
            self.timestamp = timestamp
        if logger is None:
            self.logger = get_root_logger()
        else:
            self.logger = logger
        self.log_buffer = LogBuffer()
        self.outputs = None  # a dict of losses or metrics

        self.mode = None
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

    def init_optimizer(self, optimizer):
        """Init the optimizer.

        Args:
            optimizer (dict or :obj:`~torch.optim.Optimizer`): Either an
                optimizer object or a dict used for constructing the optimizer.

        Returns:
            :obj:`~torch.optim.Optimizer`: An optimizer object.

        # Examples:
        #     >>> optimizer = dict(type='SGD', lr=0.01, momentum=0.9)
        #     >>> type(runner.init_optimizer(optimizer))
        #     <class 'torch.optim.sgd.SGD'>
        """
        if isinstance(optimizer, dict):
            optimizer = obj_from_dict(optimizer, torch.optim,
                                      dict(params=self.model.parameters()))
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer

    def current_lr(self):
        """Get current learning rates.

        Returns:
            list: Current learning rate of all param groups.
        """
        if self.optimizer is None:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return [group['lr'] for group in self.optimizer.param_groups]

    def load_checkpoint(self, filename, map_location='cpu', strict=False):
        self.logger.info('load checkpoint from %s', filename)
        return load_checkpoint(self.model, filename, map_location, strict, self.logger)

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
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
            pass
            # os.symlink(filename, osp.join(out_dir, 'latest.pth'))

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
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')

        # if hasattr(data_loader.dataset, 'setLatitude'):
        x = (1 + self._epoch) / self._max_epochs
        start = 0.2
        latitude = min(x**2 + start, 1.0)
        # latitude = (1 - start) * 0.5 * (math.tanh(4 * x - 2) + 1) + start
        # latitude = (1 - start) * ((2 * x - 1) / (1 + abs(2 * x - 1)) + 0.5) + start
        data_loader.dataset.setLatitude(latitude)
        # data_loader.dataset.setLatitude(0.0)
        for i, data_batch in enumerate(data_loader):
            # print('start', i)
            # print(len(data_batch['img']))
            self._inner_iter = i
            data_batch['iter'] = i
            data_batch['epoch'] = self._epoch
            self.call_hook('before_train_iter')
            # outputs = self.batch_processor(
            #     self.model, data_batch, train_mode=True, **kwargs)
            outputs, prediction, net_output = self.model(data_batch, True, self.optimizer)
            # print('runner')
            # print(outputs)
            # print(prediction)
            # print(net_output[0].shape)
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
            # print('end', i)

        self.call_hook('after_train_epoch')
        self._epoch += 1
        pass

    def valid(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'valid'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')

        # if hasattr(data_loader.dataset, 'setLatitude'):
        data_loader.dataset.setLatitude(0)
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            data_batch['iter'] = i
            data_batch['epoch'] = self._epoch
            self.call_hook('before_val_iter')
            with torch.no_grad():
                # outputs = self.batch_processor(
                #     self.model, data_batch, train_mode=False, **kwargs)
                outputs, prediction, net_output = self.model(data_batch, False, self.optimizer)
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
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        # self.logger.info('Start running, host: %s, work_dir: %s',
        #                  get_host_info(), work_dir)
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
            # from .hooks import lr_updater
            hook_name = lr_config['policy'].title() + 'LrUpdaterHook'
            if not hasattr(lr_updater, hook_name):
                raise ValueError('"{}" does not exist'.format(hook_name))
            hook_cls = getattr(lr_updater, hook_name)
            self.register_hook(hook_cls(**lr_config))
        else:
            raise TypeError('"lr_config" must be either a LrUpdaterHook object'
                            ' or dict, not {}'.format(type(lr_config)))

    def register_logger_hooks(self, log_config):
        log_interval = log_config['interval']
        for info in log_config['hooks']:
            logger_hook = obj_from_dict(
                info, hooks, default_args=dict(interval=log_interval))
            self.register_hook(logger_hook, priority='VERY_LOW')

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None):
        """Register default hooks for training.

        Default hooks include:

        - LrUpdaterHook
        - OptimizerStepperHook
        - CheckpointSaverHook
        - IterTimerHook
        - LoggerHook(s)
        """
        if optimizer_config is None:
            optimizer_config = {}
        if checkpoint_config is None:
            checkpoint_config = {}
        self.register_lr_hooks(lr_config)
        self.register_hook(self.build_hook(optimizer_config, OptimizerHook))
        self.register_hook(self.build_hook(checkpoint_config, CheckpointHook))
        self.register_hook(IterTimerHook())
        if log_config is not None:
            self.register_logger_hooks(log_config)
