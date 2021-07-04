

import torch
from collections import OrderedDict
from abc import abstractmethod

from meddet.model.nnModules import ModelModule


class BaseTask(ModelModule):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_losses(losses_dict):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses_dict.items():
            if isinstance(loss_value, torch.Tensor):
                try:
                    log_vars[loss_name] = loss_value.mean()
                except:
                    log_vars[loss_name] = loss_value.sum() / loss_value.numel()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError('{}:{} is not a tensor or list of tensors'.format(loss_name, type(loss_value)))

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    @abstractmethod
    def forward_train(self, data_batch, *args, **kwargs):
        """
        Args:
            data_batch: data batch in dict format, has keys img, gt_seg ...
            *args:
            **kwargs:

        Returns:
            losses: dict
            prediction: None
            net_output: tensor, monitored features
        """
        pass

    @abstractmethod
    def forward_valid(self, data_batch, *args, **kwargs):
        """
        Args:
            data_batch: data batch in dict format, has keys img, gt_seg ...
            *args:
            **kwargs:

        Returns:
            losses + metric: dict
            prediction: prediction with probability which is used in metric and inference
            net_output: tensor, monitored features
        """
        pass

    @abstractmethod
    def forward_infer(self, data_batch, *args, **kwargs):
        """
        Args:
            data_batch: data batch in dict format, has keys img
            *args:
            **kwargs:

        Returns:
            prediction: prediction with probability which is used in metric and inference
            net_output: tensor, monitored features
        """
        pass

    def forward(self, data_batch, return_loss=True, *args, **kwargs):
        assert all([i in data_batch.keys() for i in ['img_meta', 'img']]), print(data_batch.keys())
        assert 'gt' in str(data_batch.keys()), 'Only support train and valid! Using forward_infer for inference!'
        assert torch.max(data_batch['img']) <= 10.0 and torch.min(data_batch['img']) >= -10.0, \
            'max={} min={}'.format(torch.max(data_batch['img']).item(), torch.min(data_batch['img']).item())

        if not return_loss:
            out = self.forward_valid(data_batch, *args, **kwargs)
        else:
            out = self.forward_train(data_batch, *args, **kwargs)

        assert len(out) == 3, 'forward function must return three outputs'
        assert isinstance(out[0], dict)

        losses_dict, prediction, net_output = out
        losses_dict['num_samples'] = torch.tensor(data_batch['img'].shape[0]).to(data_batch['img'].device)
        return losses_dict, prediction, net_output