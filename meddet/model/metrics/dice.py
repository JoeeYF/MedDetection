

import numpy as np
import torch
from torch import nn
from torch import nn

from meddet.model.nnModules import ComponentModule


class Dice(ComponentModule):
    def __init__(self,
                 do_bg=True,
                 aggregate=None):
        super().__init__()
        self.do_bg = do_bg
        self.aggregate = aggregate
        assert aggregate in [None, 'mean', 'sum']

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(do_bg={self.do_bg}, aggregate={self.aggregate})'
        return repr_str
    
    def __call__(self,
                 net_output: torch.Tensor,
                 target: torch.Tensor,
                 num_classes: int = None):
        """
        :param net_output : b,c,d,h,w
        :param target     : b,c,d,h,w
        :param num_classes: if none, get from net output shape[1]
        :return           : dice coefficient
        """
        assert net_output.ndim == target.ndim, f'Dimension not matched! {net_output.shape}, {target.shape}'

        if num_classes is None:
            num_classes = net_output.shape[1]
            prediction = torch.argmax(net_output, dim=1, keepdim=True)
        else:
            prediction = net_output
        dice = torch.zeros(num_classes).to(target.device)
        for i in range(num_classes):
            prediction_i = (prediction == i).cpu().float()
            target_i = (target == i).cpu().float()

            smooth = 1e-5
            intersection_i = torch.sum(prediction_i * target_i)
            union_i = prediction_i.sum() + target_i.sum()
            dc_i = torch.true_divide(2. * intersection_i + smooth, union_i + smooth)
            dice[i] = dc_i
        if self.aggregate is None:
            metric = dict(zip([f"dice_class_{i}" for i in range(num_classes)], dice))
        elif self.aggregate == 'mean':
            metric = {'mean_dice': torch.mean(dice)}
        else:
            metric = {'sum_dice': torch.sum(dice)}
        return metric


# import medpy.metric.binary

# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size(0)
#     m1 = pred.view(num, -1)  # Flatten
#     m2 = target.view(num, -1)  # Flatten
#     intersection = (m1 * m2).sum()
#
#     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# a = torch.rand((2, 3, 6, 4))
# b = torch.ones((2, 3, 6, 4))
# #
# # print(dice_coeff(b, a))
#
#
# d = DiceCoefficient(batch_dice=True)
# print(d(a, b))