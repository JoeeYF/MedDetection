

from typing import List, Union
from meddet.task.builder import build_loss
from meddet.model.nnModules import ComponentModule


class CombineLoss(ComponentModule):
    def __init__(self,
                 loss_dicts: Union[List[dict], List[object]],
                 loss_weights: list):
        super().__init__()
        assert len(loss_dicts) == len(loss_weights)
        self.weights = loss_weights
        self.losses = list(map(build_loss, loss_dicts))

    def __repr__(self):
        repr_str = self.__class__.__name__
        losses_str = []
        for i, l in enumerate(self.losses):
            losses_str.append(f'{self.weights[i]}x_' + l.abbr)
        repr_str += f'({",".join(losses_str)})'
        return repr_str

    def forward(self, net_output, target, weight=None):
        losses = {}
        for i, l in enumerate(self.losses):
            # print(l.abbr)
            losses[f'{self.weights[i]}x_' + l.abbr] = self.weights[i] * l(net_output, target, weight=weight)
        return losses