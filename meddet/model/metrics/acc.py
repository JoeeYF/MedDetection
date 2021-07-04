

import torch
from meddet.model.nnModules import ComponentModule


class Acc(ComponentModule):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self, pred, gt):
        """

        :param gt: (b)
        :param pred: (b, classes)
        :return:
        """
        acc = (gt == pred.argmax(-1)).float().detach().numpy()
        acc = float(100 * acc.sum() / len(acc))
        print(acc, pred.argmax(-1), gt)
        return {"Acc": acc}