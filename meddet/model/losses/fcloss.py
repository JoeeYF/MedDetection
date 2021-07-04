import torch
from torch import nn
import torch.nn.functional as F

# @Author  : LG
from torch import nn
import torch
from torch.nn import functional as F
from meddet.model.losses.utils import weight_reduce_loss


# class focal_loss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
#         """
#         focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
#         步骤详细的实现了 focal_loss损失函数.
#         :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
#         :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
#         :param num_classes:     类别数量
#         :param size_average:    损失计算方式,默认取均值
#         """
#
#         super(focal_loss, self).__init__()
#         self.size_average = size_average
#         if isinstance(alpha, list):
#             assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
#             print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
#             self.alpha = torch.tensor(alpha)
#         else:
#             assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
#             print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
#             self.alpha = torch.zeros(num_classes)
#             self.alpha[0] += alpha
#             self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
#         self.gamma = gamma
#
#     def forward(self, preds, labels):
#         """
#         focal_loss损失计算
#         :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
#         :param labels:  实际类别. size:[B,N] or [B]
#         :return:
#         """
#         # assert preds.dim()==2 and labels.dim()==1
#         preds = preds.view(-1, preds.size(-1))
#         self.alpha = self.alpha.to(preds.device)
#         preds_softmax = F.softmax(preds, dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
#         preds_logsoft = torch.log(preds_softmax)
#         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
#         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
#         self.alpha = self.alpha.gather(0, labels.view(-1))
#         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
#         loss = torch.mul(self.alpha, loss.t())
#         if self.size_average:
#             loss = loss.mean()
#         else:
#             loss = loss.sum()
#         return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0):
        """
        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override='mean'):
        """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

        Args:
            pred (torch.Tensor): (N, C), C is the number of classes, without background class
                The prediction
            target (torch.Tensor): (N, )  from 1 ~ C
                The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            reduction_override (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        reduction = (
            reduction_override if reduction_override else self.reduction)
        pred_sigmoid = pred.sigmoid()
        target = torch.nn.functional.one_hot(target.long(), pred.shape[-1] + 1)[..., 1:]
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        loss = self.loss_weight * weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss


if __name__ == "__main__":
    pred = torch.tensor([[0.4, 0.1, 0.5],
                         [0.2, 0.6, 0.2]], requires_grad=True)
    label = torch.tensor([3, 3])
    loss = SigmoidFocalLoss()(pred, label)
    print(loss)
    print(loss.mean())
    loss.mean().backward()

    # loss = focal_loss()(pred, label)
    # print(loss)
    # loss.backward()
    #
    # from torch.nn import NLLLoss