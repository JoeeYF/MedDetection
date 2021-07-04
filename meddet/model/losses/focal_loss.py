# 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from meddet.model.nnModules import NetModule
# from meddet.ops.torch.boxes import iouNd_pytorch
#
#
# # class FocalLoss(nn.Module):
# #     def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
# #         super(FocalLoss, self).__init__()
# #         self.alpha = alpha
# #         self.gamma = gamma
# #         self.logits = logits
# #         self.reduce = reduce
# #
# #     def forward(self, inputs, targets):
# #         if self.logits:
# #             BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
# #         else:
# #             BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
# #         pt = torch.exp(-BCE_loss)
# #         F_loss = self.alpha * torch.sub(1, pt) ** self.gamma * BCE_loss
# #
# #         if self.reduce:
# #             return torch.mean(F_loss)
# #         else:
# #             return F_loss
#
#
# class FocalLoss(NetModule):
#
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, classifications, regressions, anchor, annotations):
#         """
#         :param classifications: [2, 36828, 3]
#         :param regressions: [2, 36828, 4]
#         :param anchor: [36828, 4]
#         :param annotations: [2, n, 4+1]
#         :return:
#         """
#         self.try_to_info(classifications.shape,
#                          regressions.shape,
#                          anchor.shape,
#                          annotations.shape)
#         self.try_to_info('\n', annotations)
#
#         classifications = torch.sigmoid(classifications)
#
#         alpha = self.alpha
#         gamma = self.gamma
#
#         batch_size = classifications.shape[0]
#         classification_losses = []
#         regression_losses = []
#
#         anchor_widths = anchor[:, 2] - anchor[:, 0]
#         anchor_heights = anchor[:, 3] - anchor[:, 1]
#         anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
#         anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
#
#         for j in range(batch_size):
#
#             classification = classifications[j, :, :]
#             regression = regressions[j, :, :]
#
#             bbox_annotation = annotations[j, :, :]
#             bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
#             if bbox_annotation.shape[0] == 0:
#                 regression_losses.append(torch.tensor(0).float().cuda())
#                 classification_losses.append(torch.tensor(0).float().cuda())
#
#                 continue
#
#             classification = torch.clamp(classification, 1e-5, 1.0 - 1e-5)
#             IoU = iouNd_pytorch(anchor, bbox_annotation[:, :-1])  # num_anchors x num_annotations
#
#             IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1
#             IoU_max_gt_bboxes, IoU_argmax_gt_bboxes = torch.max(IoU, dim=0)  # num_annotations x 1
#             self.try_to_info("IoU_max_gt_bboxes", IoU_max_gt_bboxes)
#             # compute the loss for classification
#             targets = torch.ones(classification.shape) * -1
#             targets = targets.to(classification.device)
#
#             negative_indices = torch.le(IoU_max, 0.4)
#             num_negative_anchors = negative_indices.sum()
#
#             positive_indices = torch.ge(IoU_max, 0.5)
#             num_positive_anchors = positive_indices.sum()
#
#             # self.try_to_log("num_positive_anchors", num_positive_anchors, num_negative_anchors, positive_indices.shape)
#             # self.try_to_log(torch.where(IoU_max >= 0.5)[0])
#
#             assigned_annotations = bbox_annotation[IoU_argmax, :]
#             # self.try_to_log(assigned_annotations[:10, :], assigned_annotations.shape)
#
#             # multi class
#             targets[negative_indices, :] = 0
#             targets[positive_indices, :] = 0
#             if classification.shape[-1] == 1:
#                 targets[positive_indices, 0] = 1
#             else:
#                 targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
#             self.try_to_info("targets", targets.shape, targets[targets != -1].sum(dim=0), targets[targets == -1].sum(dim=0))
#             alpha_factor = torch.ones(targets.shape) * alpha
#             alpha_factor = alpha_factor.to(classification.device)
#
#             alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
#             focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
#             # focal_weight = torch.clamp(focal_weight, 1e-4, 1.0)
#             focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
#
#             bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
#             self.try_to_info("targets", targets.shape, torch.sum(targets[:, 0, ...]), torch.sum(bce))
#
#             cls_loss = focal_weight * bce
#
#             cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss,
#                                    torch.zeros(cls_loss.shape).to(classification.device))
#             classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
#             self.try_to_info("cls_loss.shape", cls_loss.shape, cls_loss.sum(dim=0), cls_loss.mean(dim=0), classification_losses[-1])
#
#             # compute the loss for regression
#
#             if positive_indices.sum() > 0:
#                 assigned_annotations = assigned_annotations[positive_indices, :]
#
#                 anchor_widths_pi = anchor_widths[positive_indices]
#                 anchor_heights_pi = anchor_heights[positive_indices]
#                 anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
#                 anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
#
#                 gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
#                 gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
#                 gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
#                 gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
#
#                 # clip widths to 1
#                 gt_widths = torch.clamp(gt_widths, min=1)
#                 gt_heights = torch.clamp(gt_heights, min=1)
#
#                 targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
#                 targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
#                 targets_dw = torch.log(gt_widths / anchor_widths_pi)
#                 targets_dh = torch.log(gt_heights / anchor_heights_pi)
#
#                 targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
#                 targets = targets.t()
#
#                 targets = targets / torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(classification.device)
#
#                 regression_diff = torch.abs(targets - regression[positive_indices, :])
#                 self.try_to_info("regression_diff", regression_diff.shape)
#                 regression_loss = torch.where(
#                     torch.le(regression_diff, 1.0 / 9.0),
#                     0.5 * 9.0 * torch.pow(regression_diff, 2),
#                     regression_diff - 0.5 / 9.0
#                 )
#                 self.try_to_info("regression_loss", regression_loss.sum())
#                 regression_losses.append(regression_loss.mean())
#             else:
#                 regression_losses.append(torch.tensor(0).float().to(classification.device))
#             self.try_to_info(f"cls and reg loss of sample {j}", classification_losses[-1], regression_losses[-1])
#         return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
#                torch.stack(regression_losses).mean(dim=0, keepdim=True)
#
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.autograd import Variable
# #
# #
# # class FocalLoss(nn.Module):
# #     def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
# #         super(FocalLoss, self).__init__()
# #         self.gamma = gamma
# #         self.alpha = alpha
# #         if isinstance(alpha, (float, int)):
# #             self.alpha = torch.tensor([alpha, 1 - alpha])
# #         if isinstance(alpha, list):
# #             self.alpha = torch.tensor(alpha)
# #         self.size_average = size_average
# #
# #     def forward(self, cls, target):
# #         # if cls.dim() > 2:
# #         #     cls = cls.view(cls.size(0), cls.size(1), -1)  # N,C,H,W => N,C,H*W
# #         #     cls = cls.transpose(1, 2)  # N,C,H*W => N,H*W,C
# #         #     cls = cls.contiguous().view(-1, cls.size(2))  # N,H*W,C => N*H*W,C
# #         # target = target.view(-1, 1)
# #
# #         logpt = F.softmax(cls)
# #         print(logpt.shape, target.shape)
# #         logpt = logpt.gather(1, target)
# #         logpt = logpt.view(-1)
# #         pt = Variable(logpt.data.exp())
# #
# #         if self.alpha is not None:
# #             if self.alpha.type() != cls.data.type():
# #                 self.alpha = self.alpha.type_as(cls.data)
# #             at = self.alpha.gather(0, target.data.view(-1))
# #             logpt = logpt * Variable(at)
# #
# #         loss = -1 * (1 - pt) ** self.gamma * logpt
# #         if self.size_average:
# #             return loss.mean()
# #         else:
# #             return loss.sum()
