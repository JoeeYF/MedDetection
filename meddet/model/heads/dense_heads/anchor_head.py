 

# -*- coding:utf-8 -*-
from typing import List, Union
import warnings
import torch
from torch import nn
import numpy as np

from meddet.task.builder import build_loss, build_metric
from meddet.model.heads.dense_heads.base_dense_head import BaseDenseHead
from meddet.model.heads.utils import build_anchor_generator, build_coder, build_assigner, build_sampler
from meddet.utils import multi_apply

from medvision.ops.torch import nmsNd_pytorch, softnmsNd_pytorch
try:
    from medvision.ops import nms_nd as nmsNd_cuda
except ImportError:
    warnings.warn('Using medvision.ops.torch.nmsNd_pytorch instead.')
    from medvision.ops.torch import nmsNd_pytorch as nmsNd_cuda


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


class AnchorHead(BaseDenseHead):
    def __init__(self,
                 dim: int,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 anchor_generator: dict = None,
                 assigner: dict = None,
                 sampler: dict = None,
                 bbox_coder: dict = None,
                 proposal: dict = None,
                 nms: dict = None,
                 losses: dict = None,
                 metrics: Union[List[dict], List[object]] = None,
                 level_first: bool = False,
                 **kwargs):
        super(AnchorHead, self).__init__()
        if losses.get('cls', None) is None:
            losses['cls'] = dict(type='SigmoidFocalLoss', gamma=2.0, alpha=0.25)
        if losses.get('reg', None) is None:
            losses['reg'] = dict(type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=1.0)

        self.dim = dim
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.level_first = level_first
        self.use_sigmoid_cls = losses['cls'].get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        anchor_generator.update({'dim': self.dim})
        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.anchor_assigner = build_assigner(assigner)
        self.anchor_sampler = build_sampler(sampler)
        self.num_anchors = self.anchor_generator.num_base_anchors

        bbox_coder.update({'dim': self.dim})
        self.bbox_coder = build_coder(bbox_coder)

        self.nms = nms
        self.proposal_cfg = proposal

        self.base_criterion = nn.CrossEntropyLoss()
        self.criterion_cls = build_loss(losses['cls'])
        self.criterion_reg = build_loss(losses['reg'])
        self.metrics = nn.ModuleList([build_metric(metric) for metric in metrics])
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_anchors * 2 * self.dim, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        pass

    def forward_single_level(self, feature):
        """

        Args:
            feature: single_level_feature, (b, c, **shape)

        Returns:

        """
        cls_out = self.conv_cls(feature)  # (b, nb * class, (z,) y, x)
        reg_out = self.conv_reg(feature)  # (b, nb * coord, (z,) y, x)
        return cls_out, reg_out

    def forward(self, features):
        """Forward features from the upstream network.

        Args:
            features (tuple[Tensor]): Features from the upstream network, each is
                a 4D/5D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D/5D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D/5D-tensor, the channels number \
                    is num_anchors * coord.
        """
        return multi_apply(self.forward_single_level, features)

    def get_anchors(self, feature_shapes, device):
        """

        Args:
            feature_shapes (list[tuple]): Multi-level feature map sizes.
            device (torch.device): Device for returned tensors

        Returns:
            tuple:
                concat_level_anchors: concat multi-level anchors
                num_level_anchors: num of each level anchors
        """
        multi_level_anchors, num_level_anchors = self.anchor_generator(feature_shapes, device=device)
        concat_level_anchors = torch.cat(multi_level_anchors, dim=0)
        assert self.anchor_generator.num_levels == len(feature_shapes)
        return concat_level_anchors, num_level_anchors

    def get_flatten_out(self, multi_level_cls_out, multi_level_reg_out):
        batch_cls_out_flat, batch_reg_out_flat = multi_apply(
            self._flatten_single_level,
            multi_level_cls_out,
            multi_level_reg_out
        )
        batch_cls_out_flat = [i for i in torch.cat(batch_cls_out_flat, dim=1)]
        batch_reg_out_flat = [i for i in torch.cat(batch_reg_out_flat, dim=1)]
        return batch_cls_out_flat, batch_reg_out_flat

    def _flatten_single_level(self, cls_out, reg_out):
        batch_size = cls_out.shape[0]
        axes = [0] + list(range(2, self.dim + 2)) + [1]
        cls_out_flat = cls_out.permute(*axes).contiguous().view(batch_size, -1, self.num_classes)  # (b, Num, cls)
        reg_out_flat = reg_out.permute(*axes).contiguous().view(batch_size, -1, 2 * self.dim)  # (b, Num, 2dim)
        return cls_out_flat, reg_out_flat

    def _target_single_image(self, gt_labels, gt_bboxes, concat_anchors):
        gt_bboxes = gt_bboxes[gt_labels > 0]
        gt_labels = gt_labels[gt_labels > 0].long()

        assigned_labels, assigned_bboxes = self.anchor_assigner.assign(concat_anchors, gt_bboxes, gt_labels)

        label_targets = assigned_labels
        bbox_targets = assigned_bboxes

        pos_indices = assigned_labels > 0  # torch.nonzero(, as_tuple=False).squeeze(-1)
        neg_indices = assigned_labels == 0  # torch.nonzero(, as_tuple=False).squeeze(-1)

        return pos_indices, neg_indices, label_targets, bbox_targets

    def _loss_single_image(self,
                           anchors,
                           cls_out_flat,
                           reg_out_flat,
                           label_targets,
                           bbox_targets,
                           pos_indices,
                           neg_indices,
                           total_batch):

        self.try_to_info("cls_out_flat", cls_out_flat.shape,
                         "reg_out_flat", reg_out_flat.shape)
        self.try_to_info('pos', len(pos_indices), 'neg', len(neg_indices))

        not_ignored = pos_indices + neg_indices
        with torch.no_grad():
            losses = self.criterion_cls(
                cls_out_flat[not_ignored],
                label_targets[not_ignored],
                reduction_override='none').squeeze(-1)

        # weight = - cls_out_flat.view(-1) * pos_indices + cls_out_flat.view(-1) * neg_indices
        # weight = weight[not_ignored]

        sampled_pos_indices, sampled_neg_indices, _, _, _ = self.anchor_sampler.sample(None,
                                                                                       label_targets[not_ignored],
                                                                                       bbox_targets[not_ignored],
                                                                                       gt_labels=None,
                                                                                       gt_bboxes=None,
                                                                                       weight=losses)

        num_pos, num_neg = torch.ones(1).to(cls_out_flat.device) * len(sampled_pos_indices), \
                           torch.ones(1).to(cls_out_flat.device) * len(sampled_neg_indices)

        bbox_targets[pos_indices] = self.bbox_coder.encode(anchors[pos_indices], bbox_targets[pos_indices])

        sampled_neg_cls = cls_out_flat[not_ignored][sampled_neg_indices]
        sampled_neg_label = label_targets[not_ignored][sampled_neg_indices]
        neg_cls_loss = self.criterion_cls(
            sampled_neg_cls,
            sampled_neg_label) / total_batch
        tnr = 100.0 * (sampled_neg_cls < 0).sum() / len(sampled_neg_cls) / total_batch

        if len(sampled_pos_indices):
            sampled_pos_cls = cls_out_flat[not_ignored][sampled_pos_indices]
            sampled_pos_label = label_targets[not_ignored][sampled_pos_indices]
            pos_cls_loss = self.criterion_cls(
                sampled_pos_cls,
                sampled_pos_label
            ) / total_batch

            if getattr(self.criterion_reg, 'decode_bbox', False):
                pos_reg_loss = self.criterion_reg(
                    self.bbox_coder.decode(anchors[not_ignored][sampled_pos_indices], reg_out_flat[not_ignored][sampled_pos_indices]),
                    self.bbox_coder.decode(anchors[not_ignored][sampled_pos_indices], bbox_targets[not_ignored][sampled_pos_indices]),
                    reduction_override='none'
                )
            else:
                pos_reg_loss = self.criterion_reg(
                    reg_out_flat[not_ignored][sampled_pos_indices],
                    bbox_targets[not_ignored][sampled_pos_indices],
                    reduction_override='none'
                )
            # print(pos_reg_loss)
            pos_reg_loss = torch.mean(pos_reg_loss, dim=0).sum() / total_batch

            tpr = 100.0 * (sampled_pos_cls > 0).sum() / len(sampled_pos_cls) / total_batch
        else:
            pos_cls_loss = torch.tensor(0.0).to(cls_out_flat.device)
            pos_reg_loss = torch.tensor(0.0).to(cls_out_flat.device)
            tpr = 100.0 * torch.ones(1).to(cls_out_flat.device) / total_batch

        loss_cls = 0.5 * neg_cls_loss + 0.5 * pos_cls_loss
        loss_bbox = pos_reg_loss
        return loss_cls, loss_bbox, tnr, tpr, num_pos, num_neg

    def _loss_single_level(self,
                           anchors,
                           cls_out_flat,
                           reg_out_flat,
                           label_targets,
                           bbox_targets,
                           pos_indices,
                           neg_indices):

        anchors = anchors.contiguous().view(-1, 2 * self.dim)
        reg_out_flat = reg_out_flat.contiguous().view(-1, 2 * self.dim)
        bbox_targets = bbox_targets.contiguous().view(-1, 2 * self.dim)
        cls_out_flat = cls_out_flat.contiguous().view(-1, self.cls_out_channels)
        label_targets = label_targets.contiguous().view(-1)
        pos_indices = pos_indices.contiguous().view(-1)
        neg_indices = neg_indices.contiguous().view(-1)

        not_ignored = pos_indices + neg_indices
        with torch.no_grad():
            losses = self.criterion_cls(
                cls_out_flat[not_ignored],
                label_targets[not_ignored],
                reduction_override='none').squeeze(-1)

        # weight = - cls_out_flat * pos_indices + cls_out_flat * neg_indices
        # weight = weight[not_ignored]

        sampled_pos_indices, sampled_neg_indices, _, _, _ = self.anchor_sampler.sample(None,
                                                                                       label_targets[not_ignored],
                                                                                       bbox_targets[not_ignored],
                                                                                       gt_labels=None,
                                                                                       gt_bboxes=None,
                                                                                       weight=losses)

        bbox_targets[pos_indices] = self.bbox_coder.encode(anchors[pos_indices], bbox_targets[pos_indices])

        num_pos, num_neg = torch.ones(1).to(cls_out_flat.device) * len(sampled_pos_indices), \
                           torch.ones(1).to(cls_out_flat.device) * len(sampled_neg_indices)

        if len(sampled_pos_indices):
            sampled_pos_cls = cls_out_flat[not_ignored][sampled_pos_indices]
            sampled_pos_label = label_targets[not_ignored][sampled_pos_indices]

            pos_cls_loss = self.criterion_cls(
                sampled_pos_cls,
                sampled_pos_label
            )

            pos_reg_loss = self.criterion_reg(
                reg_out_flat[not_ignored][sampled_pos_indices],
                bbox_targets[not_ignored][sampled_pos_indices],
                reduction_override='none'
            )
            pos_reg_loss = torch.mean(pos_reg_loss, dim=0).sum()

            tpr = 100.0 * (sampled_pos_cls > 0).sum() / len(sampled_pos_cls)
        else:
            pos_cls_loss = torch.tensor(0.0).to(cls_out_flat.device)
            pos_reg_loss = torch.tensor(0.0).to(cls_out_flat.device)
            tpr = 100.0 * torch.ones(1).to(cls_out_flat.device)

        if len(sampled_neg_indices):
            sampled_neg_cls = cls_out_flat[not_ignored][sampled_neg_indices]
            sampled_neg_label = label_targets[not_ignored][sampled_neg_indices]
            neg_cls_loss = self.criterion_cls(
                sampled_neg_cls,
                sampled_neg_label)
            tnr = 100.0 * (sampled_neg_cls < 0).sum() / len(sampled_neg_cls)
        else:
            neg_cls_loss = torch.tensor(0.0).to(cls_out_flat.device)
            tnr = 100.0 * torch.ones(1).to(cls_out_flat.device)

        loss_cls = 0.5 * neg_cls_loss + 0.5 * pos_cls_loss
        loss_bbox = pos_reg_loss

        return loss_cls, loss_bbox, tpr, tnr, num_pos, num_neg

    def get_losses_images(self,
                          batch_cls_out_flat,
                          batch_reg_out_flat,
                          batch_gt_labels,
                          batch_gt_bboxes,
                          concat_level_anchors,
                          num_level_anchors=None):
        batch_pos_indices, batch_neg_indices, batch_assigned_labels, batch_assigned_bboxes = multi_apply(
            self._target_single_image,
            batch_gt_labels,
            batch_gt_bboxes,
            concat_anchors=concat_level_anchors
        )
        num_gts = (batch_gt_labels > 0).sum().float()
        batch = len(batch_gt_bboxes)

        losses_cls, losses_bbox, tnr, tpr, num_pos, num_neg = multi_apply(
            self._loss_single_image,
            [concat_level_anchors] * batch,
            batch_cls_out_flat,
            batch_reg_out_flat,
            batch_assigned_labels,
            batch_assigned_bboxes,
            batch_pos_indices,
            batch_neg_indices,
            total_batch=len(batch_gt_labels)
        )

        return dict(loss_bbox=losses_bbox,
                    loss_cls=losses_cls, tnr=tnr, tpr=tpr,
                    num_gts=num_gts, num_pos=num_pos, num_neg=num_neg)

    def get_losses_levels(self,
                          batch_cls_out_flat,
                          batch_reg_out_flat,
                          batch_gt_labels,
                          batch_gt_bboxes,
                          concat_level_anchors,
                          num_level_anchors):
        batch_pos_indices, batch_neg_indices, batch_label_targets, batch_bbox_targets = multi_apply(
            self._target_single_image,
            batch_gt_labels,
            batch_gt_bboxes,
            concat_anchors=concat_level_anchors
        )
        num_gts = (batch_gt_labels > 0).sum().float()
        batch = len(batch_gt_bboxes)

        level_anchors = images_to_levels([concat_level_anchors] * batch, num_level_anchors)
        level_cls_out = images_to_levels(batch_cls_out_flat, num_level_anchors)
        level_reg_out = images_to_levels(batch_reg_out_flat, num_level_anchors)
        level_label_targets = images_to_levels(batch_label_targets, num_level_anchors)
        level_bbox_targets = images_to_levels(batch_bbox_targets, num_level_anchors)
        level_pos_indices = images_to_levels(batch_pos_indices, num_level_anchors)
        level_neg_indices = images_to_levels(batch_neg_indices, num_level_anchors)

        losses_cls, losses_bbox, tpr, tnr, num_pos, num_neg = multi_apply(
            self._loss_single_level,
            level_anchors,
            level_cls_out,
            level_reg_out,
            level_label_targets,
            level_bbox_targets,
            level_pos_indices,
            level_neg_indices,
        )

        return dict(rpn2_loss_bbox=losses_bbox,
                    rpn2_loss_cls=losses_cls,
                    tnr=tnr, tpr=tpr,
                    num_gts=num_gts, num_pos=num_pos, num_neg=num_neg)

    def get_losses(self, *args, **kwargs):
        if self.level_first:
            return self.get_losses_levels(*args, **kwargs)
        else:
            return self.get_losses_images(*args, **kwargs)

    def get_bboxes(self, batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors):
        batch_bboxes = multi_apply(
            self._bboxes_single_image,
            batch_cls_out_flat,
            batch_reg_out_flat,
            concat_level_anchors=concat_level_anchors
        )[0]
        return torch.stack(batch_bboxes)

    def _bboxes_single_image(self, cls_out_flat, reg_out_flat, concat_level_anchors):
        nms_fun = self.nms['nms_fun']['type']
        if nms_fun == 'nms':
            if cls_out_flat.is_cuda:
                nms_fun = nmsNd_cuda
            else:
                nms_fun = nmsNd_pytorch
        elif nms_fun == 'softnms':
            nms_fun = softnmsNd_pytorch
        score_threshold = self.nms['score_thr']
        iou_threshold = self.nms['nms_fun']['iou_threshold']
        max_per_img = self.nms['max_per_img']

        self.try_to_info("postprocess reg_out_flat, anchors_flat",
                         cls_out_flat.shape, reg_out_flat.shape, concat_level_anchors.shape)

        cls_out_flat = cls_out_flat.sigmoid()

        reg_out_flat = self.bbox_coder.decode(concat_level_anchors, reg_out_flat)
        self.try_to_info('bboxes', reg_out_flat)

        results = np.ones((0, self.dim * 2 + 2)) * -1
        for i in range(cls_out_flat.shape[1]):  # multi classes
            scores = torch.squeeze(cls_out_flat[:, i])
            labels = torch.ones_like(scores) * (i + 1)
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            bboxes = reg_out_flat[scores_over_thresh]
            labels = labels[scores_over_thresh]

            nms_idx, _ = nms_fun(torch.cat([bboxes, scores.unsqueeze(-1)], dim=1), iou_threshold)
            scores, labels, bboxes = scores[nms_idx], labels[nms_idx], bboxes[nms_idx]

            high_score_indices = np.where(scores.cpu() > score_threshold)[0]
            for j in range(len(high_score_indices)):
                bbox = bboxes[high_score_indices[j]]
                bbox = bbox.detach().cpu().numpy()
                label = int(labels[high_score_indices[j]].item())
                score = scores[high_score_indices[j]].item()
                self.try_to_info("postprocess", [*bbox, label, score])
                results = np.concatenate([
                    results,
                    np.array([[*bbox, label, score]])
                ], axis=0)

        results = torch.from_numpy(results)
        _, order = torch.sort(results[:, -2], descending=True)
        results = results[order[:max_per_img]]
        padded_results = cls_out_flat.new_ones((max_per_img, self.dim * 2 + 2)) * -1
        padded_results[:results.shape[0]] = results
        return [padded_results]

    def metric(self, results, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(results, ground_truth)
            metrics.update(one_metric)
        return metrics

    def forward_train(self,
                      multi_level_cls_out: (list, tuple),
                      multi_level_reg_out: (list, tuple),
                      batch_gt_labels,
                      batch_gt_bboxes):
        feature_shapes = [f.size()[2:] for f in multi_level_cls_out]
        # (Num, 2dim), [N1, N2, N3, ...]
        # self.info(1)
        concat_level_anchors, num_level_anchors = self.get_anchors(feature_shapes, device=multi_level_cls_out[0].device)

        # self.info(2)
        batch_cls_out_flat, batch_reg_out_flat = self.get_flatten_out(multi_level_cls_out, multi_level_reg_out)
        # self.info(3)
        losses = self.get_losses(batch_cls_out_flat,
                                 batch_reg_out_flat,
                                 batch_gt_labels,
                                 batch_gt_bboxes,
                                 concat_level_anchors,
                                 num_level_anchors)
        # self.info(4)
        # batch_bboxes = self.get_bboxes(batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors)
        if self.has_proposal():
            proposals = self.get_proposals(batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors)
            return losses, proposals
        else:
            return losses, None

    def forward_valid(self,
                      multi_level_cls_out: (list, tuple),
                      multi_level_reg_out: (list, tuple),
                      batch_gt_labels,
                      batch_gt_bboxes):
        feature_shapes = [f.size()[2:] for f in multi_level_cls_out]
        concat_level_anchors, num_level_anchors = self.get_anchors(feature_shapes, device=multi_level_cls_out[0].device)

        batch_cls_out_flat, batch_reg_out_flat = self.get_flatten_out(multi_level_cls_out, multi_level_reg_out)

        losses = self.get_losses(batch_cls_out_flat, batch_reg_out_flat,
                                 batch_gt_labels, batch_gt_bboxes,
                                 concat_level_anchors,
                                 num_level_anchors)

        if self.has_proposal():
            proposals = self.get_proposals(batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors)
            return losses, proposals
        else:
            batch_bboxes = self.get_bboxes(batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors)
            return losses, batch_bboxes

    def forward_infer(self,
                      multi_level_cls_out: (list, tuple),
                      multi_level_reg_out: (list, tuple)):
        feature_shapes = [f.size()[2:] for f in multi_level_cls_out]
        concat_level_anchors, num_level_anchors = self.get_anchors(feature_shapes, device=multi_level_cls_out[0].device)

        batch_cls_out_flat, batch_reg_out_flat = self.get_flatten_out(multi_level_cls_out, multi_level_reg_out)

        if self.has_proposal():
            proposals = self.get_proposals(batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors)
            return proposals
        else:
            batch_bboxes = self.get_bboxes(batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors)
            return batch_bboxes

    def has_proposal(self):
        return self.get('proposal_cfg', None) is not None

    def get_proposals(self, batch_cls_out_flat, batch_reg_out_flat, concat_level_anchors):
        if not self.has_proposal():
            return None

        batch_proposals = []
        for (cls_out_flat, reg_out_flat) in zip(batch_cls_out_flat, batch_reg_out_flat):
            proposals = self._proposal_single_image(
                cls_out_flat,
                reg_out_flat,
                concat_level_anchors)
            batch_proposals.append(proposals)

        # batch_proposals = torch.tensor(batch_proposals).float().to(concat_level_anchors.device)
        return batch_proposals

    def _proposal_single_image(self, cls_out_flat, reg_out_flat, concat_level_anchors):
        pre_nms_limit = self.proposal_cfg['nms_pre']
        max_output_num = self.proposal_cfg['max_num']
        nms_threshold = self.proposal_cfg['nms_thr']

        self.try_to_info("Proposal")

        cls_out_flat_scores = cls_out_flat.squeeze(-1).sigmoid()  # squeeze class channel

        scores, order = torch.sort(cls_out_flat_scores, descending=True)
        order = order[:pre_nms_limit]
        scores = scores[:pre_nms_limit]
        deltas = reg_out_flat[order, :]
        cur_anchors = concat_level_anchors[order, :].clone()

        boxes = self.bbox_coder.decode(cur_anchors, deltas)

        dets = torch.cat([boxes, scores.unsqueeze(-1)], dim=1)
        if cls_out_flat.is_cuda:
            keep, _ = nmsNd_cuda(dets, nms_threshold)
        else:
            keep, _ = nmsNd_pytorch(dets, nms_threshold)
        keep = keep[:max_output_num]
        proposals = boxes[keep]
        # keep = keep[:max_output_num]
        # proposals = boxes[keep].detach().cpu().numpy()
        # scores = scores[keep].detach().cpu().numpy()
        #
        # proposals = np.concatenate([proposals, np.ones_like(scores[:, None]), scores[:, None]], axis=1)

        # self.info("proposal", proposals.shape)
        return proposals