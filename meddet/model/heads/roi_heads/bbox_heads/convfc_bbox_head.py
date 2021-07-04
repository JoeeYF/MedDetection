
# -*- coding:utf-8 -*-
from typing import Union, List
from torch import nn
import torch
import numpy as np
import warnings

from meddet.model.nnModules import ComponentModule

from meddet.task.builder import build_loss, build_metric
from meddet.model.heads.utils import build_coder, build_assigner, build_sampler, build_extractor

try:
    from medvision.ops import RoIAlign
except ImportError:
    warnings.warn('Using medvision.ops.torch.RoIAlign instead.')
    from medvision.ops.torch import RoIAlign

from medvision.ops.torch import nmsNd_pytorch, softnmsNd_pytorch, bbox2roi
from meddet.utils import multi_apply


class ConvFCBBoxHead(ComponentModule):
    def __init__(self,
                 dim,
                 num_classes,
                 in_channels,
                 feat_channels: int = 256,
                 fc_channels: int = 1024,
                 roi_output_size: int = 7,
                 assigner: dict = None,
                 sampler: dict = None,
                 bbox_coder: dict = None,
                 nms: dict = None,
                 losses: dict = None,
                 metrics: Union[List[dict], List[object]] = None):
        super(ConvFCBBoxHead, self).__init__()

        if losses.get('cls', None) is None:
            losses['cls'] = dict(type='CrossEntropyLoss')
        if losses.get('reg', None) is None:
            losses['reg'] = dict(type='SmoothL1Loss', beta=1.0 / 9.0, reduction='mean', loss_weight=1.0)

        self.dim = dim
        self.in_channels = in_channels  # using sum instead of cat
        self.num_classes = num_classes
        self.roi_output_size = (roi_output_size,) * self.dim
        self.feat_channels = feat_channels
        self.fc_channels = fc_channels
        self.flatten_features = self.feat_channels * np.prod(np.array(self.roi_output_size))

        bbox_coder.update({'dim': self.dim})
        self.bboxCoder = build_coder(bbox_coder)

        self.assigner_cfg = assigner
        self.sampler_cfg = sampler
        self.assigner = build_assigner(assigner)
        self.sampler = build_sampler(sampler)
        self.nms = nms

        # self.add_gt_to_proposal = sampler['add_gt_as_proposals']

        self.extractors = None
        self.b_extractors = nn.ModuleList([RoIAlign(self.roi_output_size, 1 / stride, 1) for stride in [4]])

        self.criterion_cls = build_loss(losses['cls'])
        self.criterion_reg = build_loss(losses['reg'])
        self.metrics = nn.ModuleList([build_metric(metric) for metric in metrics])

        # self.init_weights()
        self._init_layers()

    def _init_layers(self):
        self.shared_convs = nn.Sequential(
            self.build_conv(self.dim, self.in_channels, self.feat_channels, kernel_size=3, padding=1),
            # self.build_norm(self.dim, self.mid_channels),
            self.build_act(inplace=True),
            # self.build_conv(self.dim, self.mid_channels, self.mid_channels, kernel_size=1),
            # # self.build_norm(self.dim, self.mid_channels),
            # self.build_act(inplace=True)
        )
        # self.shared_fcs = nn.Linear(self.flatten_features, self.fc_channels)
        self.roi_cls = nn.Linear(self.flatten_features, self.num_classes + 1)
        self.roi_reg = nn.Linear(self.flatten_features, 2 * self.dim)

    def set_extractor(self, extractors):
        # pass
        # self.extractors = nn.ModuleList([RoIAlign((self.roi_output_size,) * self.dim, 1 / stride, 1) for stride in [4]])
        self.extractors = extractors

    def init_weights(self):
        pass
        # for m in self.modules():
        #     if self.is_conv(self.dim, m):
        #         n = np.prod(m.kernel_size) * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif self.is_norm(self.dim, m):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #
        # prior = 0.01
        # self.roi_cls.weight.data.fill_(0)
        # self.roi_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.roi_reg.weight.data.fill_(0)
        # self.roi_reg.bias.data.fill_(0)
        #
        # # normal_init(self.roi_cls, std=0.01)
        # # normal_init(self.roi_reg, std=0.01)

    # def forward(self,
    #             feats: List[torch.Tensor],
    #             rois: torch.Tensor):
    #     """
    #     :param feats list[tensor]
    #     :param rois: tensor, shape[roi_num, channel, ...]
    #     :return:
    #             cls: tensor, shape[roi_num, num_classes]
    #             reg: tensor, shape[roi_num, 2 * dim]
    #     """
    #     assert len(feats) == len(self.extractors) == len(self.strides), "match length"
    #     rois = rois.type(feats[0].type())
    #     self.try_to_info(rois.shape)
    #     roi_features = torch.stack([e(feats[i], rois) for i, e in enumerate(self.extractors)], dim=-1)
    #     self.try_to_info("roi_features", roi_features.shape)
    #     roi_features = torch.mean(roi_features, dim=-1)
    #     self.try_to_info("roi_features", roi_features.shape)
    #     out = self.shared_convs(roi_features)
    #     self.try_to_info("out", out.shape)
    #     out = out.view(-1, self.flatten_features)
    #     # out = self.shared_fcs(out)
    #     # self.try_to_info('shared fcs', out.shape)
    #     cls_out = self.roi_cls(out)
    #     reg_out = self.roi_reg(out)
    #
    #     self.try_to_info("cls, reg", cls_out.shape, reg_out.shape)
    #     self.try_to_info("cls, reg", cls_out[:10], reg_out[:10])
    #     if torch.isnan(cls_out[0, 0]):
    #         raise NotImplementedError
    #     return cls_out, reg_out

    def _bbox_forward(self, feats, rois):
        rois = rois.type(feats[0].type())

        roi_features = self.extractors(feats, rois)

        out = self.shared_convs(roi_features)
        out = out.view(-1, self.flatten_features)
        cls_out = self.roi_cls(out)
        reg_out = self.roi_reg(out)
        return cls_out, reg_out

    def forward_train(self, multi_level_features, batch_proposals, batch_gt_labels, batch_gt_bboxes):
        batch_multi_level_features = list(map(lambda a: [i[None, ...] for i in a], zip(*multi_level_features)))

        batch_rois, batch_labels, batch_deltas = multi_apply(
            self._target_single_image,
            batch_proposals,
            batch_gt_labels,
            batch_gt_bboxes,
            batch_multi_level_features
        )
        rois = bbox2roi(batch_rois)
        rois_deltas = torch.cat(batch_deltas, dim=0)
        rois_labels = torch.cat(batch_labels, dim=0)

        cls, reg = self._bbox_forward(multi_level_features, rois)
        roi_loss_dict = self.get_loss(cls, reg, rois_labels, rois_deltas)

        return roi_loss_dict

    def forward_valid(self, multi_level_features, batch_proposals, batch_gt_labels, batch_gt_bboxes):
        batch_multi_level_features = list(map(lambda a: [i[None, ...] for i in a], zip(*multi_level_features)))

        batch_rois, batch_labels, batch_deltas = multi_apply(
            self._target_single_image,
            batch_proposals,
            batch_gt_labels,
            batch_gt_bboxes,
            batch_multi_level_features
        )
        rois = bbox2roi(batch_rois)
        rois_deltas = torch.cat(batch_deltas, dim=0)
        rois_labels = torch.cat(batch_labels, dim=0)

        cls, reg = self._bbox_forward(multi_level_features, rois)
        roi_loss_dict = self.get_loss(cls, reg, rois_labels, rois_deltas)

        batch_bboxes = self.get_bboxes(rois, cls, reg)
        return roi_loss_dict, batch_bboxes

    def forward_infer(self, multi_level_features, batch_proposals):
        rois = bbox2roi(batch_proposals)

        cls, reg = self._bbox_forward(multi_level_features, rois)

        batch_bboxes = self.get_bboxes(rois, cls, reg)
        return batch_bboxes

    def _target_single_image(self, proposals, gt_labels, gt_bboxes, multi_level_features):
        gt_bboxes = gt_bboxes[gt_labels != -1]
        gt_labels = gt_labels[gt_labels != -1].long()

        self.try_to_info("=========")
        self.try_to_info(gt_bboxes.shape, gt_labels.shape)
        self.try_to_info("labels info", gt_bboxes, gt_labels)
        self.try_to_info(proposals.shape)

        assigned_labels, assigned_bboxes = self.assigner.assign(proposals, gt_bboxes, gt_labels)
        self.try_to_info(proposals[-10:])
        self.try_to_info(assigned_bboxes[-10:])
        self.try_to_info(assigned_labels[-10:])
        self.try_to_info("Positive", torch.sum(assigned_labels > 0))

        proposed_rois = bbox2roi([proposals])
        _cls, _ = self._bbox_forward(multi_level_features, proposed_rois)

        losses = torch.zeros_like(assigned_labels).float()
        with torch.no_grad():
            not_ignored = assigned_labels >= 0
            losses[not_ignored] = self.criterion_cls(_cls[not_ignored],
                                                     assigned_labels[not_ignored], reduction_override='none').squeeze(-1)

        pos_indices, neg_indices, proposals, assigned_labels, assigned_bboxes = self.sampler.sample(proposals,
                                                                                                    assigned_labels,
                                                                                                    assigned_bboxes,
                                                                                                    gt_labels,
                                                                                                    gt_bboxes,
                                                                                                    weight=losses)
        self.try_to_info(assigned_labels.shape, assigned_bboxes.shape, proposals.shape)
        self.try_to_info(len(pos_indices))
        # pos_indices = torch.where(assigned_labels > 0)[0]
        assigned_bboxes[pos_indices] = self.bboxCoder.encode(proposals[pos_indices], assigned_bboxes[pos_indices])

        """"""""""""""""""""""""""
        "set target"
        indices = torch.cat([pos_indices, neg_indices], dim=0)
        proposal_sample = proposals[indices]
        proposal_label = assigned_labels[indices]
        proposal_deltas = assigned_bboxes[indices]

        self.try_to_info("shape1", proposal_sample.shape, proposal_label.shape, proposal_deltas.shape)

        # 添加rois,deltas,labels
        self.try_to_info("shape2", proposal_sample.shape, proposal_label.shape, proposal_deltas.shape)
        return proposal_sample, proposal_label, proposal_deltas

    def get_loss(self, cls_out, reg_out, rois_labels, rois_deltas):
        # self.try_to_info("batch_proposals_cls", batch_proposals_cls[:10])
        # self.try_to_info("batch_proposals_cls", cls_out[:10, :])
        # self.try_to_info("batch_proposals_reg", batch_proposals_reg[:10, :])
        # self.try_to_info("batch_proposals_reg", reg_out[:10, :])
        self.try_to_info("shape", cls_out.shape, rois_labels.shape,
                         reg_out.shape, rois_deltas.shape)

        loss_cls = self.criterion_cls(cls_out, rois_labels.long())
        # self.try_to_info("1")
        # loss_cls_2 = torch.nn.CrossEntropyLoss()(cls_out, batch_proposals_cls.long())
        # self.try_to_info(loss_cls, loss_cls_2)
        if rois_labels.sum() > 0:
            pos_reg_out = reg_out[rois_labels > 0]
            target_reg = rois_deltas[rois_labels > 0]
            self.try_to_info(pos_reg_out.shape, target_reg.shape)
            loss_reg = self.criterion_reg(pos_reg_out, target_reg)
            # self.try_to_info(pos_reg_out.shape, target_reg.shape)
        else:
            loss_reg = torch.tensor(0.0).to(cls_out.device)

        self.try_to_info(loss_cls, loss_reg)
        loss_dict = {
            "roi_reg_loss": loss_reg,
            "roi_cls_loss": loss_cls,
            "roi_acc": 100.0 * torch.mean((torch.argmax(cls_out, dim=1).float() == rois_labels).float()),
        }
        return loss_dict

    def get_bboxes(self, rois, cls, reg):
        batch = int(torch.max(rois[:, 0]).item() + 1)
        self.try_to_info('batch', batch)
        batch_rois, batch_cls, batch_reg = [], [], []
        for i in range(batch):
            batch_rois.append(rois[rois[:, 0] == i])
            batch_cls.append(cls[rois[:, 0] == i])
            batch_reg.append(reg[rois[:, 0] == i])

        batch_bboxes = multi_apply(
            self._bboxes_single_image,
            batch_rois,
            batch_cls,
            batch_reg
        )[0]
        batch_bboxes = torch.tensor(batch_bboxes).float().to(cls.device)
        return batch_bboxes

    def _bboxes_single_image(self, roi, cls, reg):
        nms_fun = self.nms['nms_fun']['type']
        if nms_fun == 'nms':
            nms_fun = nmsNd_pytorch
        elif nms_fun == 'softnms':
            nms_fun = softnmsNd_pytorch
        score_threshold = self.nms['score_thr']
        iou_threshold = self.nms['nms_fun']['iou_threshold']
        max_per_img = self.nms['max_per_img']

        cls_out_flat = torch.softmax(cls, dim=1)
        scores, labels = torch.max(cls_out_flat[:, 1:], dim=1)
        bboxes = self.bboxCoder.decode(roi[:, 1:], reg)
        self.try_to_info('roi', roi)
        self.try_to_info('bboxes', bboxes)

        nms_idx, _ = nms_fun(torch.cat([bboxes, scores.unsqueeze(-1)], dim=1), iou_threshold)
        scores, labels, bboxes = scores[nms_idx], labels[nms_idx], bboxes[nms_idx]

        high_score_indices = np.where(scores.cpu() > score_threshold)[0]

        results = np.ones((max_per_img, self.dim * 2 + 2)) * -1
        for j in range(min(high_score_indices.shape[0], max_per_img)):
            bbox = bboxes[high_score_indices[j]]
            bbox = bbox.detach().cpu().numpy()
            label = int(labels[high_score_indices[j]].item()) + 1
            score = scores[high_score_indices[j]].item()
            self.try_to_info("postprocess", [*bbox, label, score])
            results[j] = [*bbox, label, score]

        return [results]

    def metric(self, results, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(results, ground_truth)
            metrics.update(one_metric)
        return metrics
