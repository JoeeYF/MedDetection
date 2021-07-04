

import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from meddet.task import builder
from meddet.model.nnModules import ComponentModule
from ..task import BaseTask


class DeepLungSim(BaseTask):
    def __init__(self,
                 dim,
                 backbone,
                 neck=None,
                 head=None):
        super(DeepLungSim, self).__init__()
        self.dim = dim
        if backbone:
            if isinstance(backbone, dict):
                backbone['dim'] = dim
            self.backbone = builder.build_backbone(backbone)
        if neck:
            if isinstance(neck, dict):
                neck['dim'] = dim
            self.neck = builder.build_neck(neck)
        if head:
            if isinstance(head, dict):
                head['dim'] = dim
            self.head = builder.build_head(head)

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    def forward_train(self, data_batch: dict, *args, **kwargs):
        img = data_batch['img']
        # gt_coord = data_batch['gt_coord']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        self.try_to_info(gt_bboxes)
        self.try_to_info(gt_labels)
        # self.info('f')
        feats = self.backbone(img)
        # feats = self.backbone(img, F.max_pool3d(gt_coord, kernel_size=4))
        # self.info('f2')
        net_output = self.head(feats)
        # self.info('los')
        loss_dict, batch_bboxes = self.head.forward_train(*net_output, gt_labels, gt_bboxes)
        # self.info('done')

        return loss_dict, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        # gt_coord = data_batch['gt_coord']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        self.try_to_info(gt_bboxes)
        self.try_to_info(gt_labels)

        feats = self.backbone(img)
        # feats = self.backbone(img, F.max_pool3d(gt_coord, kernel_size=4))

        net_output = self.head(feats)
        loss_dict, batch_bboxes = self.head.forward_valid(*net_output, gt_labels, gt_bboxes)

        metrics = self.metric(data_batch, batch_bboxes)

        metrics_losses = OrderedDict()
        metrics_losses.update(metrics)
        metrics_losses.update(loss_dict)

        return metrics_losses, batch_bboxes, net_output

    def forward_infer(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        image_shape = img.shape[2:]

        feats = self.backbone(img)
        net_output = self.head(feats)

        batch_bboxes = self.head.forward_infer(*net_output)
        return batch_bboxes, net_output

    def metric(self, data_batch: dict, prediction):
        assert 'gt_det' in data_batch.keys()
        label = data_batch['gt_det']
        metrics = self.head.metric(prediction, label)
        return metrics