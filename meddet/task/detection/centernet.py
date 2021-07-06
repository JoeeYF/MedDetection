from meddet import data
from numpy import core
import torch
from collections import OrderedDict

from meddet.task import builder
from ..task import BaseTask


class CenterNet(BaseTask):
    def __init__(self,
                 dim,
                 backbone,
                 neck=None,
                 head=None):
        super(CenterNet, self).__init__()
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

    def extract_feat(self, img, coord=None):
        if coord is not None:
            x = self.backbone(img, coord)
        else:
            x = self.backbone(img)

        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        image_shape = list(img.shape)[2:]
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]
        print(gt_labels)
        print(gt_bboxes)
        self.try_to_info(gt_bboxes.shape)
        self.try_to_info(gt_labels.shape)

        coord = None if 'gt_coord' not in data_batch else data_batch['gt_coord']
        feats = self.extract_feat(img, coord)

        net_output = self.head(feats)
        loss_dict = self.head.forward_train(*net_output,gt_bboxes, gt_labels, image_shape)

        return loss_dict, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        image_shape = list(img.shape)[2:]
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        coord = None if 'gt_coord' not in data_batch else data_batch['gt_coord']
        feats = self.extract_feat(img, coord)

        net_output = self.head(feats)
        loss_dict, batch_bboxes = self.head.forward_valid(*net_output, gt_bboxes, gt_labels, image_shape)

        metrics = self.metric(data_batch, batch_bboxes)

        metrics_losses = {}
        metrics_losses.update(metrics)
        metrics_losses.update(loss_dict)

        return metrics_losses, batch_bboxes, net_output

    def forward_infer(self, data_batch, *args, **kwargs):

        img = data_batch['img']
        image_shape = list(img.shape)[2:]
        coord = None if 'gt_coord' not in data_batch else data_batch['gt_coord']
        feats = self.extract_feat(img, coord)

        net_output = self.head(feats)
        batch_bboxes = self.head.forward_infer(*net_output, image_shape)

        return batch_bboxes, net_output

    def metric(self, data_batch: dict, net_output):
        assert 'gt_det' in data_batch.keys()
        label = data_batch['gt_det']

        self.try_to_info("roi_results", net_output)
        one_metric = self.head.metric(net_output, label)
        metrics = {}
        for k, v in one_metric.items():
            metrics['roi_' + k] = v
        return metrics


if __name__ == "__main__":
    import numpy as np
    from meddet.model.backbones.spmnet import SCPMNet, BasicBlock
    from meddet.model.heads.dense_heads.centernet_head import CenterNetHead

    backbone = SCPMNet(3,18,1,64)

    test_cfg = dict(topk=20, local_maximum_kernel=3)
    nms = dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.15,
        nms_fun=dict(type='nms', iou_threshold=0.1),
        max_per_img=100)

    # head = CenterNetHead(3, 64, 96, 1, test_cfg=test_cfg)
    head = dict(type='CenterNetHead', dim=3, in_channel=64, feat_channel=96, num_classes=1, test_cfg=test_cfg, nms=nms)
    model = CenterNet(3, backbone, head=head)

    model.setLog()

    batch = 2
    image = torch.rand((batch, 1, 96, 96, 96))
    # gra = torch.rand((2, 5, 64, 64))
    # label = torch.ones((batch, 32, 32))

    gt_bboxes = np.array([[0, 0, 0, 4.2, 4.2, 4.2],
                          [0, 1.1, 1.2, 5, 5.2, 5.3],
                          [3, 3, 3, 8, 8, 8]])

    gt_labels = np.array([1, 1, 1])[..., None]
    label = [torch.stack([torch.tensor(gt_bboxes).float()] * batch, dim=0),
             torch.stack([torch.tensor(gt_labels).float()] * batch, dim=0)]

    print(label[0].shape)
    print(label[1].shape)
    label = torch.cat(label, dim=-1)
    zz, yy, xx = np.meshgrid(
        np.linspace(-0.5, 0.5, 96),
        np.linspace(-0.5, 0.5, 96),
        np.linspace(-0.5, 0.5, 96),
        indexing='ij')
    coord = np.stack([zz, yy, xx], 0).astype('float32')
    coord = torch.from_numpy(np.stack([coord,coord]))
    print(label.shape)
    data_batch = dict(img=image,
                      img_shape=[96, 96, 96],
                      gt_det=label,
                      gt_coord=coord)

    losses, _, net_output = model.forward_train(data_batch)
    # self.try_to_log(losses)

    loss_total = 0
    for name, loss in losses.items():
        print(name, loss)
        loss_total = loss_total + loss

    print("backward...")
    loss_total.backward()

    """ view roi extractor"""
    # self.try_to_log("\n\n\n\nrois", rois[-1])
    # plt.imshow(feats[0][1].mean(dim=0).detach().cpu().numpy())
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(roi_features[-1].mean(dim=0).detach().cpu().numpy())
    # plt.colorbar()
    # plt.show()
    #
    # _, x1, y1, x2, y2 = rois[-1]
    # print(int(y1//2), int(y2//2), int(x1//2), int(x2//2))
    # img = feats[0][1].mean(dim=0).detach().cpu().numpy()
    # print(img.shape)
    # plt.imshow(img[int(y1//2):int(y2//2), int(x1//2):int(x2//2)])
    # plt.colorbar()
    # plt.show()
