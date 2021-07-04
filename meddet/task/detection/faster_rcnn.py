

from meddet import data
from numpy import core
import torch
from collections import OrderedDict

from meddet.task import builder
from ..task import BaseTask



class FasterRCNN(BaseTask):
    def __init__(self,
                 dim,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None):
        super(FasterRCNN, self).__init__()
        self.dim = dim
        if backbone:
            if isinstance(backbone, dict):
                backbone['dim'] = dim
            self.backbone = builder.build_backbone(backbone)
        if neck:
            if isinstance(neck, dict):
                neck['dim'] = dim
            self.neck = builder.build_neck(neck)
        if rpn_head:
            if isinstance(rpn_head, dict):
                rpn_head['dim'] = dim
            self.rpn_head = builder.build_head(rpn_head)
        if roi_head:
            if isinstance(roi_head, dict):
                roi_head['dim'] = dim
            self.roi_head = builder.build_head(roi_head)

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
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]
        
        self.try_to_info(gt_bboxes)
        self.try_to_info(gt_labels)
        
        coord = None if 'gt_coord' not in data_batch else data_batch['gt_coord']
        feats = self.extract_feat(img, coord)

        net_output = self.rpn_head(feats)
        rpn_loss_dict, batch_proposals = self.rpn_head.forward_train(*net_output, gt_labels, gt_bboxes)

        roi_loss_dict = self.roi_head.forward_train(feats, batch_proposals, gt_labels, gt_bboxes)

        losses = {}
        losses.update(rpn_loss_dict)
        losses.update(roi_loss_dict)

        return losses, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        gt_labels = data_batch['gt_det'][..., 2 * self.dim]
        gt_bboxes = data_batch['gt_det'][..., :2 * self.dim]

        coord = None if 'gt_coord' not in data_batch else data_batch['gt_coord']
        feats = self.extract_feat(img,coord)

        net_output = self.rpn_head(feats)
        rpn_loss_dict, batch_proposals = self.rpn_head.forward_valid(*net_output, gt_labels, gt_bboxes)

        roi_loss_dict, batch_bboxes = self.roi_head.forward_valid(feats, batch_proposals, gt_labels, gt_bboxes)

        metrics = self.metric(data_batch, batch_bboxes)

        metrics_losses = {}
        metrics_losses.update(metrics)
        metrics_losses.update(rpn_loss_dict)
        metrics_losses.update(roi_loss_dict)

        return metrics_losses, batch_bboxes, net_output

    def forward_infer(self, data_batch, *args, **kwargs):
        # img = data_batch['img']
        # image_shape = img.shape[2:]
        #
        # feats = self.extract_feat(img)
        # cls_outs, reg_outs = self.rpn_head(feats)
        #
        # self.try_to_info("features", len(feats))
        #
        # cls_outs_flat, reg_outs_flat, multi_anchors_flat = self.rpn_head.preprocess(cls_outs, reg_outs, image_shape)
        #
        # self.try_to_info(cls_outs_flat.shape, reg_outs_flat.shape, multi_anchors_flat.shape)
        #
        # batch_proposals, batch_indices = self.rpn_head.proposal(cls_outs_flat, reg_outs_flat, multi_anchors_flat)
        # self.try_to_info("self.rpn_head.proposal", batch_proposals.shape, batch_indices.shape)
        # self.try_to_info("bbox", batch_proposals[:10, :])
        #
        # rois = torch.cat((batch_indices[:, None].float().to(batch_proposals.device), batch_proposals), dim=1)
        # # self.try_to_log(rois.shape)
        #
        # cls, reg = self.roi_head(feats, rois)
        # # self.try_to_log("cls, reg", cls.shape, reg.shape)
        # # self.try_to_log(cls[:10, :], reg[:10, :])
        #
        # # roi_results = self.rpn_head.postprocess(cls_outs_flat, reg_outs_flat, multi_anchors_flat, image_shape)
        # roi_results = self.roi_head.postprocess(cls, reg, batch_proposals, batch_indices, image_shape)
        #
        # net_output = (cls_outs, reg_outs)
        # return roi_results, net_output
        img = data_batch['img']

        coord = None if 'gt_coord' not in data_batch else data_batch['gt_coord']
        feats = self.extract_feat(img,coord)

        net_output = self.rpn_head(feats)
        batch_proposals = self.rpn_head.forward_infer(*net_output)

        batch_bboxes = self.roi_head.forward_infer(feats, batch_proposals)

        return batch_bboxes, net_output

    # def forward(self, batch, return_loss=True):
    #     # print(batch['img'].device)
    #     # print(len(batch['img']))
    #     # print(batch['gt_labels'].device)
    #     # print(batch['gt_bboxes'].device)
    #     # print(type(batch['img']))
    #     # try:
    #     #     print(self.device)
    #     # except Exception as e:
    #     #     pass
    #
    #     img = batch['img']
    #     # self.try_to_log("SimpleDetector", batch.keys())
    #     if not return_loss:
    #         return self.forward_infer(batch['img_meta'], img)
    #     else:
    #         gt_labels = batch['gt_labels']
    #         gt_bboxes = batch['gt_bboxes']
    #         # self.try_to_log(img.shape, label.shape)
    #         return self.forward_train(batch['img_meta'], img, (gt_bboxes, gt_labels))

    def metric(self, data_batch: dict, net_output):
        assert 'gt_det' in data_batch.keys()
        label = data_batch['gt_det']

        self.try_to_info("roi_results", net_output)
        one_metric = self.roi_head.metric(net_output, label)
        metrics = {}
        for k, v in one_metric.items():
            metrics['roi_' + k] = v
        return metrics


# if __name__ == "__main__":
#     import numpy as np
#
#     backbone = dict(
#         type="UNetEncoder",
#         in_channels=1,
#         base_width=16,
#         out_indices=[0, 1, 2, 3, 4]
#     )
#     neck = dict(
#         type="FPN",
#         in_channels=[16, 32, 64, 128, 256],
#         out_channels=16,
#         start_level=0,
#         add_extra_convs=True,
#         num_outs=5,
#         out_indices=(0, 1),
#     )
#     rpn_head = dict(
#         type="RPNHead",
#         in_channels=16,
#         base_scales=4,
#         scales=[4, 6, 7],
#         ratios=[1]
#     )
#     bbox_head = dict(
#         type="RcnnHead",
#         feat_channels=16,
#         num_classes=2
#     )
#
#     model = FasterRCNNV2(2, backbone, neck, rpn_head, bbox_head)
#     model.setLog()
#
#     batch = 2
#     image = torch.rand((batch, 1, 32, 64))
#     # gra = torch.rand((2, 5, 64, 64))
#     # label = torch.ones((batch, 32, 32))
#
#     gt_bboxes = np.array([[0, 0, 4.2, 4.2],
#                           [0, 1.1, 5, 5.2],
#                           [3, 3, 8, 8]])
#
#     gt_labels = np.array([1, 1, 1])
#     label = (torch.stack([torch.tensor(gt_bboxes).float()] * batch, dim=0),
#              torch.stack([torch.tensor(gt_labels).float()] * batch, dim=0))
#     # print(label[0].shape, label[1].shape)
#
#     losses, net_output = model.forward_train(None, image, label)
#     # self.try_to_log(losses)
#
#     loss_total = 0
#     for name, loss in losses.items():
#         print(name, loss)
#         loss_total = loss_total + loss
#
#     print("backward...")
#     loss_total.backward()
#
#     """ view roi extractor"""
#     # self.try_to_log("\n\n\n\nrois", rois[-1])
#     # plt.imshow(feats[0][1].mean(dim=0).detach().cpu().numpy())
#     # plt.colorbar()
#     # plt.show()
#     #
#     # plt.imshow(roi_features[-1].mean(dim=0).detach().cpu().numpy())
#     # plt.colorbar()
#     # plt.show()
#     #
#     # _, x1, y1, x2, y2 = rois[-1]
#     # print(int(y1//2), int(y2//2), int(x1//2), int(x2//2))
#     # img = feats[0][1].mean(dim=0).detach().cpu().numpy()
#     # print(img.shape)
#     # plt.imshow(img[int(y1//2):int(y2//2), int(x1//2):int(x2//2)])
#     # plt.colorbar()
#     # plt.show()
