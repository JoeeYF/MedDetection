import torch.nn as nn
import torch
import warnings
import numpy as np
from meddet.model.heads.dense_heads.base_dense_head import BaseDenseHead
from meddet.task.builder import build_loss, build_metric
from meddet.utils import multi_apply
from meddet.model.heads.utils.util_gaussion import *

from medvision.ops.torch import nmsNd_pytorch, softnmsNd_pytorch

try:
    from medvision.ops import nms_nd as nmsNd_cuda
except ImportError:
    warnings.warn('Using medvision.ops.torch.nmsNd_pytorch instead.')
    from medvision.ops.torch import nmsNd_pytorch as nmsNd_cuda


class CenterNetHead(BaseDenseHead):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 dim,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 test_cfg=None,
                 nms=None,
                 metrics=None):

        super(CenterNetHead, self).__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 3)
        self.offset_head = self._build_head(in_channel, feat_channel, 3)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

        self.test_cfg = test_cfg
        self.nms = nms
        self.metrics = nn.ModuleList([build_metric(metric) for metric in metrics])

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""

        layer = nn.Sequential(
            self.build_conv(self.dim, in_channel, feat_channel, kernel_size=3, padding=1),
            self.build_act(),
            self.build_conv(self.dim, feat_channel, out_channel, kernel_size=1, padding=1),
        )

        return layer

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return self.forward_single(feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        return center_heatmap_pred, wh_pred, offset_pred

    def metric(self, results, ground_truth):
        metrics = {}
        for metric in self.metrics:
            one_metric = metric(results, ground_truth)
            metrics.update(one_metric)
        return metrics

    def get_losses(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   gt_bboxes,
                   gt_labels,
                   image_shape,
                   gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     image_shape)

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred,
            center_heatmap_target,
            avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_d, img_h, img_w = img_shape
        bs, _, feat_d, feat_h, feat_w = feat_shape
        depth_ratio = float(feat_d / img_d)
        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_d, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, self.dim, feat_d, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, self.dim, feat_d, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros([bs, self.dim, feat_d, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [3]]) * depth_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [4]]) * width_ratio / 2
            center_z = (gt_bbox[:, [2]] + gt_bbox[:, [5]]) * height_ratio / 2

            # N,3
            gt_centers = torch.cat((center_x, center_y, center_z), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int, ctz_int = ct.int()
                ctx, cty, ctz = ct

                scale_box_d = (gt_bbox[j][3] - gt_bbox[j][0]) * depth_ratio
                scale_box_h = (gt_bbox[j][4] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][5] - gt_bbox[j][2]) * width_ratio

                # radius = gaussian_radius([scale_box_d, scale_box_h, scale_box_w],
                #                          min_overlap=0.3)
                # radius = max(0, int(radius))
                radius = int(pow(scale_box_d * scale_box_h * scale_box_w, 1 / 3)) * 0.2
                radius = max(0, int(radius))
                ind = gt_label[j].int()
                if ind==-1:
                    continue
                ind-=1
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int, ctz_int], radius)

                wh_target[batch_id, 0, ctz_int, cty_int, ctx_int] = scale_box_d
                wh_target[batch_id, 1, ctz_int, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 2, ctz_int, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, ctz_int, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, ctz_int, cty_int, ctx_int] = cty - cty_int
                offset_target[batch_id, 2, ctz_int, cty_int, ctx_int] = ctz - ctz_int

                wh_offset_target_weight[batch_id, :, ctz_int, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   image_shape):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(offset_preds) == 1

        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            wh_preds[0],
            offset_preds[0],
            image_shape,
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        results = multi_apply(self._bboxes_single_image,
                              batch_det_bboxes,batch_labels)[0]

        return torch.stack(results)

    def _bboxes_single_image(self, det_bboxes, labels):

        nms_fun = self.nms['nms_fun']['type']
        if nms_fun == 'nms':
            if det_bboxes.is_cuda:
                nms_fun = nmsNd_cuda
            else:
                nms_fun = nmsNd_pytorch
        elif nms_fun == 'softnms':
            nms_fun = softnmsNd_pytorch
        score_threshold = self.nms['score_thr']
        iou_threshold = self.nms['nms_fun']['iou_threshold']
        max_per_img = self.nms['max_per_img']

        results = np.ones((0, self.dim * 2 + 2)) * -1
        for i in range(self.num_classes):  # multi classes
            target_inds = labels == (i + 1)
            scores = torch.squeeze(det_bboxes[target_inds, -1])
            bboxes = det_bboxes[target_inds, :-1]
            labels = torch.ones_like(scores) * (i + 1)
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue

            scores = scores[scores_over_thresh]
            bboxes = bboxes[scores_over_thresh]
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
        padded_results = det_bboxes.new_ones((max_per_img, self.dim * 2 + 2)) * -1
        padded_results[:results.shape[0]] = results
        return [padded_results]

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       img_shape,
                       k=100,
                       kernel=3):
        """Transform outputs into detections raw bbox prediction.

        Args:
            center_heatmap_pred (Tensor): center predict heatmap,
               shape (B, num_classes, D, H, W).
            wh_pred (Tensor): wh predict, shape (B, 2, H, W).
            offset_pred (Tensor): offset predict, shape (B, 2, H, W).
            img_shape (list[int]): image shape in [h, w] format.
            k (int): Get top k center keypoints from heatmap. Default 100.
            kernel (int): Max pooling kernel for extract local maximum pixels.
               Default 3.

        Returns:
            tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
               the following Tensors:

              - batch_bboxes (Tensor): Coords of each box with shape (B, k, 7)
              - batch_topk_labels (Tensor): Categories of each box with \
                  shape (B, k)
        """
        depth, height, width = center_heatmap_pred.shape[2:]
        inp_d, inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_zs, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

        wh = transpose_and_gather_feat(wh_pred, batch_index)
        offset = transpose_and_gather_feat(offset_pred, batch_index)

        topk_xs = topk_xs + offset[..., 0]
        topk_ys = topk_ys + offset[..., 1]
        topk_zs = topk_zs + offset[..., 2]

        tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
        tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
        tl_z = (topk_zs - wh[..., 2] / 2) * (inp_d / depth)

        br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
        br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)
        br_z = (topk_zs + wh[..., 2] / 2) * (inp_d / depth)

        batch_bboxes = torch.stack([tl_x, tl_y, tl_z, br_x, br_y, br_z], dim=2)
        batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]), dim=-1)
        return batch_bboxes, batch_topk_labels

    def forward_train(self,
                      center_heatmap_pred,
                      wh_pred,
                      offset_pred,
                      batch_gt_bboxes,
                      batch_gt_labels,
                      image_shape):

        loss_dict = self.get_losses([center_heatmap_pred],
                                    [wh_pred],
                                    [offset_pred],
                                    batch_gt_bboxes,
                                    batch_gt_labels,
                                    image_shape)
        return loss_dict

    def forward_valid(self,
                      center_heatmap_pred,
                      wh_pred,
                      offset_pred,
                      batch_gt_bboxes,
                      batch_gt_labels,
                      image_shape):

        loss_dict = self.get_losses(center_heatmap_pred,
                                    wh_pred,
                                    offset_pred,
                                    batch_gt_bboxes,
                                    batch_gt_labels,
                                    image_shape)

        bboxes = self.get_bboxes([center_heatmap_pred],
                                 [wh_pred],
                                 [offset_pred],
                                 image_shape)

        return loss_dict, bboxes

    def forward_infer(self,
                      center_heatmap_pred,
                      wh_pred,
                      offset_pred,
                      image_shape):

        bboxes = self.get_bboxes([center_heatmap_pred],
                                 [wh_pred],
                                 [offset_pred],
                                 image_shape)

        return bboxes

