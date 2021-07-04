

import torch

from medvision.ops.torch import iouNd_pytorch, distNd_pytorch


class IoUAssigner:
    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: float,
                 min_pos_iou: float = .0,
                 match_low_quality: bool = False,
                 num_neg: int = None):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
        else:
            # 0. calculate iou
            iou = iouNd_pytorch(bboxes, gt_bboxes)  # [num_bboxes, num_gts]
            try:
                anchors_iou_max, anchors_iou_argmax = torch.max(iou, dim=1)  # [num_bboxes]
                gt_iou_max, gt_iou_argmax = torch.max(iou, dim=0)  # [num_gts]
            except:
                print(num_gts, num_bboxes)
                print(bboxes)
                print(gt_bboxes)
                print(iou)
                raise

            pos_indices = anchors_iou_max >= self.pos_iou_thr
            neg_indices = anchors_iou_max < self.neg_iou_thr

            # 1. assign -1 to each bboxes
            assigned_gt_indices = torch.ones_like(anchors_iou_argmax).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            # 3. assign positive: above positive IoU threshold
            assigned_gt_indices[pos_indices] = anchors_iou_argmax[pos_indices] + 1
            # 4. assign low quality gt to best anchors
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_iou_max[i] >= self.min_pos_iou:
                        # maybe multi assign
                        # max_iou_indices = iou[:, i] == gt_iou_max[i]
                        # assigned_gt_indices[max_iou_indices] = i + 1
                        assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = torch.ones_like(anchors_iou_argmax).long() * -1
            assigned_bboxes = torch.ones_like(bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


class MaxIoUAssigner:
    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: float,
                 min_pos_iou: float = .0,
                 match_low_quality: bool = False,
                 num_neg: int = None):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
        else:
            # 0. calculate iou
            iou = iouNd_pytorch(bboxes, gt_bboxes)  # [num_bboxes, num_gts]
            try:
                anchors_iou_max, anchors_iou_argmax = torch.max(iou, dim=1)  # [num_bboxes]
                gt_iou_max, gt_iou_argmax = torch.max(iou, dim=0)  # [num_gts]
            except:
                print(num_gts, num_bboxes)
                print(bboxes)
                print(gt_bboxes)
                print(iou)
                raise

            pos_indices = anchors_iou_max >= self.pos_iou_thr
            neg_indices = anchors_iou_max < self.neg_iou_thr

            # 1. assign -1 to each bboxes
            assigned_gt_indices = torch.ones_like(anchors_iou_argmax).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            # 3. assign positive: only max iou is assigned
            assigned_gt_indices[pos_indices] = -1
            for i in range(num_gts):
                if gt_iou_max[i] >= self.pos_iou_thr:
                    assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            # 4. assign low quality gt to best anchors
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_iou_max[i] >= self.min_pos_iou:
                        # maybe multi assign
                        # max_iou_indices = iou[:, i] == gt_iou_max[i]
                        # assigned_gt_indices[max_iou_indices] = i + 1
                        assigned_gt_indices[gt_iou_argmax[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = torch.ones_like(anchors_iou_argmax).long() * -1
            assigned_bboxes = torch.ones_like(bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


class DistAssigner:
    def __init__(self,
                 pos_dist_thr: float,
                 min_pos_dist: float = .0,
                 match_low_quality: bool = True,
                 num_neg: int = None):
        self.pos_dist_thr = pos_dist_thr
        self.min_pos_dist = min_pos_dist
        self.match_low_quality = match_low_quality
        self.num_neg = num_neg

    def assign(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor,
               gt_labels: torch.Tensor):
        """
        References: mmdetection
        Args:
            bboxes: with shape [N, 2dim]
            gt_bboxes: with shape [M, 2dim]
            gt_labels: with shape [M]

        Returns:
            assigned_gt_indices: assign each anchor with the index of matched gt bbox
                -1: ignore
                 0: negative
                +n: matched index of gt_bboxes, 1 based
            assigned_labels: assign class to each anchor
                -1: ignore
                 0: negative
                +n: matched classes, 1 based
            assigned_bboxes: assigned gt bboxes
        """
        assert bboxes.ndim == gt_bboxes.ndim
        assert gt_labels.dtype == torch.long
        num_bboxes, num_gts = bboxes.shape[0], gt_bboxes.shape[0]

        if num_gts == 0:
            # No truth, assign everything to background
            assigned_labels = torch.zeros(num_bboxes).long().to(bboxes.device)
            assigned_bboxes = torch.zeros_like(bboxes).to(bboxes.device)
        else:
            # 0. calculate iou
            dist = distNd_pytorch(bboxes, gt_bboxes)  # [num_bboxes, num_gts]
            try:
                anchors_dist_min, anchors_dist_argmin = torch.min(dist, dim=1)  # [num_bboxes]
                gt_dist_min, gt_dist_argmin = torch.min(dist, dim=0)  # [num_gts]
            except:
                print(num_gts, num_bboxes)
                print(bboxes)
                print(gt_bboxes)
                print(dist)
                raise

            pos_indices = anchors_dist_min < self.pos_dist_thr
            neg_indices = anchors_dist_min > self.pos_dist_thr

            # 1. assign -1 to each bboxes
            assigned_gt_indices = torch.ones_like(anchors_dist_argmin).long() * -1
            # 2. assign negative: below the negative threshold are set to be 0
            assigned_gt_indices[neg_indices] = 0
            # 3. assign positive: above positive IoU threshold
            assigned_gt_indices[pos_indices] = anchors_dist_argmin[pos_indices] + 1
            # 4. assign low quality gt to best anchors
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_dist_min[i] >= self.min_pos_dist:
                        # maybe multi assign
                        # min_dist_indices = dist[:, i] == gt_dist_min[i]
                        # assigned_gt_indices[min_dist_indices] = i + 1
                        assigned_gt_indices[gt_dist_argmin[i]] = i + 1

            pos_indices = torch.where(assigned_gt_indices > 0)[0]
            neg_indices = torch.where(assigned_gt_indices == 0)[0]

            if self.num_neg is not None:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:self.num_neg]]

            assigned_labels = torch.ones_like(anchors_dist_argmin).long() * -1
            assigned_bboxes = torch.ones_like(bboxes) * -1.0
            if pos_indices.numel() > 0:
                assigned_labels[pos_indices] = gt_labels[assigned_gt_indices[pos_indices] - 1]
                assigned_bboxes[pos_indices] = gt_bboxes[assigned_gt_indices[pos_indices] - 1]
            assigned_labels[neg_indices] = 0

        return assigned_labels, assigned_bboxes


if __name__ == "__main__":
    import numpy as np

    anchors = torch.tensor(np.array([[1.0, 1.0, 5.0, 5.0],
                                     [1.0, 1.0, 6.0, 6.0],
                                     [1.0, 1.0, 2.0, 2.0],
                                     [1.0, 1.0, 2.0, 2.0],
                                     [1.0, 1.0, 2.5, 2.5],
                                     [1.0, 1.0, 3.0, 3.0]]))
    gt_bboxes = torch.tensor(np.array([[1.0, 1.0, 6.0, 6.0],
                                       [3.0, 3.0, 4.0, 4.0]]))
    gt_labels = torch.tensor(np.array([1, 2, 3], dtype=np.int)).long()

    iou = iouNd_pytorch(anchors, gt_bboxes)
    print(iou)

    i = IoUAssigner(0.7, 0.3, min_pos_iou=0.3, match_low_quality=True, num_neg=2)
    assigned_gt_indices, assigned_labels = i.assign(anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    print(assigned_gt_indices)
    print(assigned_labels)

    # a = MaxIoUAssigner(0.7, 0.3, min_pos_iou=0.3, match_low_quality=True, num_neg=2)
    # assigned_labels, assigned_bboxes = a.assign(anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)
    # [print(i) for i in [assigned_labels, assigned_bboxes]]
