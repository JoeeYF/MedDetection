 

# -*- coding:utf-8 -*-
from meddet.model.nnModules import ComponentModule


class BaseDenseHead(ComponentModule):
    """Base class for DenseHeads."""

    def __init__(self):
        super(BaseDenseHead, self).__init__()

    def get_losses(self, **kwargs):
        """Compute losses of the head."""
        pass

    def get_bboxes(self, *args, **kwargs):
        """Transform network output for a batch into bbox predictions."""
        pass

    def get_proposals(self, *args, **kwargs):
        pass

    def forward_train(self,
                      multi_level_cls_out: (list, tuple),
                      multi_level_reg_out: (list, tuple),
                      batch_gt_labels,
                      batch_gt_bboxes) -> tuple:
        """

        Args:
            multi_level_cls_out: [(B, Classes, **shape1), (B, Classes, **shape2), ..]
            multi_level_reg_out: [(B, Coords, **shape1), (B, Coords, **shape2), ..]
            batch_gt_labels: (B, N, Coords)
            batch_gt_bboxes: (B, N, Classes)

        Returns:
            if train:
                return losses, None
            if valid:
                return losses, bboxes
            if infer:
                return None, bboxes

        """
        raise NotImplementedError

    def forward_valid(self,
                      multi_level_cls_out: (list, tuple),
                      multi_level_reg_out: (list, tuple),
                      batch_gt_labels,
                      batch_gt_bboxes) -> tuple:
        raise NotImplementedError

    def forward_infer(self,
                      multi_level_cls_out: (list, tuple),
                      multi_level_reg_out: (list, tuple)) -> tuple:
        raise NotImplementedError