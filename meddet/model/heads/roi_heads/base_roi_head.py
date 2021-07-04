 
#

from typing import Union, List
from torch import nn
import torch
import numpy as np
import math
import warnings

from meddet.model.nnModules import ComponentModule

from meddet.task.builder import build_loss, build_metric
from meddet.model.heads.utils import build_extractor
from meddet.task.builder import build_head

from meddet.model.nnModules import ComponentModule

class ROIHead(ComponentModule):
    def __init__(self,
                 dim=None,
                 bbox_roi_extractor=None,
                 bbox_head=None):
        super(ROIHead, self).__init__()

        # bbox_roi_extractor.update({'dim': dim})
        self.bbox_roi_extractor = build_extractor(bbox_roi_extractor)

        self.bbox_head = build_head(bbox_head)
        self.bbox_head.set_extractor(self.bbox_roi_extractor)

    def forward_train(self,
                      multi_level_features,
                      batch_proposals,
                      batch_gt_labels,
                      batch_gt_bboxes):
        return self.bbox_head.forward_train(multi_level_features,
                                            batch_proposals,
                                            batch_gt_labels,
                                            batch_gt_bboxes)

    def forward_valid(self,
                      multi_level_features,
                      batch_proposals,
                      batch_gt_labels,
                      batch_gt_bboxes):
        return self.bbox_head.forward_valid(multi_level_features,
                                            batch_proposals,
                                            batch_gt_labels,
                                            batch_gt_bboxes)

    def forward_infer(self,
                      multi_level_features,
                      batch_proposals):
        return self.bbox_head.forward_infer(multi_level_features,
                                            batch_proposals)

    def metric(self, results, ground_truth):
        return self.bbox_head.metric(results, ground_truth)
