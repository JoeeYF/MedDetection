 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch
from abc import ABCMeta, abstractmethod


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_ub (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_random_gt (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """
    def __init__(self,
                 pos_fraction,
                 num=-1,
                 neg_pos_ub=-1,
                 add_random_gt=-1,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_random_gt = add_random_gt
        self.pos_sampler = self
        self.neg_sampler = self

        assert isinstance(add_random_gt, int), 'add_random_gt must be int'

    @abstractmethod
    def _sample_pos(self, assigned_labels, num_expected, **kwargs):
        """Sample positive samples.
        Args:
            assigned_labels : Assigned labels
            num_expected (int): Number of expected positive samples
        Returns:
            torch.Tensor: Indices  of positive samples
        """
        pass

    @abstractmethod
    def _sample_neg(self, assigned_labels, num_expected, **kwargs):
        """Sample negative samples.
        Args:
            assigned_labels : Assigned labels
            num_expected (int): Number of expected negative samples
        Returns:
            torch.Tensor: Indices  of negative samples
        """
        pass

    def sample(self,
               bboxes,
               assigned_labels,
               assigned_bboxes,
               gt_labels,
               gt_bboxes,
               infer=False,
               weight=None):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            bboxes (Tensor): Boxes to be sampled from.
            assigned_labels (:obj:`AssignResult`): Bbox assigning results.
            assigned_bboxes (:obj:`AssignResult`): Bbox assigning results.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.
            infer
            weight

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        dim = assigned_bboxes.shape[-1] // 2
        assert dim in (2, 3)
        assert weight is not None

        if self.add_random_gt >= 0 and not infer and gt_labels is not None:
            random_gt_bboxes = []
            for i in range(self.add_random_gt):
                bbox_shape = gt_bboxes[..., dim:] - gt_bboxes[..., :dim]
                shift = torch.cat([bbox_shape * (torch.rand_like(bbox_shape) - 0.5) / 10,
                                   bbox_shape * (torch.rand_like(bbox_shape) - 0.5) / 10], dim=-1)
                random_gt_bboxes.append(gt_bboxes + shift)
            bboxes = torch.cat([bboxes, gt_bboxes] + random_gt_bboxes, dim=0)
            assigned_bboxes = torch.cat([assigned_bboxes, gt_bboxes] + [gt_bboxes] * self.add_random_gt, dim=0)
            assigned_labels = torch.cat([assigned_labels, gt_labels] + [gt_labels] * self.add_random_gt, dim=0)
            if weight is not None:
                weight = torch.cat([weight, 100 * torch.ones_like(gt_labels)] + [100 * torch.ones_like(gt_labels)] * self.add_random_gt, dim=0)
        # assert len(weight) == len(assigned_labels)
        # print(torch.max(weight))

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_indices = self.pos_sampler._sample_pos(assigned_labels, num_expected_pos, weight=weight)

        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_indices = pos_indices.unique()
        num_sampled_pos = pos_indices.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_indices = self.neg_sampler._sample_neg(assigned_labels, num_expected_neg, weight=weight)
        neg_indices = neg_indices.unique()

        return pos_indices, neg_indices, bboxes, assigned_labels, assigned_bboxes


class RandomSampler(BaseSampler):
    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.
        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            sampled indices.
        """
        assert len(gallery) >= num
        assert isinstance(gallery, torch.Tensor)

        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_indices = gallery[perm]
        return rand_indices

    def _sample_pos(self, assigned_labels, num_expected, **kwargs):
        pos_indices = torch.nonzero(assigned_labels > 0, as_tuple=False)
        if pos_indices.numel() != 0:
            pos_indices = pos_indices.squeeze(1)
        if pos_indices.numel() <= num_expected:
            return pos_indices
        else:
            return self.random_choice(pos_indices, num_expected)

    def _sample_neg(self, assigned_labels, num_expected, **kwargs):
        neg_indices = torch.nonzero(assigned_labels == 0, as_tuple=False)
        if neg_indices.numel() != 0:
            neg_indices = neg_indices.squeeze(1)
        if len(neg_indices) <= num_expected:
            return neg_indices
        else:
            return self.random_choice(neg_indices, num_expected)


class OHEMSampler(BaseSampler):
    r"""Online Hard Example Mining Sampler described in `Training Region-based
    Object Detectors with Online Hard Example Mining
    <https://arxiv.org/abs/1604.03540>`_.
    """

    def __init__(self, *args, **kwargs):
        super(OHEMSampler, self).__init__(*args, **kwargs)

    @staticmethod
    def hard_mining(gallery, num, weight):
        """

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.
            weight: loss

        Returns:
            sampled indices.
        """
        _, indices = torch.topk(weight, num, dim=0)
        return gallery[indices]

    def _sample_pos(self,
                    assigned_labels,
                    num_expected,
                    weight=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_indices = torch.nonzero(assigned_labels > 0, as_tuple=False)
        if pos_indices.numel() != 0:
            pos_indices = pos_indices.squeeze(1)
        if pos_indices.numel() <= num_expected:
            return pos_indices
        else:
            # print('hard mining pos')
            # print(pos_indices)
            sampled = self.hard_mining(pos_indices, num_expected,
                                       weight[pos_indices])
            # print(sampled)
            return sampled

    def _sample_neg(self,
                    assigned_labels,
                    num_expected,
                    weight=None,
                    **kwargs):
        # Sample some hard negative samples
        neg_indices = torch.nonzero(assigned_labels == 0, as_tuple=False)
        if neg_indices.numel() != 0:
            neg_indices = neg_indices.squeeze(1)
        if len(neg_indices) <= num_expected:
            return neg_indices
        else:
            # print('hard mining neg')
            # print(neg_indices)
            sampled = self.hard_mining(neg_indices, num_expected,
                                       weight[neg_indices])
            # print(sampled)
            return sampled


class HNEMSampler(BaseSampler):
    r"""Hard Negative Example Mining Sampler
    """

    def __init__(self, *args, **kwargs):
        super(HNEMSampler, self).__init__(*args, **kwargs)

    @staticmethod
    def hard_mining(gallery, num, weight):
        """

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.
            weight: loss

        Returns:
            sampled indices.
        """
        _, indices = torch.topk(weight, num, dim=0)
        return gallery[indices]

    def _sample_pos(self,
                    assigned_labels,
                    num_expected,
                    weight=None,
                    **kwargs):
        # Sample some hard positive samples
        pos_indices = torch.nonzero(assigned_labels > 0, as_tuple=False)
        return pos_indices

    def _sample_neg(self,
                    assigned_labels,
                    num_expected,
                    weight=None,
                    **kwargs):
        # Sample some hard negative samples
        neg_indices = torch.nonzero(assigned_labels == 0, as_tuple=False)
        if neg_indices.numel() != 0:
            neg_indices = neg_indices.squeeze(1)
        if len(neg_indices) <= num_expected:
            return neg_indices
        else:
            # print('hard mining neg')
            # print(neg_indices)
            sampled = self.hard_mining(neg_indices, num_expected,
                                       weight[neg_indices])
            # print(sampled)
            return sampled

    def sample(self,
               bboxes,
               assigned_labels,
               assigned_bboxes,
               gt_labels,
               gt_bboxes,
               infer=False,
               weight=None):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            bboxes (Tensor): Boxes to be sampled from.
            assigned_labels (:obj:`AssignResult`): Bbox assigning results.
            assigned_bboxes (:obj:`AssignResult`): Bbox assigning results.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.
            infer
            weight

        Returns:
            :obj:`SamplingResult`: Sampling result.
        """
        dim = assigned_bboxes.shape[-1] // 2
        assert dim in (2, 3)
        assert weight is not None

        if self.add_random_gt >= 0 and not infer and gt_labels is not None:
            random_gt_bboxes = []
            for i in range(self.add_random_gt):
                bbox_shape = gt_bboxes[..., dim:] - gt_bboxes[..., :dim]
                shift = torch.cat([bbox_shape * (torch.rand_like(bbox_shape) - 0.5) / 10,
                                   bbox_shape * (torch.rand_like(bbox_shape) - 0.5) / 10], dim=-1)
                random_gt_bboxes.append(gt_bboxes + shift)
            bboxes = torch.cat([bboxes, gt_bboxes] + random_gt_bboxes, dim=0)
            assigned_bboxes = torch.cat([assigned_bboxes, gt_bboxes] + [gt_bboxes] * self.add_random_gt, dim=0)
            assigned_labels = torch.cat([assigned_labels, gt_labels] + [gt_labels] * self.add_random_gt, dim=0)
            if weight is not None:
                weight = torch.cat([weight, 100 * torch.ones_like(gt_labels)] + [100 * torch.ones_like(gt_labels)] * self.add_random_gt, dim=0)
        # assert len(weight) == len(assigned_labels)
        # print(torch.max(weight))

        pos_indices = self.pos_sampler._sample_pos(assigned_labels, -1, weight=weight)

        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_indices = pos_indices.unique()
        num_sampled_pos = pos_indices.numel()
        num_expected_neg = int(num_sampled_pos * (1 - self.pos_fraction) / self.pos_fraction)
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_indices = self.neg_sampler._sample_neg(assigned_labels, num_expected_neg, weight=weight)
        neg_indices = neg_indices.unique()

        return pos_indices, neg_indices, bboxes, assigned_labels, assigned_bboxes