 

import torch

from medvision.ops.torch import deltaNd_pytorch, applyDeltaNd_pytorch


class DeltaBBoxCoder(object):
    """
    References: mmdet
    Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 dim,
                 target_means=None,
                 target_stds=None):
        super().__init__()
        self.dim = dim
        if target_means is None:
            target_means = [.0] * 2 * dim
        if target_stds is None:
            target_stds = [.1] * dim + [.2] * dim
        if isinstance(target_means, (list, tuple)):
            target_means = torch.tensor(target_means)
        if isinstance(target_stds, (list, tuple)):
            target_stds = torch.tensor(target_stds)

        self.means = target_means
        self.stds = target_stds

    def encode(self,
               bboxes: torch.Tensor,
               gt_bboxes: torch.Tensor):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] // 2 == gt_bboxes.shape[-1] // 2 == self.dim
        bboxes = bboxes[..., :2 * self.dim]
        gt_bboxes = gt_bboxes[..., :2 * self.dim]
        encoded_bboxes = deltaNd_pytorch(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes

    def decode(self, bboxes, deltas):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes.
            deltas (torch.Tensor): Encoded boxes with shape

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert deltas.shape[0] == bboxes.shape[0], f"{deltas.shape}, {bboxes.shape}"
        bboxes = bboxes[..., :2 * self.dim]
        deltas = deltas[..., :2 * self.dim]
        decoded_bboxes = applyDeltaNd_pytorch(bboxes, deltas, self.means, self.stds)

        return decoded_bboxes


if __name__ == "__main__":
    import numpy as np

    anchors = np.array([[1.0, 1.0, 5, 5],
                        [1.0, 1.0, 6, 6],
                        [1.0, 1.0, 2, 2],
                        [1.0, 1.0, 3, 3]])
    targets = np.array([[1.0, 1, 4, 4],
                        [1.0, 1, 4, 4],
                        [1.0, 1, 4, 4],
                        [1.0, 1, 4, 4]])

    d = DeltaBBoxCoder(dim=2, target_means=[0, 0, 0, 0], target_stds=[1., 1., 1., 1.])
    deltas = d.encode(torch.tensor(anchors), torch.tensor(targets))
    box = d.decode(torch.tensor(anchors), deltas)
    print(deltas)
    print(box)
