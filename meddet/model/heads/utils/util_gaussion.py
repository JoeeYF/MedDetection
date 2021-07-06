from math import sqrt

import torch
import torch.nn.functional as F


def gaussian3D(radius, sigma=1, dtype=torch.float32, device='cpu'):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    x = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(-1, 1, 1)
    y = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, -1, 1)
    z = torch.arange(
        -radius, radius + 1, dtype=dtype, device=device).view(1, 1, -1)
    h = (-(x * x + y * y + z * z) / (2 * sigma * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (int): Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter = 2 * radius + 1
    gaussian_kernel = gaussian3D(
        radius, sigma=diameter / 6, dtype=heatmap.dtype, device=heatmap.device)

    x, y, z = center

    depth, height, width = heatmap.shape[:3]

    xmin, xmax = min(x, radius), min(width - x, radius + 1)
    ymin, ymax = min(y, radius), min(height - y, radius + 1)
    zmin, zmax = min(z, radius), min(depth - z, radius + 1)

    masked_heatmap = heatmap[z - zmin:z + zmax, y - ymin:y + ymax, x - xmin:x + xmax]
    masked_gaussian = gaussian_kernel[radius - zmin:radius + zmax,
                      radius - ymin:radius + ymax,
                      radius - zmin:radius + zmax]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[z - zmin:z + zmax, y - ymin:y + ymax, x - xmin:x + xmax])

    return out_heatmap


def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernal.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool3d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, depth, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, depth, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (depth * height * width)
    topk_inds = topk_inds % (depth * height * width)
    topk_zs = topk_inds // (height * width)
    topk_ys = topk_inds % (height * width) // width
    topk_xs = (topk_inds % (height * width) % width).int().float()
    return topk_scores, topk_inds, topk_clses+1, topk_zs, topk_ys, topk_xs

def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 4, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(4))
    feat = gather_feat(feat, ind)
    return feat

if __name__ == '__init__':
    a = torch.zeros(96, 96, 96).float()
    gen_gaussian_target(a, [34, 53, 23], 5)
    import SimpleITK as sitk

    sitk.WriteImage(sitk.GetImageFromArray(a.numpy()), 'temp.nii')
