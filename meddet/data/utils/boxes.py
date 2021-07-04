

from typing import Union
import numpy as np
def nmsNd_numpy(dets: np.ndarray, threshold: float):
    """
    :param dets:  [[x1,y1,x2,y2,score],  |  [[x1,y1,z1,x2,y2,z2,score],
                   [x1,y1,x2,y2,score]]  |   [x1,y1,z1,x2,y2,z2,score]]
    :param threshold: for example 0.5
    :return: the rest ids of dets
    """
    dim = dets.shape[-1] // 2
    assert dim in (2, 3), dets.shape

    scores = dets[:, -1].copy()
    bboxes = dets[:, :-1].copy()
    assert bboxes.shape[-1] == 2 * dim

    area = np.prod(bboxes[:, dim:] - bboxes[:, :dim] + 1, axis=-1)
    # print(area)

    order = scores.argsort()[::-1]

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)

        overlap = np.minimum(bboxes[i, dim:], bboxes[order[1:]][:, dim:])
        overlap = overlap - np.maximum(bboxes[i, :dim], bboxes[order[1:]][:, :dim]) + 1
        overlap = np.maximum(overlap, 0)
        inter = np.prod(overlap, axis=-1)
        # print(inter)

        union = area[i] + area[order[1:]] - inter
        iou = inter / union
        # print(iou)

        index = np.where(iou <= threshold)[0]
        # print(index)

        # similar to soft nmsNd_cuda
        # weight = np.exp(-(iou * iou) / 0.5)
        # scores[order[1:]] = weight * scores[order[1:]]

        order = order[index + 1]

    dets = np.concatenate((bboxes, scores[:, None]), axis=1)
    keep = np.array(keep)
    return keep, dets

def iouNd_numpy(anchors, targets, dim=None):
    """
    :param anchors:  [N, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param targets:  [M, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param dim: dimension of bbox
    :return:   IoU:  [N,M]
    """

    if not dim:
        dim = targets.shape[-1] // 2
        assert dim in (2, 3)

    anchors = anchors[..., :2 * dim]
    targets = targets[..., :2 * dim]

    # expand dim
    anchors = np.expand_dims(anchors, axis=1)  # [N, 1, 2*dim]
    targets = np.expand_dims(targets, axis=0)  # [1, M, 2*dim]

    # overlap on each dim
    overlap = np.minimum(anchors[..., dim:], targets[..., dim:])
    overlap = overlap - np.maximum(anchors[..., :dim], targets[..., :dim])
    overlap = np.maximum(0.0, overlap)  # [N,M,dim]

    # intersection
    intersection = np.prod(overlap, axis=-1).astype(np.float)  # [N,M]

    # areas
    area_a = np.prod(anchors[..., dim:] - anchors[..., :dim], axis=-1).astype(np.float)  # [N,1]
    area_b = np.prod(targets[..., dim:] - targets[..., :dim], axis=-1).astype(np.float)  # [1,M]

    # iou
    iou = intersection / (area_a + area_b - intersection)
    return iou


def deltaNd_numpy(anchors: np.ndarray,
                  targets: np.ndarray,
                  means: np.ndarray = None,
                  stds: np.ndarray = None):
    """
    :param  anchors:  [N_pos, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param  targets:  [N_pos, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param  means:
    :param  stds:
    :return: deltas:  [N_pos, (dx,dy,dw,dh) | (dx,dy,dz,dw,dh,dd)]
    """

    dim = anchors.shape[-1] // 2
    assert dim in (2, 3)
    assert len(anchors) == len(targets)
    if means is None:
        means = np.array([[0.0] * 2 * dim])
    if stds is None:
        stds = np.array([[0.1] * dim + [0.2] * dim])

    # shape of anchor and target
    anchor_shape = anchors[..., dim:] - anchors[..., :dim]
    target_shape = targets[..., dim:] - targets[..., :dim]
    # print("anchor_shape\n", anchor_shape)
    # print("target_shape\n", target_shape)

    # center of anchor and target
    anchor_center = 0.5 * (anchors[..., dim:] + anchors[..., :dim])
    target_center = 0.5 * (targets[..., dim:] + targets[..., :dim])
    # print("anchor_center\n", anchor_center)
    # print("target_center\n", target_center)

    # delta of shape and center
    delta_center = (target_center - anchor_center) / anchor_shape
    delta_shape = np.log(target_shape / anchor_shape)
    # print("delta_center\n", delta_center)
    # print("delta_shape\n", delta_shape)

    deltas = np.hstack([delta_center, delta_shape])
    # print(regression.shape)

    deltas = (deltas - means) / stds
    return deltas


def applyDeltaNd_numpy(anchors: np.ndarray,
                       deltas: np.ndarray,
                       means: np.ndarray = None,
                       stds: np.ndarray = None):
    """
    应用回归目标到边框,用rpn网络预测的delta refine anchor
    :param deltas:  [N_pos, (dx,dy,dw,dh) | (dx,dy,dz,dw,dh,dd)]
    :param anchors: [N_pos, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    :param  means:
    :param  stds:
    :return:        [N_pos, (x1,y1,x2,y2) | (x1,y1,z1,x2,y2,z2)]
    """
    dim = anchors.shape[-1] // 2
    assert dim in (2, 3)
    assert len(anchors) == len(deltas)
    if means is None:
        means = np.array([[0.0] * 2 * dim])
    if stds is None:
        stds = np.array([[0.1] * dim + [0.2] * dim])

    deltas = deltas * stds + means

    anchor_shape = anchors[..., dim:] - anchors[..., :dim]
    # print(anchor_shape)

    anchor_center = 0.5 * (anchors[..., dim:] + anchors[..., :dim])
    # print(anchor_center)

    anchor_center = anchor_center + deltas[..., :dim] * anchor_shape
    # print(anchor_center)

    anchor_shape = anchor_shape * np.exp(deltas[..., dim:])
    # print(anchor_shape)

    anchors_refined = np.hstack([anchor_center - 0.5 * anchor_shape,
                                 anchor_center + 0.5 * anchor_shape])

    return anchors_refined


def clipBBoxes(dim: int,
               bboxes: np.ndarray,
               image_shape: list) -> np.ndarray:
    """
    Args:
        dim: dimension of image
        bboxes:
            shape is [N, 2*dim], [N, 2*dim + 1 or 2], xyz order
            sometimes, it will be used to handle boxes with 'class' and 'score'
        image_shape: [d,] h, w; zyx order, reversed order of bboxes coords

    Returns:
        clipped bboxes
    """
    if bboxes.size == 0:
        return bboxes
    dim = len(image_shape)
    assert dim <= bboxes.shape[1] // 2, f"image is {dim}D, but bboxes is {bboxes.shape}"

    bboxes[:, :2 * dim] = np.maximum(bboxes[:, :2 * dim], 0)
    bboxes[:, :2 * dim] = np.minimum(bboxes[:, :2 * dim], np.array(image_shape[::-1] * 2) - 1)
    bboxes = bboxes[np.all(bboxes[:, dim:2 * dim] > bboxes[:, :dim], axis=1)]
    return bboxes


def cropBBoxes(dim: int,
               bboxes: np.ndarray,
               start_coord: Union[list, tuple, np.ndarray],
               end_coord: Union[list, tuple, np.ndarray],
               dim_iou_thr: float = 0) -> np.ndarray:
    """
    Args:
        dim: image dimension
        bboxes: [N, >=2 * dim], xyz order
        start_coord: shape is [dim, ], xyz order, e.g. [0, 0, 0]
        end_coord: shape is [dim, ], xyz order, e.g. [96, 96, 96]
        dim_iou_thr:

    Returns:
        cropped bboxes

    """
    start_coord = np.array(start_coord)
    end_coord = np.array(end_coord)
    patch_shape = (end_coord - start_coord).tolist()[::-1]  # zyx order
    assert start_coord.shape[-1] == end_coord.shape[-1] == dim

    cropped_bboxes = bboxes.copy()
    cropped_bboxes[:, :2 * dim] = bboxes[:, :2 * dim] - np.tile(start_coord, 2)
    cropped_bboxes = clipBBoxes(dim, cropped_bboxes, patch_shape)
    assert all(np.all(cropped_bboxes[:, dim:2 * dim] > cropped_bboxes[:, :dim], axis=1))

    padded_bboxes = padBBoxes(dim, cropped_bboxes, start_coord, end_coord)
    iou = iouNd_numpy(padded_bboxes[:, :2*dim], bboxes[:, :2*dim])
    iou_per_cropped_bbox = np.max(iou, axis=1)
    cropped_bboxes = cropped_bboxes[iou_per_cropped_bbox > dim_iou_thr ** dim]
    return cropped_bboxes


def padBBoxes(dim: int,
              bboxes: np.ndarray,
              start_coord: Union[list, tuple, np.ndarray],
              end_coord: Union[list, tuple, np.ndarray]) -> np.ndarray:
    """
    Args:
        dim: image dimension
        bboxes: [N, >=2 * dim], xyz order
        start_coord: shape is [dim, ], xyz order, e.g. [0, 0, 0]
        end_coord: shape is [dim, ], xyz order, e.g. [96, 96, 96]

    Returns:
        cropped bboxes

    """
    start_coord = np.array(start_coord)
    end_coord = np.array(end_coord)
    assert start_coord.shape[-1] == end_coord.shape[-1] == dim

    padded_bboxes = bboxes.copy()
    padded_bboxes[:, :2 * dim] = padded_bboxes[:, :2 * dim] + np.tile(start_coord, 2)
    return padded_bboxes


def objs2bboxes(objs: list) -> np.ndarray:
    """convert found objs to bboxes format

    Args:
        objs: list of slice ( without convert to int)

    Returns:
        bboxes: np.array

    """
    bboxes = []
    for obj in objs:
        one_det = [i.start for i in obj][::-1] + [i.stop for i in obj][::-1]
        bboxes.append(one_det)
    bboxes = np.array(bboxes)
    return bboxes


def bboxes2objs(bboxes: np.ndarray) -> list:
    """convert bboxes format to objs format
    
    Args:
        bboxes: [N, 2 * dim]

    Returns:
        objs: list of slice

    """
    dim = bboxes.shape[-1] // 2
    assert dim in (2, 3), bboxes.shape
    assert bboxes.ndim == 2
    objs = []
    for one_det in bboxes:
        obj = [slice(one_det[i], one_det[dim + i]) for i in range(dim)]
        obj = obj[::-1]
        objs.append(obj)
    return objs


if __name__ == "__main__":
    start = [0, 0, 0]  # zyx order
    end = [100, 96, 96]  # zyx order

    bboxes = [
        # x1,y1,z1,x2,y2,z2, class, score
        [22, 54, 35, 177, 199, 164, 1, 1.00],
        [32, 54, 45, 67, 87, 99, 2, 1.00],
        [67, 87, 89, 77, 97, 99, 1, 1.00],
        [122, 154, 135, 177, 199, 164, 3, 1.00],
    ]

    # cropped_bboxes = cropBBoxes(3, np.array(bboxes), start[::-1], end[::-1], dim_iou_thr=0.7)
    # print(cropped_bboxes)

    print(np.array(bboxes))

    objs = bboxes2objs(np.array(bboxes)[:, :6])
    print(objs)

    det = objs2bboxes(objs)
    print(det)