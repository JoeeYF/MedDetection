

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


def pseudo2det(pseudo_mask):
    labeled = ndi.label(pseudo_mask.astype(np.int))[0]
    plt.imshow(labeled)
    plt.show()
    objs = ndi.find_objects(labeled)
    print(objs)
    return det


def det2pseudo(det, image_shape):
    dim = det.shape[-1] // 2
    assert dim in (2, 3)
    assert det.shape[0] <= 10, 'Only used for sparse detection!'
    area = np.prod(det[..., dim:2 * dim] - det[..., :dim], axis=-1).astype(np.float)
    order = np.argsort(area)[::-1]

    det = det[order]

    pseudo_mask = []
    for i, ann in enumerate(det):
        slices = tuple(map(slice, reversed(np.int32(ann[:dim])), reversed(np.int32(ann[dim:2 * dim]))))
        one_pseudo = np.zeros(image_shape)
        one_pseudo[slices] = ann[-1]
        pseudo_mask.append(one_pseudo)
    pseudo_mask = np.stack(pseudo_mask, axis=-1)
    print(pseudo_mask.shape)

    # dim = pseudo_mask.ndim - 1
    # for ann in det:
    #     slices = tuple(map(slice, reversed(np.int32(ann[:dim])), reversed(np.int32(ann[dim:2 * dim]))))
    #     if np.max(pseudo_mask[slices]) == 0:
    #         pseudo_mask[slices] = ann[-1]
    #     else:
    #         to_fill = pseudo_mask[slices] == 0
    #         not_to_fill = pseudo_mask[slices] != 0
    #         num_features = ndi.label(to_fill)[1]
    #         for i in range(dim):
    #             np.min(to_fill, axis=1)
    #         if num_features == 1 and

    return pseudo_mask


if __name__ == "__main__":
    det = np.array([[1, 2, 3, 9, 1],
                    [2, 4, 8, 7, 2]])
    pseudo_mask = det2pseudo(det, image_shape=(15, 15))
    print(pseudo_mask.shape)
    # plt.imshow(pseudo_mask[..., 0])
    # plt.grid()
    # plt.show()
    #
    # pseudo2det(pseudo_mask)