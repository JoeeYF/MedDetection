

import numpy as np


def getSphere(dim, size, diameter):
    radius = diameter / 2 - 0.5
    structure = np.zeros((size,) * dim)

    center = [i / 2 for i in structure.shape]
    ctr = np.meshgrid(*[np.arange(0.5, size)]*dim, indexing='ij')
    ctr = np.stack(ctr, axis=0)
    ctr = np.transpose(ctr, [*range(1, dim + 1), 0])

    distance = np.sum(np.power(np.abs(ctr - center), 2), axis=-1)
    structure = (distance <= radius ** 2).astype(np.float32)
    return structure