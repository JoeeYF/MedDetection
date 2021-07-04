

# -*- coding:utf-8 -*-
from scipy.ndimage.filters import uniform_filter


def window_stdev(arr, window):
    c1 = uniform_filter(arr, window, mode='nearest', origin=0)
    c2 = uniform_filter(arr * arr, window, mode='nearest', origin=0)
    print(arr.shape, c1.shape)
    return c1, ((c2 - c1 * c1) ** .5)
