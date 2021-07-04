

# -*- coding:utf-8 -*-
import os
import numpy as np
from PIL import Image
from imageio import mimsave, imsave
from skimage import io
import matplotlib.pyplot as plt


def volume2gif(image: np.ndarray, file_path: str):
    """

    Args:
        image: ndim = 3, zyx
        file_path:

    Returns:

    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    images = []
    for i in range(len(image)):
        im = image[i, ...] * 255
        im = Image.fromarray(im.astype(np.uint8))
        images.append(im)
    mimsave(file_path, images)


def volume2tiled(image: np.ndarray, file_path: str, sampling_ratio=10, col=None):
    """

    Args:
        image: ndim = 3, zyx
        file_path:
        sampling_ratio:
        col:

    Returns:

    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image[::sampling_ratio, ...]
    z = image.shape[0]
    if not col:
        col = int(np.ceil(np.sqrt(z)))
    row = int(np.ceil(z / col))
    new_image = np.zeros((row * col, image.shape[1], image.shape[2]))
    new_image[:z] = image
    new_image = np.concatenate([np.concatenate(i, axis=1) for i in np.split(new_image, row, axis=0)], axis=0)
    io.imsave(file_path, new_image.astype(np.uint8), check_contrast=False)