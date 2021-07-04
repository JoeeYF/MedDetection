

from scipy import ndimage as ndi
from skimage import measure, morphology
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from meddet.data import vtkShow


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    # print(vals)
    # print(counts)

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


class Lung:
    @staticmethod
    def segment_lung_mask(numpyImage, fill_lung_structures=True, debug=False):
        d, h, w = numpyImage.shape
        # if debug:
        #     vtkShow(numpyImage)

        # => outer/inner air = 1, body = 0
        numpyImage = ndi.gaussian_filter(numpyImage, 5)
        numpyImage = ndi.grey_closing(numpyImage, 4)  # some black voxel will cause failure
        binary_image = np.array(numpyImage < -320, dtype=np.float)
        binary_image[:2, ...] = 0
        binary_image[-2:, ...] = 0
        # binary_image = morphology.binary_closing(binary_image, np.ones([7, 7, 7]))
        # binary_image = morphology.binary_opening(binary_image, np.ones([2, 2, 2]))

        # Fill the air around the person
        # => inner air = 1, body/outer air = 0
        labels = measure.label(binary_image, connectivity=2)
        print(labels.shape, numpyImage.shape)
        # if debug:
        #     for s in range(0, numpyImage.shape[0], numpyImage.shape[0] // 20):
        #         plt.imshow(labels[s], vmin=0, vmax=10)
        #         plt.colorbar()
        #         plt.title(s)
        #         plt.show()

        tmp = labels.copy()
        tmp[:, 1:-1, 1:-1] = 0
        for background_label in np.unique(tmp):
            binary_image[background_label == labels] = 0
        origin_binary_image = np.copy(binary_image)
        # Method of filling the lung structures (that is superior to something like
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice + 1
                labeling = measure.label(axial_slice, connectivity=2)
                l_max = largest_label_volume(labeling, bg=0)

                if l_max is not None:  # This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        if debug:
            # vtkShow(binary_image)
            plt.subplot(2, 2, 1)
            plt.imshow(numpyImage[d // 2, :, :], 'gray')
            plt.title("input image")
            plt.subplot(2, 2, 2)
            plt.imshow(Normalize()(labels[d // 2, :, :]), 'jet')
            plt.colorbar()
            plt.title(f'labeled outer/inner air')
            plt.subplot(2, 2, 3)
            plt.imshow(origin_binary_image[d // 2, :, :], 'gray')
            plt.title("inner air")
            plt.subplot(2, 2, 4)
            plt.imshow(binary_image[d // 2, :, :], 'gray')
            plt.title("inner air filled")
            plt.show()

        # Remove other air pockets insided body
        # if debug:
        #     plt.imshow(Normalize()(labels[d // 2, :, :]), 'jet')
        #     plt.title(f'labeled in {np.max(labels.flatten())} classes after fill outer air')
        #     plt.show()
        # labels = measure.label(binary_image, background=0, connectivity=2)
        # l_max = largest_label_volume(labels, bg=0)
        # if l_max is not None:  # There are air pockets
        #     binary_image[labels != l_max] = 0
        # if debug:
        #     # vtkShow(binary_image)
        #     plt.imshow(binary_image[d // 2, :, :], 'gray')
        #     plt.show()
        return binary_image
