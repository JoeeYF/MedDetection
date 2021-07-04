from collections.abc import Sequence
from hashlib import new
from typing import List

import numpy as np
import torch
import time

from ..registry import PIPELINES
from .aug_base import Stage


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    elif isinstance(data, List):
        new_data = []
        for item in data:
            new_data.append(to_tensor(item))
        return new_data
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def to_numpy(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data.numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return np.append(data)
    elif isinstance(data, torch.LongTensor):
        return data.numpy()
    # elif isinstance(data, float):
    #     return torch.FloatTensor([data])
    elif isinstance(data, List):
        new_data = []
        for item in data:
            new_data.append(to_numpy(item))
        return new_data
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))

# @PIPELINES.register_module
# class ToTensor(object):
#
#     def __init__(self, keys):
#         self.keys = keys
#
#     def __call__(self, results):
#         for key in self.keys:
#             results[key] = to_tensor(results[key])
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={})'.format(self.keys)


# @PIPELINES.register_module
# class ImageToTensor(object):
#
#     def __init__(self, keys):
#         self.keys = keys
#
#     def __call__(self, results):
#         for key in self.keys:
#             results[key] = to_tensor(results[key].transpose(2, 0, 1))
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={})'.format(self.keys)


# @PIPELINES.register_module
# class Transpose(object):
#
#     def __init__(self, keys, order):
#         self.keys = keys
#         self.order = order
#
#     def __call__(self, results):
#         for key in self.keys:
#             results[key] = results[key].transpose(self.order)
#         return results
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(keys={}, order={})'.format(
#             self.keys, self.order)


# @PIPELINES.register_module
# class DefaultFormatBundle(Stage):
#     """Default formatting bundle.
#
#     It simplifies the pipeline of formatting common fields, including "img",
#     "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
#     These fields are formatted as follows.
#
#     - img:       (1)transpose, (2)to tensor
#     - gt_seg:    (1)transpose, (2)to tensor
#     """
#     def __repr__(self):
#         return self.__class__.__name__
#
#     @property
#     def canBackward(self):
#         return True
#
#     def __call__(self, results, forward=True):
#         if isinstance(results, dict):
#             results = [results]
#
#         if forward:
#             return [self.forward(r.copy()) for r in results]
#         else:
#             return [self.backward(r.copy()) for r in results]
#
#     def forward(self, results):
#         # for key in results['seg_fields']:
#         #     img = results[key]
#         #     results[key] = to_tensor(img)  # DC(to_tensor(img), stack=True)
#         # for key in results['img_fields']:
#         #     img = results[key]
#         #     results[key] = to_tensor(img)  # DC(to_tensor(img), stack=True)
#         # if 'patches_img' in results.keys():
#         #     patches = []
#         #     for patch in results['patches_img']:
#         #         patch = to_tensor(patch)
#         #         patches.append(patch)
#         #     patches = torch.stack(patches, dim=0)
#         #     results['patches_img'] = patches
#         for key in ['gt_det', 'gt_cls']:
#             if key not in results:
#                 continue
#             results[key] = to_tensor(results[key])  # DC(to_tensor(results[key]))
#         results['history'].append(self.name)
#         return results
#
#     def backward(self, results):
#         # for key in results['seg_fields']:
#         #     results[key] = to_numpy(results[key])
#         # for key in results['img_fields']:
#         #     results[key] = to_numpy(results[key])
#         # if 'patches_img' in results.keys():
#         #     patches = []
#         #     for patch in results['patches_img']:
#         #         patch = to_numpy(patch)
#         #         patches.append(patch)
#         #     patches = np.stack(patches, axis=0)
#         #     results['patches_img'] = patches
#         for key in ['proposals', 'gt_det', 'gt_cls']:
#             if key not in results:
#                 continue
#             results[key] = to_numpy(results[key])  # DC(to_tensor(results[key]))
#         last = results['history'].pop()
#         assert last == self.name
#         return results


@PIPELINES.register_module
class Collect(Stage):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)

    @property
    def canBackward(self):
        return True

    def __call__(self, results, forward=True):
        if isinstance(results, dict):
            results = [results]

        if forward:
            return [self._forward(r.copy()) for r in results]
        else:
            return [self.backward(r.copy()) for r in results]

    def _forward(self, results):
        _tic_ = time.time()
        data = {}
        img_meta = {}
        for key in self.keys:
            data[key] = to_tensor(results.pop(key))
        for key in ['patches_img']:
            if key in results.keys():
                data[key] = to_tensor(results.pop(key))
        for key in results.keys():
            img_meta[key] = results[key]
        data['img_meta'] = img_meta
        data['img_meta']['history'].append(self.name)
        data['img_meta']['time'].append(f'{self.name}-{time.time() - _tic_:.03f}s')

        return data

    def backward(self, data):
        results = {}
        for key in self.keys:
            results[key] = to_numpy(data[key])
        for key in ['patches_img']:
            if key in data.keys():
                results[key] = to_numpy(data[key])
        for k, v in data['img_meta'].items():
            results[k] = v
        last = results['history'].pop()
        assert last == self.name
        return results
