

import platform
from functools import partial
import numpy as np
import random
import torch
from torch.utils.data import DataLoader


if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def collate(batch_data_list):
    # a list of list
    import time

    _tic_ = time.time()

    if isinstance(batch_data_list[0], list):
        # assert all([len(i) == 1 and type(i[0]) == dict for i in batch_data_list]), 'All data should only contains one dict'
        # if not all([len(i) == 1 and type(i[0]) == dict for i in data_list]):  # 'All data should only contains one dict'
        #     assert len(data_list) == 1  # if data contains multi dict
        #     data_list = [[i] for i in data_list[0]]
        batch_data_list = [j for i in batch_data_list for j in i]

    try:
        batch = {}
        img = [s['img'] for s in batch_data_list]
        img = torch.stack(img, dim=0)
        batch['img'] = img

        if 'patches_img' in batch_data_list[0].keys():
            patches_img = [s['patches_img'] for s in batch_data_list]
            patches_img = torch.stack(patches_img, dim=0)
            batch['patches_img'] = patches_img
    except:
        [print(s['img_meta']) for s in batch_data_list]
        [print(s['img'].shape, s['img'].dtype) for s in batch_data_list]
        raise

    batch['img_meta'] = [s['img_meta'] for s in batch_data_list]

    if 'gt_bboxes' in batch_data_list[0].keys():
        batch['gt_bboxes'] = torch.stack([s['gt_bboxes'] for s in batch_data_list], dim=0)
        batch['gt_labels'] = torch.stack([s['gt_labels'] for s in batch_data_list], dim=0)
    # print("collater", batch['img_meta'])

    if 'gt_det' in batch_data_list[0].keys():
        dim = batch_data_list[0]['gt_det'].shape[1] // 2 - 1
        gt_bboxes = [s['gt_det'] for s in batch_data_list]

        max_num_labels = max(b.shape[0] for b in gt_bboxes)
        min_num_labels = min(b.shape[0] for b in gt_bboxes)
        # print(dim, min_num_labels, max_num_labels)

        # if max_num_labels > 0:
        gt_bboxes_pad = torch.ones((len(gt_bboxes), max_num_labels, 2 * dim + 2)) * -1

        for idx, b in enumerate(gt_bboxes):
            # print(b, b.shape[0], gt_labels[idx])
            try:
                if b.shape[0] > 0:
                    gt_bboxes_pad[idx, :b.shape[0], :] = b
            except Exception as e:
                # print(b)
                # print(gt_labels[idx])
                # print(gt_bboxes_pad)
                # print(gt_bboxes)
                # print(gt_labels)
                raise e

        batch['gt_det'] = gt_bboxes_pad

    for k in batch_data_list[0].keys():
        if k not in ['img', 'patches_img', 'gt_det', 'img_meta', 'gt_bboxes', 'gt_labels']:
            gt_cls = [s[k] for s in batch_data_list]
            gt_cls = torch.stack(gt_cls, dim=0)
            batch[k] = gt_cls

    # if 'gt_seg' in batch_data_list[0].keys():
    #     gt_seg = [s['gt_seg'] for s in batch_data_list]
    #     gt_seg = torch.stack(gt_seg, dim=0)
    #     batch['gt_seg'] = gt_seg
    # if 'gt_seg_skeleton' in batch_data_list[0].keys():
    #     gt_seg = [s['gt_seg_skeleton'] for s in batch_data_list]
    #     gt_seg = torch.stack(gt_seg, dim=0)
    #     batch['gt_seg_skeleton'] = gt_seg
    # if 'cutout_mask' in batch_data_list[0].keys():
    #     gt_seg = [s['cutout_mask'] for s in batch_data_list]
    #     gt_seg = torch.stack(gt_seg, dim=0)
    #     batch['cutout_mask'] = gt_seg
    # if 'gt_cls' in batch_data_list[0].keys():
    #     gt_cls = [s['gt_cls'] for s in batch_data_list]
    #     gt_cls = torch.stack(gt_cls, dim=0)
    #     batch['gt_cls'] = gt_cls
    # print(f"{time.time() - _tic_:.03}s")
    return batch


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """

    batch_size = num_gpus * imgs_per_gpu
    num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,  # to make different batch act different
        **kwargs)

    return data_loader


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0]
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    # print(f'init_seed {seed}, worker_id {worker_id} worker_seed {worker_seed}')