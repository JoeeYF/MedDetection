

import numpy as np
import os

from .BasicDataset import BasicDetPairDataset
from ..registry import DATASETS
from ...utils import is_list_of


@DATASETS.register_module
class LungDetPairDataset(BasicDetPairDataset):
    dataset_format = {
        "name": "Luna16",
        "description": "Luna16",
        "categories": [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "target1"}
        ],
        "pairs": [
            {"image": "images/1.png", "label": [{'bbox': ['x', 'y', 'w', 'h'], 'category_id': 1}]},  # 2D Detection
            {"image": "images/1.nii", "label": [{'bbox': ['x', 'y', 'z', 'w', 'h', 'd'], 'category_id': 2}]},  # 3D Detection
            {"image": "images/1.png"}  # Inference
        ]
    }

    def __init__(self, phase='train', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = phase
        assert self.task == 'DET'
        if not self.infer_mode:
            if phase == 'train':
                self.r_rand = 0.0
                self.img_indices = []
                self.bboxes = []
                for idx, pair in enumerate(self.pairs):
                    if len(pair['label']) > 0:
                        for label in pair['label']:
                            if label['bbox'][-1] > 2.5:
                                self.img_indices += [idx] * 1
                                self.bboxes += [label] * 1
                            if label['bbox'][-1] > 10:
                                self.img_indices += [idx] * 2
                                self.bboxes += [label] * 2
                            if label['bbox'][-1] > 20:
                                self.img_indices += [idx] * 4
                                self.bboxes += [label] * 4
                    # else:
                    #     self.img_indices += [idx] * 1
                    #     self.bboxes += [{}] * 1
            elif phase == 'valid':
                self.r_rand = 0.0
                self.img_indices = []
                self.bboxes = []
                for idx, pair in enumerate(self.pairs):
                    for label in pair['label']:
                        self.img_indices += [idx] * 1
                        self.bboxes += [label] * 1
            else:
                raise NotImplementedError
        else:
            self.img_indices = range(self.__len__())

    def __len__(self):
        if not self.infer_mode:
            return int(len(self.bboxes) / (1 - self.r_rand))
        else:
            return len(self.pairs)

    # def __getitem__(self, i):
    #     if i >= len(self.bboxes):
    #         r = np.random.RandomState(0)
    #         r_idx = r.randint(len(self.pairs))
    #         return super().__getitem__(r_idx)
    #     else:
    #         return super().__getitem__(i)

    def __getitem__(self, i):
        image_file = os.path.join(self.image_prefix, self.pairs[self.img_indices[i]]['image']).replace("\\", '/')
        weight_file = image_file.replace('.nii.gz', '_dist.nii.gz')
        new_label_file = None
        assert os.path.exists(image_file), f"image file: {image_file} not exist"

        if not self.infer_mode:
            label_file = self.pairs[self.img_indices[i]]['label']
            assert is_list_of(label_file, dict)
            target_bbox = self.bboxes[i]
            new_label_file = [target_bbox]  # used with first det crop
            for bbox in label_file:
                if bbox != target_bbox:
                    new_label_file.append(bbox)

        results = self.pre_pipeline(image_file, new_label_file, weight_path=weight_file)
        results = self.pipeline(results)
        return results