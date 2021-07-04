

import os
import json
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

from ..pipelines import ForwardCompose, LoadPrepare
from ..registry import DATASETS
from ...utils import is_list_of


@DATASETS.register_module
class BasicPairDataset(Dataset):
    dataset_format = {
        "name":        "Luna16",
        "description": "Luna16",
        "categories":  [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "target1"}
        ],
        "pairs":       [
            {"image": "images/1.png", "label": 1},  # Classification
            {"image": "images/1.png", "label": "labels/1.png"},  # 2D Segmentation
            {"image": "images/1.nii", "label": "labels/1.nii"},  # 3D Segmentation
            {"image": "images/1.png", "label": [{'bbox': ['x', 'y', 'w', 'h'], 'category_id': 1}]},  # 2D Detection
            {"image": "images/1.nii", "label": [{'bbox': ['x', 'y', 'z', 'w', 'h', 'd'], 'category_id': 2}]},
            # 3D Detection
            {"image": "images/1.png"}  # Inference
        ]
    }

    def __init__(self, task, dataset_file, image_prefix, pipeline, infer_mode=False):
        self.task = task
        self.dataset = json.load(open(dataset_file))
        self.check_dataset_format(self.dataset)

        self.dataset_file = dataset_file
        self.image_prefix = image_prefix
        self.infer_mode = infer_mode
        self.pairs = self.dataset['pairs']

        self.pre_pipeline = LoadPrepare()
        self.pipeline = ForwardCompose(pipeline)
        self.properties = OrderedDict()

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dataset_file={}, image_prefix={},infer_mode={})'.format(
            self.dataset_file, self.image_prefix, self.infer_mode)
        repr_str += '\nPipeline: \n{}'.format(self.pipeline)
        return repr_str

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        image_file = os.path.join(self.image_prefix, self.pairs[i]['image']).replace("\\", '/')
        label_file = ''
        assert os.path.exists(image_file), f"image file: {image_file} not exist"

        if not self.infer_mode:
            # Classification
            if self.task == 'CLS':
                label_file = self.pairs[i]['label']
            # Segmentation
            elif self.task == 'SEG':
                label_file = os.path.join(self.image_prefix, self.pairs[i]['label']).replace("\\", '/')
                assert os.path.exists(label_file), f"label file: {label_file} not exist"
            # Detection
            elif self.task == 'DET':
                label_file = self.pairs[i]['label']
                assert is_list_of(label_file, dict)
                # print(label_file)

        results = self.pre_pipeline(image_file, label_file)
        results = self.pipeline(results)
        # for i in results:
        #     print(i['img_meta'])
        return results

    def setLatitude(self, val):
        self.pipeline.setLatitude(val)

    def getLatitude(self):
        return self.pipeline.getLatitude()

    def cross_valid_split(self, K_FOLD, fold):
        kf = KFold(n_splits=K_FOLD, shuffle=True)
        train_indices, valid_indices = list(kf.split(range(len(self))))[fold]
        train_indices, valid_indices = train_indices.tolist(), valid_indices.tolist()
        return train_indices, valid_indices

    def check_dataset_format(self, dataset):
        assert "categories" in dataset.keys(), "categories not in the keys"
        assert "pairs" in dataset.keys(), "images not in the keys"
        assert is_list_of(dataset['categories'], dict), json.dumps(self.dataset_format, indent=4)
        assert is_list_of(dataset['pairs'], dict), print(self.dataset_format)
        return True

    def statistic(self):
        assert 'Property' in self.pipeline

        for data in tqdm(self):
            for d in data:
                for k, v in d['properties'].items():
                    if k not in self.properties.keys():
                        self.properties[k] = []
                    self.properties[k].append(v)

        df = pd.DataFrame(self.properties)
        df.to_csv(os.path.join(self.image_prefix, 'dataset_properties.csv'))

        with open(os.path.join(self.image_prefix, 'dataset_properties.txt'), 'w') as f:
            for k, v in self.properties.items():
                if isinstance(v[0], str):
                    continue
                elif isinstance(v[0], np.ndarray):
                    self.properties[k] = np.stack(v, axis=0)
                else:
                    self.properties[k] = np.array(v)
                stat = OrderedDict({'name':   k,
                                    "min  :": np.min(self.properties[k], axis=0),
                                    "max  :": np.max(self.properties[k], axis=0),
                                    "mean :": np.mean(self.properties[k], axis=0),
                                    "75%  :": np.percentile(self.properties[k], 75, axis=0),
                                    "50%  :": np.percentile(self.properties[k], 50, axis=0),
                                    "25%  :": np.percentile(self.properties[k], 25, axis=0)})
                for _k, _v in stat.items():
                    print(_k, _v)
                    f.write(f"{_k} {_v}\n")
                print('\n')
                f.write("\n")
        f.close()


@DATASETS.register_module
class BasicDetPairDataset(BasicPairDataset):
    dataset_format = {
        "name":        "Luna16",
        "description": "Luna16",
        "categories":  [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "target1"}
        ],
        "pairs":       [
            {"image": "images/1.png", "label": [{'bbox': ['x', 'y', 'w', 'h'], 'category_id': 1}]},  # 2D Detection
            {"image": "images/1.nii", "label": [{'bbox': ['x', 'y', 'z', 'w', 'h', 'd'], 'category_id': 2}]},
            # 3D Detection
            {"image": "images/1.png"}  # Inference
        ]
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.task == 'DET'
        if not self.infer_mode:
            self.img_indices = []
            self.bboxes = []
            for idx, pair in enumerate(self.pairs):
                for label in pair['label']:
                    self.img_indices += [idx]
                    self.bboxes += [label]
        else:
            self.img_indices = range(self.__len__())

    def __len__(self):
        if not self.infer_mode:
            return len(self.bboxes)
        else:
            return len(self.pairs)

    def __getitem__(self, i):
        image_file = os.path.join(self.image_prefix, self.pairs[self.img_indices[i]]['image']).replace("\\", '/')
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

        results = self.pre_pipeline(image_file, new_label_file)
        results = self.pipeline(results)
        return results

    def check_dataset_format(self, dataset):
        assert "categories" in dataset.keys(), "categories not in the keys"
        assert "pairs" in dataset.keys(), "images not in the keys"
        assert is_list_of(dataset['categories'], dict), json.dumps(self.dataset_format, indent=4)
        assert is_list_of(dataset['pairs'], dict), print(self.dataset_format)
        return True

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dataset_file={}, image_prefix={},infer_mode={})'.format(
            self.dataset_file, self.image_prefix, self.infer_mode)
        repr_str += '\nPipeline: \n{}'.format(self.pipeline)
        return repr_str
