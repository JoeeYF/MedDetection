

from abc import abstractmethod
from torch.utils.data import Dataset
import json

from ..pipelines import ForwardCompose, LoadPrepare


class BasicDataset(Dataset):
    N_FOLD = 5
    dataset_format = ""

    def __init__(self, dataset_file, image_prefix, pipeline, infer_mode=False, task=None, fold=None):
        self._TASK = task
        self._FOLD = fold

        self.dataset = json.load(open(dataset_file))
        self.check_dataset_format(self.dataset)

        self.dataset_file = dataset_file
        self.image_prefix = image_prefix
        self.infer_mode = infer_mode
        self.pairs = self.dataset['pairs']

        self.pre_pipeline = LoadPrepare()
        self.pipeline = ForwardCompose(pipeline)

    def __len__(self):
        return len(self.pairs)

    @abstractmethod
    def check_dataset_format(self, dataset) -> bool:
        return False

    @abstractmethod
    def run_training_strategy(self):
        pass

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(dataset_file={}, image_prefix={},infer_mode={})'.format(
            self.dataset_file, self.image_prefix, self.infer_mode)
        repr_str += '\nPipeline: \n{}'.format(self.pipeline)
        return repr_str