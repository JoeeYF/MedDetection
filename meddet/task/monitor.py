

from typing import Union, List
import shutil
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from meddet.data import BackwardCompose


class Monitor(object):
    def __init__(self, work_dir, sample_rate=1, result_pipeline=[]):
        self.work_dir = work_dir
        self.sample_rate = sample_rate

        self.train_results_dir = "{}/train_results".format(work_dir)
        self.valid_results_dir = "{}/valid_results".format(work_dir)
        self.infer_results_dir = "{}/infer_results".format(work_dir)

        os.makedirs(self.train_results_dir, exist_ok=True)
        os.makedirs(self.valid_results_dir, exist_ok=True)
        os.makedirs(self.infer_results_dir, exist_ok=True)

        self.mode = 'train'
        self.result_dir = self.train_results_dir
        self.displayable = eval(os.environ.get('SHOW', 'False'))

        self.result_pipeline = BackwardCompose(result_pipeline)

        if matplotlib.get_backend() == "Qt5Agg" or not self.displayable:
            print('not displayable')
            matplotlib.use('Agg')

        # self.clearAll()

    def setTrainMode(self):
        self.mode = 'train'
        self.result_dir = self.train_results_dir

    def setValidMode(self):
        self.mode = 'valid'
        self.result_dir = self.valid_results_dir

    def setInferMode(self):
        self.mode = 'infer'
        self.result_dir = self.infer_results_dir

    def clearAll(self):
        for dirname in [self.train_results_dir, self.valid_results_dir, self.infer_results_dir]:
            self._clear(dirname)

    @staticmethod
    def _clear(dirname):
        for filename in os.listdir(dirname):
            file_path = os.path.join(dirname, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    @staticmethod
    def normalize(image):
        return Normalize()(image)

    def _viewData(self, data):
        raise NotImplementedError

    def viewData(self,
                 data: torch.Tensor,
                 force_save=False):
        try:
            # do some plot job without plt.show
            self._viewData(data)

            fig = plt.gcf()
            if not self.displayable or force_save:
                filename = data['img_meta'][0]['filename']
                fig.savefig(os.path.join(self.result_dir,
                                         f"Epoch{data['epoch']}_Iter{data['iter']}_{filename}_viewData.jpg"), dpi=200)
                plt.close(fig)
            else:
                fig.show()
        except Exception as e:
            print(e)

    def _viewFeatures(self, data, net_output):
        raise NotImplementedError

    def viewFeatures(self,
                     data: dict,
                     net_output: Union[List[torch.Tensor], torch.Tensor],
                     force_save=False):
        try:
            self._viewFeatures(data, net_output)

            fig = plt.gcf()
            if not self.displayable or force_save:
                filename = data['img_meta'][0]['filename']
                fig.savefig(os.path.join(self.result_dir,
                                         f"Epoch{data['epoch']}_Iter{data['iter']}_{filename}_viewFeatures.jpg"), dpi=200)
                plt.close(fig)
            else:
                fig.show()
        except Exception as e:
            print(e)

    def _viewResults(self, data, results):
        raise NotImplementedError

    def viewResults(self, data, results, force_save=False):
        self._viewResults(data, results)

        fig = plt.gcf()
        if not self.displayable or force_save:
            filename = data['img_meta'][0]['filename']
            fig.savefig(os.path.join(self.result_dir,
                                     f"Epoch{data['epoch']}_Iter{data['iter']}_{filename}_viewDataResults.jpg"), dpi=200)
            plt.close(fig)
        else:
            fig.show()

    def saveResults(self,
                    data,
                    net_output,
                    save_raw_image=True,
                    save_prob=True,
                    save_label=True,
                    sp=None,
                    *args,
                    **kwargs):
        raise NotImplementedError
