

import numpy as np
import torch
import os.path as osp
import matplotlib.pyplot as plt

from ..monitor import Monitor

from meddet.data import ImageIO
from meddet.data.visulaize import getBBox2D


class DetMonitor(Monitor):
    def _viewData(self, data):
        img = data['img'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
        dim = img.ndim
        bboxes = data['gt_det'][0, :, :2*dim].detach().cpu().numpy()
        labels = data['gt_det'][0, :, 2*dim].detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(5, 5))
        if dim == 2:
            show = getBBox2D(img, bboxes, labels)
            ax.imshow(show, vmin=0, vmax=1)
        else:
            z_idx = int(bboxes[0][2] + bboxes[0][5]) // 2
            tmp_bboxes = []
            tmp_labels = []
            for idx, bbox in enumerate(bboxes):
                if bbox[2] <= z_idx <= bbox[5]:
                    tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                    tmp_labels.append(labels[idx])
            show = getBBox2D(img[z_idx, ...], tmp_bboxes, tmp_labels)
            ax.imshow(show, vmin=0, vmax=1)

    def _viewFeatures(self, data, net_output: tuple):
        img = data['img'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
        cls_scores, reg_scores = net_output
        z_idx = None
        if img.ndim == 3:
            bboxes = data['gt_det'][0, :, :6].detach().cpu().numpy()
            z_idx = int(bboxes[0][2] + bboxes[0][5]) // 2

        num_levels = len(cls_scores)
        num_scales = cls_scores[0].shape[1]  # different anchor scales

        fig, ax = plt.subplots(num_levels, num_scales + 1, figsize=(5 * num_scales // 1 + 5, 5 * num_levels))
        ax = ax.flatten()
        for i in range(num_levels):
            cls_score, reg_score = cls_scores[i][0].float(), reg_scores[i][0].float()
            stride = img.shape[0] // cls_score[0].shape[0]

            if img.ndim == 3:
                show = img[z_idx, ...]
            else:
                show = img

            ax0 = ax[i * (num_scales + 1)].imshow(show, vmin=0, vmax=1)
            fig.colorbar(ax0, ax=ax[i * (num_scales + 1)])

            for j in range(num_scales):
                x = cls_score[j]
                if img.ndim == 3:
                    x = x[z_idx // stride, ...]
                x = torch.sigmoid(x)
                x = x.detach().cpu().numpy()

                ax1 = ax[i * (num_scales + 1) + j + 1].imshow(x, vmin=0, vmax=1)
                fig.colorbar(ax1, ax=ax[i * (num_scales + 1) + j + 1])

    def _viewResults(self, data, results):
        img = data['img'][0, 0].detach().cpu().numpy() * 0.5 + 0.5
        dim = img.ndim
        bboxes = data['gt_det'][0, :, :2*dim].detach().cpu().numpy()
        labels = data['gt_det'][0, :, 2*dim].detach().cpu().numpy()
        dets = results[0]
        dets = dets[dets[..., -1] != -1].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(5, 5))
        if dim == 2:
            show = getBBox2D(img, bboxes, labels)
            ax[0].imshow(show, vmin=0, vmax=1)
            det_bboxes = dets[:, :4]
            det_labels = dets[:, 4]
            det_scores = dets[:, 5]
            show = getBBox2D(img, det_bboxes, det_labels, det_scores)
            ax[1].imshow(show, vmin=0, vmax=1)
        else:
            z_idx = int(bboxes[0][2] + bboxes[0][5]) // 2

            tmp_bboxes = []
            tmp_labels = []
            for idx, bbox in enumerate(bboxes):
                if bbox[2] <= z_idx <= bbox[5]:
                    tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                    tmp_labels.append(labels[idx])
            show = getBBox2D(img[z_idx, ...], tmp_bboxes, tmp_labels)
            ax[0].imshow(show, vmin=0, vmax=1)

            tmp_bboxes = []
            tmp_labels = []
            tmp_scores = []
            for idx, bbox in enumerate(dets):
                if bbox[2] <= z_idx <= bbox[5]:
                    tmp_bboxes.append(bbox[[0, 1, 3, 4]])
                    tmp_labels.append(bbox[6])
                    tmp_scores.append(bbox[7])
            show = getBBox2D(img[z_idx, ...], tmp_bboxes, tmp_labels, tmp_scores)
            ax[1].imshow(show, vmin=0, vmax=1)

    def saveResults(self,
                    data,
                    net_output,
                    save_raw_image=True,
                    save_prob=True,
                    save_label=True,
                    sp=None,
                    *args,
                    **kwargs):
        """
        :param data:
        :param net_output: cls_scores, reg_scores; both are a list of result on a single level feature map
        :param save_raw_image:
        :param save_prob:
        :param save_label:
        :param sp:
        :return:
        """
        if not sp:
            sp = self.sample_rate
        if np.random.random() >= sp:
            return
        
        assert all([i in data.keys() for i in ['img', 'img_meta']])

        cls_scores, reg_scores = net_output
        num_samples = data['img'].shape[0]
        for i in range(num_samples):
            img = data['img'][i]  # c, d, h, w
            filename = data['img_meta'][i]['filename']
            # print(filename)
            cls_score, reg_score = cls_scores[0][i], reg_scores[0][i]
            # print(cls_score.shape)
            img = img.cpu().numpy()
            cls_score = torch.sigmoid(cls_score).detach().cpu().numpy()  # .squeeze(0)
            reg_score = reg_score.detach().cpu().numpy()  # .squeeze(0)

            if len(cls_score.shape) == 4:
                print('0000')
                # 3D output, img  [C,D,W,H]
                if save_prob:
                    for c in range(cls_score.shape[0]):
                        ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_output_cls_{c}.nii.gz"), cls_score[c])
                    for c in range(reg_score.shape[0]):
                        ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_output_reg_{c}.nii.gz"), reg_score[c])
                if save_raw_image:
                    for c in range(img.shape[0]):
                        ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_img_{c}.nii.gz"), img[c])

                # ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_img_seg.nii.gz"), segmentation)

            elif len(cls_score.shape) == 3:
                # print(img.shape)
                # 2D output, img  [C,H,W]
                img = (img + 1.0) / 2.0  # [-1.0, 1.0] => [0.0, 1.0]
                if img.shape[0] == 1:
                    img = np.stack([img] * 3, axis=0)
                if img.shape[0] != 3:
                    img = img[1:4, ...]
                assert img.shape[0] == 3, f'{img.shape} must be RGB image'
                # img = img.transpose((1, 2, 0))  # CHW => HWC

                if save_prob:
                    ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_output_cls.nii.gz"), cls_score)
                    ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_output_reg.nii.gz"), reg_score)
                if save_raw_image:
                    ImageIO.saveArray(osp.join(self.result_dir, f"{filename}_img.nii.gz"), img.squeeze())
