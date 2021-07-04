import torch
from torch import nn
import numpy as np

from meddet.model.losses.deeplung_loss import Loss
from meddet.model.metrics import IOU, Dist
from medvision.ops.torch import nmsNd_pytorch
from ..task import BaseTask


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class GetPBB(object):
    def __init__(self, stride, anchors):
        self.stride = stride
        self.anchors = np.asarray(anchors)

    def __call__(self, output, thresh=0.15, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output.detach().cpu().numpy())
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)

        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa]
        else:
            return output


class DeepLung(BaseTask):
    def __init__(self,
                 dim,
                 add_coord=True):
        super(DeepLung, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.dim = dim
        self.add_coord = add_coord
        self.anchors = [5., 10., 20.]
        self.loss = Loss(2)
        self.get_pbb = GetPBB(stride=4, anchors=self.anchors)
        self.metrics = [
            Dist(aggregate='none', dist_threshold=5)
        ]
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3 if self.add_coord else 0
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                    nn.ReLU(),
                                    # nn.Dropout3d(p = 0.3),
                                    nn.Conv3d(64, 5 * len(self.anchors), kernel_size=1))

    def _forward(self, img_meta, img, coord):
        x = img
        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        # out2 = self.drop(out2)
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96
        # out4 = self.drop(out4)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96
        # comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)

        if self.add_coord:
            comb2 = torch.cat((rev2, out2, coord), 1)
        else:
            comb2 = torch.cat((rev2, out2), 1)
        comb2 = self.back2(comb2)  # 64+64
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        size = out.size()  # b, 15(anchor * 5), 24, 24, 24
        out = out.view(out.size(0), out.size(1), -1)
        # out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(self.anchors), 5)
        # out = out.view(-1, 5)

        return out

    def get_loss(self, out, labels):
        loss_output = self.loss(out, labels)
        # print("loss_output", loss_output)
        loss_dict = {
            'tpr':       100.0 * loss_output[6] / loss_output[7],
            'tnr':       100.0 * loss_output[8] / loss_output[9],
            'reference': torch.zeros(1),
            'cls_ls':    torch.tensor(loss_output[1]),
            'reg_ls':    torch.tensor(sum(loss_output[2:6])),
            'loss':      loss_output[0],
        }
        net_output = [out[..., 0].permute(0, 4, 1, 2, 3)], [out[..., 1:]]
        return loss_dict, None, net_output

    def get_bboxes(self, out):
        nms_pre = 1000
        score_thr = 0.15
        nms_threshold = 0.1
        max_output_num = 100
        predictions = torch.ones((len(out), nms_pre, 8)) * -1
        for b in range(len(out)):
            b_out = out[b]
            b_out[..., 0] = torch.sigmoid(b_out[..., 0])
            pred, _ = self.get_pbb(b_out, thresh=score_thr, ismask=True)
            pred = torch.tensor(pred).float()
            pred[:, 1:4] = torch.fliplr(pred[:, 1:4])

            scores, order = torch.sort(pred[:, 0], descending=True)
            pred = pred[order[:min(nms_pre, len(pred))]]

            if len(pred) > 0:
                dets = torch.ones((len(pred), 8)) * -1
                dets[:len(pred), :3] = pred[:, 1:4] - pred[:, [4]] / 2
                dets[:len(pred), 3:6] = pred[:, 1:4] + pred[:, [4]] / 2
                dets[:len(pred), 6] = 1
                dets[:len(pred), 7] = pred[:, 0]

                keep, _ = nmsNd_pytorch(dets[:, [0, 1, 2, 3, 4, 5, 7]], nms_threshold)
                keep = keep[:max_output_num]
                predictions[b, :len(keep)] = dets[keep]

                # predictions[b, :len(dets)] = dets

                # print(dets[0])

        # for b in range(len(img)):
        #     print(img_meta[b]['filename'])
        #     center = labels[2][b][:, :3] + labels[2][b][:, 3:6]
        #     print(center[0].detach().cpu().numpy() / 2)
        #     anchor = coord[b].permute([1, 2, 3, 0]).contiguous().view(-1, 3)
        #     max_anchor = torch.argmax(torch.max(out[b, ..., 0].view(-1, 3), dim=0)[0]).item()
        #     print(max_anchor)
        #     one_out = out[b, ..., max_anchor, :].view(-1, 5)
        #     z = torch.argmax(one_out[:, 0], keepdim=True)
        #     target_anchor = (anchor[z] - anchor[0]) / (coord[b][:, 1, 1, 1] - coord[b][:, 0, 0, 0])
        #     print(target_anchor.detach().cpu().numpy()[::-1] * 4)
        return predictions

    def forward_train(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        gt_labels = data_batch['gt_labels']
        coord = data_batch['gt_bboxes']
        gt_det = data_batch['gt_det']
        # self.info('feature')
        out = self._forward(data_batch['img_meta'], img, coord)
        # self.info('loss')

        loss_dict, _, net_output = self.get_loss(out.clone(), gt_labels)
        # self.info('done')

        # predictions = self.get_bboxes(out.clone())
        # metrics = self.metric(data_batch, predictions)
        # print(metrics)
        return loss_dict, None, net_output

    def forward_valid(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        gt_labels = data_batch['gt_labels']
        coord = data_batch['gt_bboxes']
        gt_det = data_batch['gt_det']

        out = self._forward(data_batch['img_meta'], img, coord)

        loss_dict, _, net_output = self.get_loss(out.clone(), gt_labels)

        predictions = self.get_bboxes(out.clone())

        metrics = self.metric(data_batch, predictions)
        loss_dict.update(metrics)
        return loss_dict, predictions, net_output

    def forward_infer(self, data_batch, *args, **kwargs):
        img = data_batch['img']
        coord = data_batch['coord']

        out = self._forward(data_batch['img_meta'], img, coord)

        predictions = self.get_bboxes(out.clone())
        return predictions, None

    def metric(self, data_batch, predictions):
        assert 'gt_det' in data_batch.keys()
        label = data_batch['gt_det']

        self.try_to_info("roi_results", predictions)

        metrics = {}
        for metric in self.metrics:
            one_metric = metric(predictions, label)
            metrics.update(one_metric)
        return metrics


if __name__ == "__main__":
    net = DeepLung(3)
    x = torch.rand((1, 1, 96, 96, 96))
    data = {
        'img':       x,
        'img_meta':  None,
        'gt_labels': 2 * torch.rand((1, 24, 24, 24, 3, 5)) - 1,
        'gt_bboxes': torch.ones((1, 3, 24, 24, 24)),
    }
    out = net.forward_train(data)
    print(out)
