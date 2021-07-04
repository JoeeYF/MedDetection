import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate

from meddet.data.registry import DATASETS

config = {}
config['anchors'] = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
config['chanel'] = 1
config['crop_size'] = [96, 96, 96]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 2.5  # 3 #6. #mm
config['sizelim2'] = 10  # 30
config['sizelim3'] = 20  # 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.0
config['pad_value'] = 170
config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}

config['augtype'] = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']


class SplitComb:
    def __init__(self, side_len, max_stride, stride, margin, pad_value):
        self.side_len = side_len
        self.max_stride = max_stride
        self.stride = stride
        self.margin = margin
        self.pad_value = pad_value
        self.nzhw = None

    def split(self, data, side_len=None, max_stride=None, margin=None):
        if side_len is None:
            side_len = self.side_len
        if max_stride is None:
            max_stride = self.max_stride
        if margin is None:
            margin = self.margin

        assert (side_len > margin)
        assert (side_len % max_stride == 0)
        assert (margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))

        nzhw = [nz, nh, nw]
        self.nzhw = nzhw

        pad = [[0, 0],
               [margin, nz * side_len - z + margin],
               [margin, nh * side_len - h + margin],
               [margin, nw * side_len - w + margin]]
        data = np.pad(data, np.int64(pad), 'edge')

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len + 2 * margin
                    sh = ih * side_len
                    eh = (ih + 1) * side_len + 2 * margin
                    sw = iw * side_len
                    ew = (iw + 1) * side_len + 2 * margin

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, nzhw

    def combine(self, output, nzhw=None, side_len=None, stride=None, margin=None):

        if side_len is None:
            side_len = self.side_len
        if stride is None:
            stride = self.stride
        if margin is None:
            margin = self.margin
        if nzhw is None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz, nh, nw = nzhw
        assert (side_len % stride == 0)
        assert (margin % stride == 0)
        side_len /= stride
        margin /= stride

        splits = []
        for i in range(len(output)):
            splits.append(output[i])

        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3],
            splits[0].shape[4]), np.float32)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len

                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len]
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1

        return output


@DATASETS.register_module
class DataBowl3Detector(Dataset):
    def __init__(self,
                 phase='train',
                 split_comber=None,
                 **kwargs):
        split_path = []

        if phase == 'train':
            data_dir = os.getenv("DATASETS") + "/Detection/DeepLungDataset/"
            data_path = [data_dir + '/subset0/',
                         data_dir + '/subset1/',
                         data_dir + '/subset2/',
                         data_dir + '/subset3/',
                         data_dir + '/subset4/',
                         data_dir + '/subset5/',
                         data_dir + '/subset6/',
                         data_dir + '/subset7/',
                         data_dir + '/subset8/']
        else:
            sidelen = 144
            margin = 32
            split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
            data_dir = os.getenv("DATASETS") + "/Detection/DeepLungDataset/"
            data_path = [data_dir + '/subset9/']

        for folder in data_path:
            for f in os.listdir(folder):
                if f.endswith('_clean.npy') and f[:-10] not in []:
                    split_path.append(folder.split('/')[-2] + '/' + f[:-10])
        split_path = sorted(split_path)

        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = split_path  # np.load(split_path)
        if phase != 'test':
            idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = sorted([os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs])
        # print self.filenames
        self.kagglenames = [f for f in self.filenames]  # if len(f.split('/')[-1].split('_')[0])>20]
        # self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0])<20]

        labels = []
        print(self.phase)
        print(len(idcs))
        for idx in idcs:
            # print data_dir, idx
            l = np.load(data_dir + idx + '_label.npy', allow_pickle=True)
            # print l, os.path.join(data_dir, '%s_label.npy' %idx)
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)

        self.sample_bboxes = labels  # bboxes of each images
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):
                # print l
                if len(l) > 0:
                    for t in l:
                        # self.bboxes.append([np.concatenate([[i], t])])
                        # bigger nodule, more samples
                        if t[3] > sizelim:
                            self.bboxes.append([np.concatenate([[i], t])])
                        if t[3] > sizelim2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > sizelim3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4
            self.bboxes = np.concatenate(self.bboxes, axis=0)  # augmented bboxes, 5 (idx, z, y, x, d)
            print(len(self.bboxes))
        print(len(self.sample_bboxes))

        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def getLatitude(self):
        return -1

    def setLatitude(self, val):
        return -1

    def __getitem__(self, idx, split=None):
        # print(idx, self.__len__())
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        isRandomImg = False
        if self.phase != 'test':
            if idx >= len(self.bboxes):
                isRandom = True
                idx = idx % len(self.bboxes)
                isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False

        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype['scale'] and (self.phase == 'train')
                # cropped image , target in image, augment bboxes, anchors [(x,y,z), shape of image]
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if self.phase == 'train' and not isRandom:
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:
                # print("random")
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype['scale'] and (self.phase == 'train')
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
            # print(sample.shape, target.shape, bboxes.shape)
            label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            # 0=ignore, -1=neg, 1=pos
            # print(filename)
            # print(np.sum(label[..., 0] == 0), np.sum(label[..., 0] == -1), np.sum(label[..., 0] == 1))
            sample = (sample.astype(np.float32) - 128) / 128
            # if filename in self.kagglenames and self.phase=='train':
            #    label[label==-1]=0
            tmp_target = target[None, :].copy()
            tmp_target[:, :3] = tmp_target[:, :3][:, ::-1]
            gt_det = np.ones((tmp_target.shape[0], 8))
            gt_det[:, :4] = tmp_target
            gt_det[:, 4] = gt_det[:, 3]
            gt_det[:, 5] = gt_det[:, 3]
            gt_det[:, :3] -= gt_det[:, 3:6] / 2
            gt_det[:, 3:6] += gt_det[:, :3]

            return {'img':       torch.from_numpy(sample),
                    'gt_labels': torch.from_numpy(label),
                    'gt_bboxes': torch.from_numpy(coord),
                    'img_meta':  {'filename': os.path.basename(filename)},
                    'gt_det':    torch.from_numpy(gt_det),
                    }
        else:
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] // self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len // self.stride,
                                                    max_stride=self.split_comber.max_stride / self.stride,
                                                    margin=self.split_comber.margin // self.stride)
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return {'patches_img':       imgs,
                    'gt_bboxes': bboxes,
                    'patches_coord': torch.from_numpy(coord2),
                    'nzhw':      np.array(nzhw),
                    'filename':  os.path.basename(self.filenames[idx]),
                    'img_dim': 3,
                    }

    def __len__(self):
        if self.phase == 'train':
            # print(int(len(self.bboxes) / (1 - self.r_rand)))
            return int(len(self.bboxes) / (1 - self.r_rand))
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord


class Crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size']
        self.bound_size = config['bound_size']
        self.stride = config['stride']
        self.pad_value = config['pad_value']

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        """
        Args:
            imgs: raw images with channel, shape is [1, z, y, x]
            target: target bbox, shape is (4,), repr (z,y,x,d)
            bboxes: all bboxes in the image, shape is (n, 4)
            isScale: if augment has scale
            isRand: if is random image

        Returns:
            crop, target, bboxes, coord
            crop: cropped image, shape is crop shape
            target: subtract crop start position, shape is (4,)
            bboxes: subtract crop start position, shape is (n, 4)
            coord: anchors with crop size, shape is (3, shape_z, shape_y, shape_x) 3=z_pos,y_pos,x_pos value, usually is 2 * dim
        """
        if isScale:
            # scale image and bbox
            # if nodule is smaller than 8., enlarge the nodule
            # else make the nodule smaller
            radiusLim = [8., 120.]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        zz, yy, xx = np.meshgrid(
            np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] // self.stride),
            np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] // self.stride),
            np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] // self.stride),
            indexing='ij')
        coord = np.concatenate([zz[np.newaxis, ...], yy[np.newaxis, ...], xx[np.newaxis, :]], 0).astype('float32')

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale
        return crop, target, bboxes, coord


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        self.debug = False
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']

    def try_to_print(self, *args):
        if self.debug:
            print(*args)

    def __call__(self, input_size, target, bboxes, filename):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            if input_size[i] % stride != 0:
                print(filename)
            # assert(input_size[i] % stride == 0)
            output_size.append(input_size[i] // stride)

        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)  # 5 = cls, dz, dh, dw, dd
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True
        if len(iz) == 0:
            self.try_to_print('adding most matching anchor')
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            self.try_to_print("sampling i anchor from matched anchors", len(iz), len(ih), len(iw))
            self.try_to_print("gt bbox d=", target[3])
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
            self.try_to_print(pos)
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      print(['iou not large', iou.max()])
        # else:
        #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


if __name__ == "__main__":
    from meddet.data import ImageIO
    from scipy.ndimage import zoom, binary_dilation

    d = DataBowl3Detector()
    for i in range(len(d)):
        print(i)
        a = d.__getitem__(i)
        filename = a['img_meta']['filename']
        print(filename)
        ImageIO.saveArray(f'/home/timothy/PycharmProjects/MedToolkit/medtk/data/tmp/DSB/{filename}_img.nii', a['img'].numpy())
        ImageIO.saveArray(f'/home/timothy/PycharmProjects/MedToolkit/medtk/data/tmp/DSB/{filename}_det_1.nii', zoom(a['gt_labels'][None, ..., 0, 0].numpy(), (1, 4, 4, 4), order=0))
        ImageIO.saveArray(f'/home/timothy/PycharmProjects/MedToolkit/medtk/data/tmp/DSB/{filename}_det_2.nii', zoom(a['gt_labels'][None, ..., 1, 0].numpy(), (1, 4, 4, 4), order=0))
        ImageIO.saveArray(f'/home/timothy/PycharmProjects/MedToolkit/medtk/data/tmp/DSB/{filename}_det_3.nii', zoom(a['gt_labels'][None, ..., 2, 0].numpy(), (1, 4, 4, 4), order=0))
        if i == 5:
            break
