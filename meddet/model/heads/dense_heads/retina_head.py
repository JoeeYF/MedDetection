

from typing import Union, List
import torch.nn as nn
import torch
import warnings
import math
import numpy as np

from meddet.model.heads.dense_heads.anchor_head import AnchorHead, images_to_levels
from meddet.model.nnModules import ComponentModule



class SubnetReg(ComponentModule):
    def __init__(self, dim: int, in_channels: int, num_base_anchors=9, feature_size=256):
        self.dim = dim
        super(SubnetReg, self).__init__()

        self.conv1 = self.build_conv(dim, in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = self.build_conv(dim, feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act3 = nn.ReLU()

        self.conv4 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = nn.ReLU()

        self.output = self.build_conv(dim, feature_size, num_base_anchors * 2 * dim, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        return out


class SubnetCls(ComponentModule):
    def __init__(self, dim: int, in_channels: int, num_base_anchors=9, num_classes=80, feature_size=256):
        super(SubnetCls, self).__init__()

        self.num_classes = num_classes
        self.base_anchors = num_base_anchors

        self.conv1 = self.build_conv(dim, in_channels, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = self.build_conv(dim, feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act3 = nn.ReLU()

        self.conv4 = self.build_conv(dim, feature_size, feature_size, kernel_size=1, padding=0)
        self.act4 = nn.ReLU()

        self.output = self.build_conv(dim, feature_size, num_base_anchors * num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        return out


class RetinaHead(AnchorHead):
    def __init__(self, **kwargs):
        super(RetinaHead, self).__init__(**kwargs)

        self.cls_out_channels = self.num_classes  # because serious classes imbalance problem, remove background class

    def _init_layers(self):
        self.conv_cls = SubnetCls(self.dim,
                                  in_channels=self.in_channels,
                                  num_base_anchors=self.num_anchors,
                                  num_classes=self.cls_out_channels,
                                  feature_size=self.feat_channels)
        self.conv_reg = SubnetReg(self.dim,
                                  in_channels=self.in_channels,
                                  num_base_anchors=self.num_anchors,
                                  feature_size=self.feat_channels)
        # self.conv_cls = nn.Sequential(nn.Conv3d(128, 64, kernel_size=1),
        #                               nn.ReLU(),
        #                               # nn.Dropout3d(p = 0.3),
        #                               nn.Conv3d(64, 1 * self.num_anchors, kernel_size=1))
        # self.conv_reg = nn.Sequential(nn.Conv3d(128, 64, kernel_size=1),
        #                               nn.ReLU(),
        #                               # nn.Dropout3d(p = 0.3),
        #                               nn.Conv3d(64, 6 * self.num_anchors, kernel_size=1))

    def init_weights(self):
        pass
        # for m in self.modules():
        #     if self.is_conv(self.dim, m):
        #         n = np.prod(m.kernel_size) * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif self.is_norm(self.dim, m):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #
        # prior = 0.01
        # self.conv_cls.output.weight.data.fill_(0)
        # self.conv_cls.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        #
        # self.conv_reg.output.weight.data.fill_(0)
        # self.conv_reg.output.bias.data.fill_(0)

    def forward_single_level(self, feature):
        cls_out = self.conv_cls(feature)  # (b, nb * classes, (z,) y, x)
        reg_out = self.conv_reg(feature)  # (b, nb * 2 * dim, (z,) y, x)
        return cls_out, reg_out

    # def _init_layers(self):
    #     self.output = nn.Sequential(nn.Conv3d(128, 64, kernel_size=1),
    #                                 nn.ReLU(),
    #                                 # nn.Dropout3d(p = 0.3),
    #                                 nn.Conv3d(64, 7 * self.num_anchors, kernel_size=1))
    #
    # def init_weights(self):
    #     pass
    #     # for m in self.modules():
    #     #     if self.is_conv(self.dim, m):
    #     #         n = np.prod(m.kernel_size) * m.out_channels
    #     #         m.weight.data.normal_(0, math.sqrt(2. / n))
    #     #     elif self.is_norm(self.dim, m):
    #     #         m.weight.data.fill_(1)
    #     #         m.bias.data.zero_()
    #     #
    #     # prior = 0.01
    #     # self.rpn_cls.weight.data.fill_(0)
    #     # self.rpn_cls.bias.data.fill_(-math.log((1.0 - prior) / prior))
    #     #
    #     # self.rpn_reg.weight.data.fill_(0)
    #     # self.rpn_reg.bias.data.fill_(0)
    #
    #     # normal_init(self.rpn_bone, std=0.01)
    #     # normal_init(self.rpn_cls, std=0.001, bias=-math.log((1.0 - prior) / prior))
    #     # normal_init(self.rpn_reg, std=0.01)
    #
    # def forward_single_level(self, single_level_feature):
    #     out = self.output(single_level_feature)
    #     cls_out = out[:, :self.anchor_generator.num_base_anchors, ...]
    #     reg_out = out[:, self.anchor_generator.num_base_anchors:, ...]
    #     return cls_out, reg_out

