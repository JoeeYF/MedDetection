import torch.nn as nn
import math

from ..nd import AdaptiveAvgPoolNd
from .res_block import BasicBlockNd, BottleneckNd
from ..nnModules import BlockModule


class SEModule(BlockModule):

    def __init__(self, dim, channels, reduction):
        super(SEModule, self).__init__()
        self.dim = dim
        self.avg_pool = AdaptiveAvgPoolNd(dim)(1)
        self.fc1 = self.build_conv(dim, channels, channels // reduction, kernel_size=1,
                                   padding=0)
        self.relu = self.build_act()
        self.fc2 = self.build_conv(dim, channels // reduction, channels, kernel_size=1,
                                   padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBasicBlockNd(BasicBlockNd):
    expansion = 1

    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, groups=1,
                 width_per_group=64, dilation=1, reduction=16):
        super(SEBasicBlockNd, self).__init__(dim, in_planes, planes, stride, downsample, groups,
                                             width_per_group, dilation)
        self.se_module = SEModule(dim, planes * self.expansion, reduction=reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneckNd(BottleneckNd):
    expansion = 4

    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, groups=1,
                 width_per_group=64, dilation=1, reduction=16):
        super(SEBottleneckNd, self).__init__(dim, in_planes, planes, stride, downsample, groups,
                                             width_per_group, dilation)
        self.se_module = SEModule(dim, planes * self.expansion, reduction=reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out
