

from ..nnModules import BlockModule
from medvision.ops import DeformConv3dPack as DCNv1
from medvision.ops import ModulatedDeformConv3dPack as DCNv2


class BasicBlockNd(BlockModule):
    expansion = 1

    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, groups=1,
                 width_per_group=64, dilation=1, dcn=None):
        super(BasicBlockNd, self).__init__()
        if groups != 1 or width_per_group != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        assert dcn is None, 'Not implemented yet.'
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.dim = dim
        self.conv1 = self.build_conv(self.dim, in_planes, planes, stride=stride, kernel_size=3, padding=1,
                                     bias=False)  # conv3x3x3(in_planes, planes, stride)
        self.bn1 = self.build_norm(self.dim, planes)
        self.relu = self.build_act()
        self.conv2 = self.build_conv(self.dim, planes, planes, stride=1, kernel_size=3, padding=1,
                                     bias=False)  # conv3x3x3(planes, planes)
        self.bn2 = self.build_norm(self.dim, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckNd(BlockModule):
    expansion = 4

    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, groups=1,
                 width_per_group=64, dilation=1, dcn=None):
        super(BottleneckNd, self).__init__()
        width = int(planes * (width_per_group / 64.)) * groups
        # print(in_planes, planes, width, planes * self.expansion)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.dim = dim
        self.conv1 = self.build_conv(self.dim, in_planes, width, kernel_size=1, stride=1,
                                     bias=False)  # conv1x1x1(in_planes, width)
        self.bn1 = self.build_norm(self.dim, width)

        if dcn is not None:
            self.conv2 = self.build_dcn(self.dim, width, width, kernel_size=3, stride=stride, padding=dilation,groups=groups,dilation=dilation, bias=False,**dcn)
        else:
            self.conv2 = self.build_conv(self.dim, width, width, stride=stride, kernel_size=3, padding=dilation,
                                         groups=groups, dilation=dilation,
                                         bias=False)

        self.bn2 = self.build_norm(self.dim, width)
        self.conv3 = self.build_conv(self.dim, width, planes * self.expansion, kernel_size=1, stride=1,
                                     bias=False)  # conv1x1x1(width, planes * self.expansion)
        self.bn3 = self.build_norm(self.dim, planes * self.expansion)
        self.relu = self.build_act()
        self.downsample = downsample
        self.stride = stride

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

        out += residual
        out = self.relu(out)

        return out
