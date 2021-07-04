import math
import numpy as np
import torch.nn as nn

from meddet.runner.checkpoint import load_checkpoint
from meddet.model.utils import kaiming_init, constant_init
from meddet.model.nd import MaxPoolNd
from meddet.model.blocks import BasicBlockNd, BottleneckNd
from meddet.model.nnModules import ComponentModule


class ResNet(ComponentModule):
    arch_settings = {
        18: (BasicBlockNd, (2, 2, 2, 2)),
        34: (BasicBlockNd, (3, 4, 6, 3)),
        50: (BottleneckNd, (3, 4, 6, 3)),
        101: (BottleneckNd, (3, 4, 23, 3)),
        152: (BottleneckNd, (3, 8, 36, 3))
    }

    model_urls = {
        18: 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        34: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        50: 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        101: 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        152: 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }

    def __init__(self,
                 dim,
                 depth,
                 downsample,
                 in_channels=3,
                 base_width=64,
                 stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,
                 pretrained=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 ):

        if depth not in [18, 34, 50, 101, 152]:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        # print("ResNet Model Init")

        super(ResNet, self).__init__(conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg)
        self.dim = dim
        self.downsample = downsample
        self.depth = depth
        self.num_stages = stages
        assert 4 >= stages >= 1
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == stages
        self.out_indices = out_indices
        assert max(out_indices) < stages
        self.groups = groups
        self.width_per_group = width_per_group
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:stages]
        self.base_width = base_width
        self.inplanes = base_width
        self.pretrained = pretrained
        self.dcn = dcn
        self.stage_with_dcn=stage_with_dcn

        if dcn is not None:
            assert len(stage_with_dcn) == stages

        if self.downsample == 1:
            self.conv1 = self.build_conv(self.dim, in_channels, self.inplanes, kernel_size=7, stride=1, padding=3,
                                         bias=False)
        elif self.downsample in [2, 4]:
            self.conv1 = self.build_conv(self.dim, in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                                         bias=False)

        self.bn1 = self.build_norm(dim, self.inplanes)
        self.relu = self.build_act()
        self.maxpool = MaxPoolNd(self.dim)(kernel_size=3, stride=2, padding=1)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            dcn = self.dcn if self.stage_with_dcn[i] else None
            stride = strides[i]
            dilation = dilations[i]
            planes = self.base_width * 2 ** i

            layer_name = 'layer{}'.format(i + 1)
            # print(layer_name)

            res_layer = self._make_layer(self.block, self.inplanes, planes, num_blocks, stride, dilation,dcn=dcn)
            self.inplanes = planes * self.block.expansion

            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # self.freeze_bn()

        self.init_weights()
        # print(self)

    def _make_layer(self, block, in_planes, planes, num_blocks, stride, dilation, dcn=None):
        downsample = None
        if stride != 1 or in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                self.build_conv(self.dim, in_planes, planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False),
                self.build_norm(self.dim, planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.dim,
                in_planes=in_planes,
                planes=planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                groups=self.groups,
                width_per_group=self.width_per_group,
                dcn=dcn))
        in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    self.dim,
                    in_planes=in_planes,
                    planes=planes,
                    stride=1,
                    dilation=dilation,
                    groups=self.groups,
                    width_per_group=self.width_per_group,
                    dcn=dcn))

        return nn.Sequential(*layers)

    def init_weights(self):
        # init in mmdetection
        # for m in self.modules():
        #     if self.is_conv(self.dim, m):
        #         kaiming_init(m)
        #     elif self.is_norm(self.dim, m):
        #         constant_init(m, 1)
        #
        # if self.zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, BottleneckNd):
        #             constant_init(m.bn3, 0)
        #         elif isinstance(m, BasicBlockNd):
        #             constant_init(m.bn2, 0)

        if self.pretrained and self.dim == 2:
            load_checkpoint(self, self.model_urls[self.depth])
        else:
            # in retina net
            for m in self.modules():
                if self.is_conv(self.dim, m):
                    n = np.prod(m.kernel_size) * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif self.is_norm(self.dim, m):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            # pass

    #
    # def freeze_bn(self):
    #     '''Freeze BatchNorm layers.'''
    #     for layer in self.modules():
    #         if isinstance(layer, BatchNormNd(self.dim)):
    #             layer.eval()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        if self.downsample == 4:
            x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
                # print(i, x.shape)
        return outs


if __name__ == '__main__':
    import torch


    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    init_seed(666)

    r = ResNet(2, 50, downsample=4, in_channels=3)
    print(r)

    r.print_model_params()
    data = torch.ones((1, 3, 128, 128))
    outs = r(data)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))

    # from meddet.runner.checkpoint import load_checkpoint

    # load_checkpoint(r, 'https://download.pytorch.org/models/resnet18-5c106cde.pth')
    # load_checkpoint(r, 'https://download.pytorch.org/models/resnet50-19c8e357.pth')
