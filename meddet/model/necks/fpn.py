import torch.nn as nn
import torch.nn.functional as F

from meddet.model.utils import xavier_init
from meddet.model.nnModules import ComponentModule
from meddet.model.nd import ConvNd, BatchNormNd


class FPN(ComponentModule):

    def __init__(self,
                 dim,
                 in_channels: (list, tuple),
                 out_channels: int,
                 num_outs,
                 out_indices,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False
                 ):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.out_indices = out_indices

        self.start_level = start_level
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.backbone_end_level - self.start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.relu_before_extra_convs = relu_before_extra_convs

        # self.extra_convs_on_inputs = True
        # self.add_extra_convs = True
        # self.relu_before_extra_convs = True

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvNd(self.dim)(in_channels[i], out_channels, kernel_size=1, stride=1, padding=0)
            fpn_conv = ConvNd(self.dim)(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvNd(self.dim)(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                self.fpn_convs.append(extra_fpn_conv)

        self.init_weights()

    def init_weights(self):
        import math
        for m in self.modules():
            if self.is_conv(self.dim, m):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif self.is_norm(self.dim, m):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # [print('laterals', i, l.shape) for i, l in enumerate(laterals)]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # print(f'add upsample laterals_{i} to laterals_{i-1}')
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # [print('fpns', i, o.shape) for i, o in enumerate(outs)]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # print('add extra levels')
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                # default extra_convs_on_inputs is True
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                # print('extra', len(outs)-1, outs[-1].shape)
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
                    # print('more', len(outs) - 1, outs[-1].shape)

        return tuple([outs[i] for i in self.out_indices])


if __name__ == '__main__':
    import torch

    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    init_seed(666)

    FPN = FPN(
        dim=2,
        in_channels=[16, 32, 64, 128],
        out_channels=64,
        num_outs=4,
        out_indices=(0, 1, 2, 3),)
    model = FPN

    # UNet = UNetDecoder(
    #     dim=2,
    #     in_channels=(16, 32, 64, 128),
    #     out_channels=(16, 32, 64, 128),
    #     strides=(1, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='concat')
    # model = UNet

    print(model)
    model.print_model_params()

    inputs = [
        torch.ones((1, 16, 32, 32)),
        torch.ones((1, 32, 16, 16)),
        torch.ones((1, 64, 8, 8)),
        torch.ones((1, 128, 4, 4)),
    ]

    outs = model(inputs)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))