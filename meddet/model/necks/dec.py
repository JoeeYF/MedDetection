

from typing import Union, List
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F

from meddet.model.nnModules import ComponentModule, BlockModule
from meddet.model.nd import ConvTransposeNd, ConvNd
from meddet.model.blocks import ConvNormAct, ConvTransposeNormAct


class ConcatLayer(BlockModule):
    S_MODE = False

    def __init__(self, dim, deeper_channels, lower_channels, out_channels, stride, num_blocks=2, groups=1,
                 base_width=64, dilation=1):
        super().__init__()
        self.dim = dim
        self.mid_channels = lower_channels if self.S_MODE else deeper_channels
        self.deeper_conv = ConvTransposeNormAct(dim, deeper_channels, self.mid_channels, kernel_size=stride, stride=stride)

        # mode = 'trilinear' if dim == 3 else 'bilinear'
        # self.deeper_conv = nn.Sequential(
        #     nn.Upsample(scale_factor=stride, mode=mode, align_corners=False),
        #     ConvNormAct(dim, deeper_channels, lower_channels, kernel_size=3, padding=1)
        # )

        fusion_conv = [ConvNormAct(dim, self.mid_channels + lower_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        for i in range(1, num_blocks):
            fusion_conv.append(
                ConvNormAct(dim, out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        self.fusion_conv = nn.Sequential(*fusion_conv)

    def forward(self, x_deeper, x_lower):
        mid = self.deeper_conv(x_deeper)
        if self.dim == 3:
            # input is CDHW
            diffZ = x_lower.size()[2] - mid.size()[2]
            diffY = x_lower.size()[3] - mid.size()[3]
            diffX = x_lower.size()[4] - mid.size()[4]

            mid = F.pad(mid, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
        elif self.dim == 2:
            diffY = x_lower.size()[2] - mid.size()[2]
            diffX = x_lower.size()[3] - mid.size()[3]

            mid = F.pad(mid, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        else:
            raise NotImplementedError
        output = self.fusion_conv(torch.cat([mid, x_lower], dim=1))
        return output


class SConcatLayer(ConcatLayer):
    S_MODE = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class AddLayer(BlockModule):
    def __init__(self, dim, deeper_channels, lower_channels, out_channels, stride, num_blocks=1, groups=1,
                 base_width=64,
                 dilation=1):
        super().__init__()
        self.dim = dim
        if deeper_channels != out_channels:
            self.deeper_conv = nn.Sequential(
                ConvNd(dim)(deeper_channels, out_channels, kernel_size=1, stride=1),
                nn.Upsample(scale_factor=stride, mode='nearest'),
            )
        else:
            self.deeper_conv = nn.Upsample(scale_factor=stride, mode='nearest')
        self.lower_conv = ConvNd(dim)(lower_channels, out_channels, kernel_size=1, stride=1)

        # mode = 'trilinear' if dim == 3 else 'bilinear'
        # self.up_conv = nn.Sequential(
        #     nn.Upsample(scale_factor=stride, mode=mode, align_corners=False),
        #     ConvNormAct(dim, in_channels, out_channels, kernel_size=3, padding=1)
        # )

        fusion_conv = [ConvNd(dim)(out_channels, out_channels, kernel_size=3, stride=1, padding=1)]
        for i in range(1, num_blocks):
            fusion_conv.append(
                ConvNormAct(dim, out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        self.fusion_conv = nn.Sequential(*fusion_conv)

    def forward(self, x_deeper, x_lower):
        x_deeper = self.deeper_conv(x_deeper)
        x_lower = self.lower_conv(x_lower)
        if self.dim == 3:
            # input is CDHW
            diffZ = x_lower.size()[2] - x_deeper.size()[2]
            diffY = x_lower.size()[3] - x_deeper.size()[3]
            diffX = x_lower.size()[4] - x_deeper.size()[4]

            x_deeper = F.pad(x_deeper, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2,
                                        diffZ // 2, diffZ - diffZ // 2])
        elif self.dim == 2:
            diffY = x_lower.size()[2] - x_deeper.size()[2]
            diffX = x_lower.size()[3] - x_deeper.size()[3]

            x_deeper = F.pad(x_deeper, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        else:
            raise NotImplementedError

        output = self.fusion_conv(torch.add(x_deeper, x_lower))
        return output


class Decoder(ComponentModule):
    LAYERS = {
        'concat': ConcatLayer,
        'sconcat': SConcatLayer,
        'add':    AddLayer,
    }

    def __init__(self,
                 dim: int,
                 in_channels: Union[list, tuple] = (16, 32, 64, 128),
                 out_channels: Union[list, tuple] = (16, 32, 64, 128),
                 out_indices: Union[list, tuple] = (0, 1, 2, 3),
                 strides=(1, 2, 2, 2),
                 layer_type: str = 'concat',
                 upsample=1):
        super(Decoder, self).__init__()
        assert isinstance(in_channels, (list, tuple)), 'in_channels must be a list/tuple'
        assert isinstance(out_indices, (list, tuple)), \
            'out_indices must be a list/tuple but get a {}'.format(type(out_indices))
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.out_indices = out_indices
        self.upsample = upsample

        self.layer_type = layer_type
        self.layer = self.LAYERS[self.layer_type]
        self.conv_last = self.layer_type == 'add' or self.in_channels[-1] != self.out_channels[-1]

        if self.conv_last:
            self.conv1 = ConvNd(self.dim)(self.in_channels[-1],
                                          self.out_channels[-1],
                                          kernel_size=1,
                                          stride=1,
                                          padding=0)
            self.conv2 = ConvNd(self.dim)(self.out_channels[-1],
                                          self.out_channels[-1],
                                          kernel_size=3,
                                          stride=1,
                                          padding=1)

        self.layers = self.init_layers()
        self.init_weights()

    def init_layers(self):
        layers = []
        for i in range(len(self.in_channels) - 1, 0, -1):
            layer_name = f'layer{i + 1}+{i}'
            layer = self.layer(self.dim,
                               deeper_channels=self.out_channels[i],
                               lower_channels=self.in_channels[i - 1],
                               out_channels=self.out_channels[i - 1],
                               stride=self.strides[i])
            self.add_module(layer_name, layer)
            layers.append(layer)

        return layers

    def init_weights(self, bn_std=0.02):
        for m in self.modules():
            if self.is_conv(self.dim, m):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif self.is_norm(self.dim, m):
                nn.init.normal_(m.weight, 1.0, bn_std)
                m.bias.data.zero_()

    def forward(self, inputs):
        assert len(self.in_channels) == len(inputs)

        outs_up = [inputs[-1]]

        if self.conv_last:
            outs_up[0] = self.conv1(outs_up[0])

        for i in range(len(self.in_channels) - 1):
            x_deeper, x_lower = outs_up[-1], inputs[-i - 2]
            y = self.layers[i](x_deeper, x_lower)
            outs_up.append(y)

        if self.conv_last:
            outs_up[0] = self.conv2(outs_up[0])

        outs_up.reverse()

        outs = []
        for i in self.out_indices:
            outs.append(outs_up[i])
        return outs


if __name__ == "__main__":
    import torch


    def init_seed(SEED):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    init_seed(666)

    # FPN = Decoder(
    #     dim=2,
    #     in_channels=(16, 32, 64, 128),
    #     out_channels=(64, 64, 64, 64),
    #     strides=(1, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='add')
    # model = FPN

    # UNet = Decoder(
    #     dim=2,
    #     in_channels=(16, 32, 64, 128),
    #     out_channels=(16, 32, 64, 128),
    #     strides=(1, 2, 2, 2),
    #     out_indices=(0, 1, 2, 3),
    #     layer_type='concat')
    # model = UNet

    VNet = Decoder(
        dim=3,
        in_channels=(16, 32, 64, 128),
        out_channels=(32, 64, 128, 128),
        strides=(1, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        layer_type='concat')
    model = VNet

    print(model)
    model.print_model_params()

    inputs = [
        torch.ones((1, 16, 32, 32, 32)),
        torch.ones((1, 32, 16, 16, 16)),
        torch.ones((1, 64, 8, 8, 8)),
        torch.ones((1, 128, 4, 4, 4)),
    ]

    outs = model(inputs)
    for o in outs:
        print(o.shape)
        print(torch.sum(o))
