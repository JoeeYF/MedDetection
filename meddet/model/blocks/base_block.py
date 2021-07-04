

import torch
import torch.nn as nn

from meddet.model.nd import ConvNd, ConvTransposeNd
from meddet.model.nnModules import BlockModule


class ConvNormAct(BlockModule):
    """ classic combination: conv+ normalization [+ relu] post-activation mode """

    def __init__(self, dim, in_channels, out_channels, kernel_size, padding=0, stride=1, do_act=True, groups=1,
                 dilation=1, bias=True):
        super().__init__()
        self.dim = dim
        self.conv = self.build_conv(dim,
                                    in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias)
        self.norm = self.build_norm(dim, out_channels)
        self.do_act = do_act
        if do_act:
            self.act = self.build_act()

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.do_act:
            out = self.act(out)
        return out


class VResConvNormAct(ConvNormAct):
    def __init__(self, *args, **kwargs):
        kwargs['stride'] = 1
        kwargs['padding'] = 1
        kwargs['do_act'] = True
        super().__init__(*args, **kwargs)

    def forward(self, x):
        out = self.norm(self.conv(x)) + x
        out = self.act(out)
        return out


class ConvTransposeNormAct(BlockModule):
    """ classic combination: conv+ normalization [+ relu] post-activation mode """

    def __init__(self, dim, in_channels, out_channels, kernel_size, padding=0, stride=1, do_act=True, groups=1,
                 dilation=1, bias=True):
        super().__init__()
        self.dim = dim
        self.conv = ConvTransposeNd(dim)(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.norm = self.build_norm(dim, out_channels)
        self.do_act = do_act
        if do_act:
            self.act = self.build_act()

    def forward(self, x):
        out = self.norm(self.conv(x))
        if self.do_act:
            out = self.act(out)
        return out
