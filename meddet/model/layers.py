

# -*- coding:utf-8 -*-
# reference : https://github.com/charleshouston/unet-pytorch/blob/master/models/custom_layers.py

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
from torch.nn.modules.utils import _ntuple
from typing import Union, Tuple

from torch.nn.modules.padding import ReflectionPad2d as OriReflectionPad2d


def prepare_onnx_paddings(dim, pad):
    assert isinstance(dim, int)
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    assert len(pad) <= dim * 2
    # assume zero-dimensions in the beginning
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    assert len(paddings) == dim * 2
    return paddings


def flip(x: Variable, dim: int) -> Variable:
    """Flip torch Variable along given dimension axis."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous().view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
        getattr(torch.arange(x.size(1) - 1, -1, -1),
                ('cpu', 'cuda')[x.is_cuda])().long(), :]
    x = x.view(xsize)
    return x


class ReflectionPad2d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""

    def __init__(self, padding: Union[int, Tuple[int]]):
        super(ReflectionPad2d, self).__init__()
        self.padding = _ntuple(4)(padding)

    def forward(self, input: Variable) -> Variable:
        return ReflectionPadNd.apply(input, self.padding)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + str(self.padding) + ')'


class ReflectionPad3d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""

    def __init__(self, padding: Union[int, Tuple[int]]):
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input: Variable) -> Variable:
        return ReflectionPadNd.apply(input, self.padding)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + str(self.padding) + ')'


class ReflectionPadNd(Function):
    """Padding for same convolutional layer."""

    @staticmethod
    def symbolic(g, input: Variable, padding: Union[int, Tuple[int]]):
        paddings = prepare_onnx_paddings(len(input.type().sizes()), padding)
        return g.op("Pad", input, pads_i=paddings, mode_s="reflect")

    @staticmethod
    def forward(ctx: Function, input: Variable, pad: Tuple[int]) -> Variable:
        ctx.pad = pad
        ctx.input_size = input.size()
        ctx.l_inp = len(input.size())
        ctx.pad_tup = tuple([(a, b)
                             for a, b in zip(pad[:-1:2], pad[1::2])]
                            [::-1])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i])
                         for i, d in enumerate(input.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # Create output tensor by concatenating with reflected chunks.
        output = input.new(input.size()[:(ctx.l_diff)] + new_dim).zero_()
        c_input = input

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                chunk1 = flip(c_input.narrow(i, 1, pad[0]), i)
                c_input = torch.cat((chunk1, c_input), i)
            if p[1] > 0:
                chunk2 = flip(c_input.narrow(i, c_input.shape[i] - p[1] - 1, p[1]), i)
                c_input = torch.cat((c_input, chunk2), i)
        output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx: Function, grad_output: Variable) -> Variable:
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x, ) for x in ctx.input_size]

        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0],
                                             cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0,
                                             cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None


if __name__ == "__main__":
    x = torch.nn.Parameter(torch.arange(9).reshape(1, 1, 3, 3).float())
    print(x)

    y = ReflectionPad2d(1)(x)
    print(y.shape)
    print(y)

    y2 = OriReflectionPad2d(1)(x)
    print(y2.shape)
    print(y2)

    assert torch.all(y == y2)

    c = nn.Conv2d(1, 1, 3)
    loss = nn.MSELoss()(c(y), torch.ones_like(x))
    print(loss)
    loss2 = nn.MSELoss()(c(y2), torch.ones_like(x))
    print(loss2)

    loss.backward()
    loss2.backward()

    # x = torch.arange(27).reshape(1, 1, 3, 3, 3)
    # y = ReflectionPad3d(1)(x)
    # print(x)
    # print(y)
