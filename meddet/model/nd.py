

from torch import nn
from .layers import ReflectionPad2d, ReflectionPad3d
from medvision.ops import DeformConv3dPack,DeformConv2dPack,ModulatedDeformConv2dPack,ModulatedDeformConv3dPack

def ConvNd(dim):
    if dim == 3:
        ConvNd = nn.Conv3d
    elif dim == 2:
        ConvNd = nn.Conv2d
    else:
        raise NotImplementedError
    return ConvNd

def DCNv1Nd(dim):
    if dim == 3:
        ConvNd = DeformConv3dPack
    elif dim == 2:
        ConvNd = DeformConv2dPack
    else:
        raise NotImplementedError
    return ConvNd

def DCNv2Nd(dim):
    if dim == 3:
        ConvNd = ModulatedDeformConv3dPack
    elif dim == 2:
        ConvNd = ModulatedDeformConv2dPack
    else:
        raise NotImplementedError
    return ConvNd

def BatchNormNd(dim):
    if dim == 3:
        BatchNormNd = nn.BatchNorm3d
    elif dim == 2:
        BatchNormNd = nn.BatchNorm2d
    else:
        raise NotImplementedError
    return BatchNormNd


def GroupNormNd(dim):
    if dim == 3:
        GroupNormNd = nn.GroupNorm
    elif dim == 2:
        GroupNormNd = nn.GroupNorm
    else:
        raise NotImplementedError
    return GroupNormNd


# def InstanceNormNd(dim):
#     if dim == 3:
#         InstanceNormNd = nn.InstanceNorm3d
#     elif dim == 2:
#         InstanceNormNd = nn.InstanceNorm2d
#     else:
#         raise NotImplementedError
#     return InstanceNormNd
#
#
# def LayerNormNd(dim):
#     if dim == 3:
#         LayerNormNd = nn.LayerNorm
#     elif dim == 2:
#         LayerNormNd = nn.LayerNorm
#     else:
#         raise NotImplementedError
#     return LayerNormNd


def MaxPoolNd(dim):
    if dim == 3:
        MaxPoolNd = nn.MaxPool3d
    elif dim == 2:
        MaxPoolNd = nn.MaxPool2d
    else:
        raise NotImplementedError
    return MaxPoolNd


def AvgPoolNd(dim):
    if dim == 3:
        AvgPoolNd = nn.AvgPool3d
    elif dim == 2:
        AvgPoolNd = nn.AvgPool2d
    else:
        raise NotImplementedError
    return AvgPoolNd


def AdaptiveAvgPoolNd(dim):

    if dim == 3:
        AdaptiveAvgPoolNd = nn.AdaptiveAvgPool3d
    elif dim == 2:
        AdaptiveAvgPoolNd = nn.AdaptiveAvgPool2d
    else:
        raise NotImplementedError
    return AdaptiveAvgPoolNd


def ConvTransposeNd(dim):
    if dim == 3:
        ConvTransposeNd = nn.ConvTranspose3d
    elif dim == 2:
        ConvTransposeNd = nn.ConvTranspose2d
    else:
        raise NotImplementedError
    return ConvTransposeNd


def DropoutNd(dim):
    if dim == 3:
        DropoutNd = nn.Dropout3d
    elif dim == 2:
        DropoutNd = nn.Dropout2d
    else:
        raise NotImplementedError
    return DropoutNd