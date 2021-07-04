import os

os.environ.setdefault('DIM', '3')
import torch
from torch import nn
from inspect import getframeinfo, stack
from collections.abc import Iterable
from datetime import datetime

from meddet import setting
from meddet.data import ImageIO
from meddet.model.nd import BatchNormNd, GroupNormNd, ConvNd, DCNv1Nd, DCNv2Nd
from meddet.model.layers import ReflectionPad3d


class TensorSaver:
    def __init__(self):
        self.root = os.path.join(setting.ROOT, 'tensor')
        self.idx = 0

    def __call__(self, obj, name):
        self.idx = 0
        self.save_tensor(obj, name)

    def save_tensor(self, obj, name):
        if isinstance(obj, torch.Tensor):
            ImageIO.saveArray(self.root + f'/{name}_{self.idx}.nii.gz', obj[0, ...].cpu().numpy())
            self.idx += 1
        else:
            if isinstance(obj, (list, tuple)):
                [self.save_tensor(o, name) for i, o in enumerate(obj)]


def fun(module: nn.Module, x, y):
    # print(1, module)
    is_NetModule = [m.__class__.__name__ for m in list(module.modules())[1:] if
                    issubclass(m.__class__, ComponentModule)]
    # print(is_NetModule)
    s = TensorSaver()
    if True:
        if hasattr(module, 'hook_enable'):
            if module.hook_enable:
                # print('in')
                # if isinstance(x, torch.Tensor):
                #     print('!!!!!!!!!!!!!!')
                # print_tensor(x)
                s(x, module.__class__.__name__ + 'In')
                # print('out')
                # print_tensor(y)
                s(y, module.__class__.__name__ + 'Out')
    # print('\n')


class BlockModule(nn.Module):
    def __init__(self, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.log_enable = False
        # self.dim = eval((os.environ.get('DIM', '2')))
        # self.conv_cfg = setting.CONV_configs[os.environ.get('CONV_CFG', 'Conv')]
        # self.norm_cfg = setting.NORM_configs[os.environ.get('NORM_CFG', 'BatchNorm')]
        # self.act_cfg = setting.ACT_configs[os.environ.get('ACT_CFG','ReLU')]
        self.conv_cfg = setting.CONV_CFG if conv_cfg is None else conv_cfg
        self.norm_cfg = setting.NORM_CFG if norm_cfg is None else norm_cfg
        self.act_cfg = setting.ACT_CFG if act_cfg is None else act_cfg

    def is_conv(self, dim, obj):
        conv_type = self.conv_cfg['type']
        if conv_type == 'Conv':
            conv_layer = ConvNd(dim)
        else:
            raise NotImplementedError
        return isinstance(obj, conv_layer)

    def is_norm(self, dim, obj):
        norm_type = self.norm_cfg['type']
        if norm_type == 'BatchNorm':
            norm_layer = BatchNormNd(dim)
        elif norm_type == 'GroupNorm':
            norm_layer = GroupNormNd(dim)
        else:
            raise NotImplementedError
        return isinstance(obj, norm_layer)

    def build_conv(self, dim, in_channels, out_channels, **kwargs):
        conv_cfg = self.conv_cfg.copy()
        conv_type = conv_cfg.pop('type')
        for k, v in kwargs.items():
            conv_cfg[k] = v
        if conv_type == 'Conv':
            if conv_cfg.get('padding_mode') == 'reflect' and dim == 3:
                padding = conv_cfg.pop('padding') if 'padding' in conv_cfg.keys() else 0
                padding_mode = conv_cfg.pop('padding_mode')
                return nn.Sequential(
                    ReflectionPad3d(padding=padding),
                    ConvNd(dim)(in_channels, out_channels, **conv_cfg)
                )
            else:
                return ConvNd(dim)(in_channels, out_channels, **conv_cfg)
        else:
            raise NotImplementedError

    def build_dcn(self, dim, in_channels, out_channels, **kwargs):
        _dcn_cfg = kwargs.copy()
        dcn_type = _dcn_cfg.pop('type')

        if dcn_type == 'DCNv1':
            return DCNv1Nd(dim)(in_channels, out_channels, **_dcn_cfg)
        elif dcn_type == 'DCNv2':
            return DCNv2Nd(dim)(in_channels, out_channels, **_dcn_cfg)
        else:
            raise NotImplementedError

    def build_norm(self, dim, num_channels):
        norm_cfg = self.norm_cfg.copy()
        norm_type = norm_cfg.pop('type')
        if norm_type == 'BatchNorm':
            return BatchNormNd(dim)(num_channels, **norm_cfg)
        elif norm_type == 'GroupNorm':
            num_groups = norm_cfg.pop('num_groups')
            if num_groups == -1:
                num_groups = num_channels
            return GroupNormNd(dim)(num_groups, num_channels, **norm_cfg)

    def build_act(self, **kwargs):
        act_cfg = self.act_cfg.copy()
        act_type = act_cfg.pop('type')
        for k, v in kwargs.items():
            act_cfg[k] = v
        if act_type == 'ReLU':
            return nn.ReLU(**act_cfg)
        elif act_type == 'LeakyReLU':
            return nn.LeakyReLU(**act_cfg)
        elif act_type == 'ELU':
            return nn.ELU(**act_cfg)
        elif act_type == 'PReLU':
            return nn.PReLU(**act_cfg)
        else:
            raise NotImplementedError

    def get(self, k, default=None):
        try:
            return self.__getattribute__(k)
        except:
            return default

    @property
    def log(self):
        return self.log_enable

    def setLog(self):
        self.log_enable = True
        for name, m in self._modules.items():
            if isinstance(m, Iterable):
                for mm in m:
                    if hasattr(mm, 'setLog'):
                        mm.setLog()
            else:
                if hasattr(m, 'setLog'):
                    m.setLog()

    def resetLog(self):
        self.log_enable = False
        for name, m in self._modules.items():
            if isinstance(m, Iterable):
                for mm in m:
                    if hasattr(mm, 'resetLog'):
                        mm.resetLog()
            else:
                if hasattr(m, 'resetLog'):
                    m.resetLog()

    def try_to_info(self, *args):
        if self.log_enable:
            caller = getframeinfo(stack()[1][0])
            print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                  "- {} - line {} - {} :".format(self.__class__.__name__, caller.lineno, caller.function), end='')
            self.print(*args)

    def info(self, *args):
        caller = getframeinfo(stack()[1][0])
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
              "- {} - line {} - {} :".format(self.__class__.__name__, caller.lineno, caller.function), end='')
        self.print(*args)

    @staticmethod
    def print(*args):
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 1:
                print('\n\t', end='')
            print(arg, end=' ')
        print('')


class ComponentModule(BlockModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hook_enable = False
        self.handle = self.register_forward_hook(fun)

    @property
    def hook(self):
        return self.hook_enable

    def setHook(self):
        self.hook_enable = True
        for name, m in self._modules.items():
            if isinstance(m, Iterable):
                for mm in m:
                    if hasattr(mm, 'setHook'):
                        mm.setHook()
            else:
                if hasattr(m, 'setHook'):
                    m.setHook()

    def resetHook(self):
        self.hook_enable = False
        for name, m in self._modules.items():
            if isinstance(m, Iterable):
                for mm in m:
                    if hasattr(mm, 'resetHook'):
                        mm.resetHook()
            else:
                if hasattr(m, 'resetHook'):
                    m.resetHook()

    def print_model_params(self):
        print(self.__class__.__name__)

        for k, p in self.named_parameters():
            print(k, p.numel())

        total_params = sum(p.numel() for p in self.parameters())
        print('Total params：{}'.format(total_params))

        total_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Total trainable parameters：{}'.format(total_trainable_parameters))
        print('\n')


class ModelModule(ComponentModule):
    def __init__(self):
        super().__init__()
        self.handle.remove()


if __name__ == '__main__':
    n = BlockModule(norm_cfg=setting.NORM_configs['GroupNorm'])
    print(n.dim)
    c = n.build_conv(3, 1, 3, kernel_size=3)
    print(c)
    n = n.build_norm(n.dim, 12)
    print(n)
