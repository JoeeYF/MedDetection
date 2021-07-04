

import os

CONV_configs = {
    'Conv': dict(
        type='Conv',
        padding_mode='zeros'
    )
}
NORM_configs = {
    'BatchNorm': dict(
        type='BatchNorm',
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True),
    'GroupNorm': dict(
        type='GroupNorm',
        num_groups=-1,  # 1=Layer Norm, -1=Instance Norm, other=groups
        eps=1e-05,
        affine=True),
}
ACT_configs = {
    'ReLU':      dict(type='ReLU', inplace=True),
    'LeakyReLU': dict(type='LeakyReLU', negative_slope=1e-2, inplace=True),
    'ELU':       dict(type='ELU', alpha=1., inplace=True),
    'PReLU':     dict(type='PReLU', num_parameters=1, init=0.25),
}

# ----- #
DATASETS = '/././Datasets/'
ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'work_dirs')  # .../MedToolkit/work_dirs
FP16 = True  # forced
K_FOLD = 5
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

CONV_CFG = CONV_configs['Conv']
NORM_CFG = NORM_configs['BatchNorm']
ACT_CFG = ACT_configs['ReLU']
DIM = 3


print(f'Default configs '
      f'\nConv{DIM}d: {CONV_CFG}'
      f'\nAct{DIM}d : {ACT_CFG}'
      f'\nNorm{DIM}d: {NORM_CFG}')
