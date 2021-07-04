# -*- coding:utf-8 -*-
import os
import re
import numpy as np
from glob import glob
import argparse

'''
python max_froc.py --config folds_medtk/cfg_sub_model_3_nodule_det_rcnn_v_base_5_roi_0.01 --fold 9
'''

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='folds_medtk/faster_rcnn_deeplung_coord1')
parser.add_argument('--fold', type=int, default=9)
args = parser.parse_args()

if not args.config:
    quit(-1)

SUFFIX = 'medtk'
FOLD = args.fold
model_dirname = os.path.splitext(os.path.basename(args.config))[0]  # 'cfg_sub_model_3_nodule_det_rcnn_v_base_5_roi_0.01'
print(model_dirname)

cur_dir = os.path.abspath(os.path.dirname(__file__))
output_dir = '{}/folds_{}/{}/{}/*/CADAnalysis.txt'.format(cur_dir, SUFFIX, model_dirname, FOLD)
print(output_dir)


frocs = {}
senss = {}
for i in sorted(glob(output_dir)):
    with open(i, 'r') as f:
        epoch = int(os.path.abspath(i).split(os.path.sep)[-2][:-2])
        content = ''.join(f.readlines())
        froc = eval(re.findall('Average Precision: ([\d\.]+)', content)[0])
        sens = eval(re.findall('Sensitivity: ([\d\.]+)', content)[0])
        frocs[epoch] = froc
        senss[epoch] = sens

print(f'froc 150ep {frocs[150]:.08f}')
print('max', [f'{v:.08f} @ {k}' for k, v in frocs.items() if v == np.max(list(frocs.values()))][0])
print(f'mean froc {np.mean(list(frocs.values())):.08f}')
print('')
print('Average Precision')
print('\n'.join([f"{k: 4d}ep-{frocs[k]:.08f}" for k in sorted(frocs.keys())]))
print('Sensitivity')
print('\n'.join([f"{k: 4d}ep-{senss[k]:.08f}" for k in sorted(senss.keys())]))
