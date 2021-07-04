

# -*- coding:utf-8 -*-

import os
import pickle
import sys

from meddet.data.visulaize.vtktools import vtkShow
from meddet.data.visulaize.vtkViewClass import vtkWindowView

# mode = '1'
mode = sys.argv[1]
print(mode)
if int(mode) == 1:
    loaded_data = pickle.load(open(os.path.dirname(__file__) + '/tmp.pkl', 'rb'))
    data, spacing = loaded_data['data'], loaded_data['spacing']
    vtkShow(data, spacing)
elif int(mode) == 2:
    loaded_data = pickle.load(open(os.path.dirname(__file__) + '/tmp.pkl', 'rb'))
    data, spacing = loaded_data['data'], loaded_data['spacing']
    vtkWindowView(data, spacing=spacing)
