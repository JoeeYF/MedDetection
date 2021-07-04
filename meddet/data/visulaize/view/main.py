import os
import pickle
import sys

from meddet.data import ImageIO
from meddet.data.visulaize.vtktools import vtkShow
from meddet.data.visulaize.vtkViewClass import vtkWindowView

data, dim, spacing, origin = ImageIO.loadArray(sys.argv[1])
print(sys.argv[1])
print(data.shape)
mode = 2
if int(mode) == 1:
    # loaded_data = pickle.load(open(os.path.dirname(__file__) + '/tmp.pkl', 'rb'))
    # data, spacing = loaded_data['data'], loaded_data['spacing']
    vtkShow(data[0], spacing)
elif int(mode) == 2:
    # loaded_data = pickle.load(open(os.path.dirname(__file__) + '/tmp.pkl', 'rb'))
    # data, spacing = loaded_data['data'], loaded_data['spacing']
    vtkWindowView(data[0], spacing=spacing)