from os.path import dirname as ospdn
from os.path import abspath as ospap
from os.path import join as ospj
import sys
sys.path.insert(0, ospj(ospdn(ospdn(ospdn(ospap(__file__)))), 'cocoapi/PythonAPI'))

import pycocotools.mask as mask_util
import numpy as np


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if(Polys[i-1]):
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen