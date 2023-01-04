
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import scipy.io as io
import numpy as np
from multiply_by_H_band import multiply_by_H
from scipy.io import savemat

class projection:
    def __init__(self, minvalue, maxvalue, proj_flag):
        self.min = minvalue
        self.max = maxvalue
        self.flag = proj_flag


    def projection_func(self, x):
        x[x>self.max] = self.max
        x[x<self.min] = self.min
        y = x
        return y