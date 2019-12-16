# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:09:09 2019

@author: rum
"""

import unittest
from HirachicalHomVoxMod import  hom_voxel
import numpy as np

bins=(1,2,2)
upper=np.array([1,2,2])
lower=np.array([-0,-0,-0])

mod=hom_voxel(bins,np.stack((lower,upper),axis=0))

starts=np.array([0,0,0])
ends=np.array([1,1,1])

pt=np.array([0.2       , 0.06368546, 1.        ])

bin_size=np.array([.5,.5,.5])

def get_index(pt):
    return np.floor(((pt-starts)/bin_size)).astype(int)