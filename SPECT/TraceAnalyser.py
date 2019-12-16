# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 08:40:38 2019

@author: rum
"""

from math import pi
import math
import numpy as np
import matplotlib 
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
from matplotlib.patches import Circle, PathPatch
from scipy.stats import chi2
import scipy
import matplotlib.colors as colors
from matplotlib.patches import Circle, PathPatch




"""Class to analyse the matrices from MCMC runs """
class TraceAnalyser:
    def __init__(self):
        self.length=0
        self.bins_x=0
        self.bins_y=0
        self.mat_trace
        
    def set_tract(self,trace):
        """ Assuming np_array of shape (length,binsx,binsy)"""
        self.mat_trace=trace
        shape=self.mat_trace.shape
        self.length=shape[0]
        self.bins_x=shape[1]
        self.bins_y=shape[2]
        print("Trace with ", self.length," Entries and ",self.bins_x," x ",self.bins_y," Bins")

    def show_single(self,i):
        if i<0 and i>=self.length:
            print("wrong index")
            return 0
        




if __name__ == '__main__':      
        
    observed=datfile['data']
    bins_x=bins_y=50
    m=model_gen()
    m.set_observation(observed)
    m.make_sens_mats(bins_x,bins_y)
    m.make_mats()
    
    m.gen_upper_limit()
    z=m.upper_lim
    
    m.make_plot()  
    