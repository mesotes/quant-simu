# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:46:44 2019

@author: rum
"""

import numpy as np

import matplotlib.pyplot as plt
import math
from math import sqrt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle



def PlotVoxel(boxes):
    
    x_min=y_min=-10
    x_max=y_max=10
    
    edgecolor='Black'
    facecolor='None'
    alpha=0.5
    
    fig, ax = plt.subplots()
    plt.axis('equal')
        
    x_center=y_center=0
    width=hight=1
    
    errorboxes = boxes
    
        # Loop over data points; create box from errors at each point
    
    rect = Rectangle((x_center-width, y_center-hight), width, hight,linestyle="-")
    errorboxes.append(rect)
    
        # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,edgecolor=edgecolor)
    
        # Add collection to axes
    ax.add_collection(pc)
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min,y_max)
    
    plt.show()