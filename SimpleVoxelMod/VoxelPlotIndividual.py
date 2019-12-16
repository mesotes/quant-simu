# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:57:37 2019

@author: rum
"""

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def cuboid_data(o, size=(3,.2,.3)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:,:,i] *= size[i]
    X += np.array(o)
    return X

def plotCubeAt(positions,sizes=None,colors=None, **kwargs):
    if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
    if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
    g = []
    for p,s,c in zip(positions,sizes,colors):
        g.append( cuboid_data(p) )
    return Poly3DCollection(np.concatenate(g),  
                            facecolors=np.repeat(colors,6, axis=0), **kwargs)

N1 = 10
N2 = 10
N3 = 10
ma = np.random.choice([0,1], size=(N1,N2,N3), p=[0.99, 0.01])#
x,y,z = np.indices((N1,N2,N3))-.5

colors= np.random.rand(len(positions),3)

positions = np.c_[x[ma==1],y[ma==1],z[ma==1]]

fig = plt.figure()
ax = fig.gca(projection='3d')
 

pc = plotCubeAt(positions, colors=colors,edgecolor="k")
ax.add_collection3d(pc)

ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_zlim([0,10])
#plotMatrix(ax, ma)
#ax.voxels(ma, edgecolor="k")
