import numpy as np
import matplotlib as plt
import math
from math import sqrt
 
import matplotlib.pyplot as pyplt

"""
Center of barrel is at (0,0,0) similar to root sim

rectangular box, voxel outside of volume are defined i.e. one can loop till one reach the edge
outside voxels mu is set to 0

important: grids contain the center coordinate of voxel

"""       

class hom_voxel:
    def __init__(self):
        self.R_Barrel=30     
        #self.bins=np.array([40,40,80])
        self.starts=np.array([-30.0,-30.0,-40.0])
        self.ends=np.array([30.0,30.0,40.0])
        self.bin_size=np.zeros((3,))
        self.num_bins=np.array([60,60,80])
        self.mu_data=np.zeros((80,60,60))
        self.calc_bin_size()
        
        # variables for tracking
        self.boarders=np.zeros((2,3))
        self.next_pt=np.zeros((3,))
        self.next_box=np.zeros((3,))
        self.norm_vec=np.zeros((3,))
        
    def calc_bin_size(self):
        self.bin_size=(self.ends-self.starts)/self.num_bins

    def GetIndex(self,pt):
        self.next_box=((pt-self.starts)/self.bin_size).astype(int)
        return  ((pt-self.starts)/self.bin_size).astype(int)

    def get_boarders(self,pt): # fine for points within box 
        bin_coord=self.GetIndex(pt)
        left= bin_coord*self.bin_size+self.starts+self.bin_size/2
        right= bin_coord*self.bin_size-self.bin_size/2+self.starts
        self.boarders=np.stack((left,right),axis=0)
        return np.stack((left,right),axis=0)
    
    def get_boarders_next_box(self):
        bin_coord=self.next_box
        left= bin_coord*self.bin_size+self.starts+self.bin_size/2
        right= bin_coord*self.bin_size-self.bin_size/2+self.starts
        self.boarders=np.stack((left,right),axis=0)
        return np.stack((left,right),axis=0)
    
    def calc_intercept(self,pt,norm):
        """
        y=norm*lambda+pt
        lambda>0(!) correct direction
        (left)=(left,low,bottom)
        """
        boarders=self.get_boarders_next_box() #get_boarders(pt)
        lambdas=(boarders-pt)/norm
        return np.amin(lambdas[np.where(lambdas>0)])
                
    def GetNextBox(self):
        edge_check=(self.next_pt==self.boarders)
        # [0]--> Dekrement [1]-> Increment
        inc=np.where(edge_check[0]==True)
        dec=np.where(edge_check[1]==True)
        #print(inc,dec)
        #old=self.next_box
        self.next_box[inc]=self.next_box[inc]+1
        self.next_box[dec]=self.next_box[dec]-1
        #update=self.next_box
        #print(update-old)
        #return old,update
        
    def next_point(self,pt,norm):
        lamb=self.calc_intercept(pt,norm)
        self.next_pt=pt + norm*lamb
        #print(lamb)
        return self.next_pt
        
    def gen_voxels(self,nx,ny,nz):
                
        x_grid=np.linspace(x_start,x_end,nx)
        y_grid=np.linspace(y_start,y_end,ny)
        z_grid=np.linspace(z_start,z_end,nz)
        
    
"""
next_box contains idx for current box

GetIndex
GetNextBox() (given the next(!) point)
 --> Checks where next pt sits on current box boarder 
 --> increments index on those edges (can be one two or even 3 )
"""
mod=hom_voxel()

pt=np.array([0,4,-10])
phi=2*math.pi*(342/360) # 30 deg

nv=np.array([0,math.cos(phi),math.sin(phi)])

out_glob_box=False

x=np.empty((0,))
y=np.empty((0,))

mod.next_box=mod.GetIndex(pt) # initialiseren 

while out_glob_box==False:
    x=np.append(x,[pt[1]])
    y=np.append(y,[pt[2]])
    idx=mod.next_box  
    out_glob_box=((idx<0).any() or (idx>=mod.num_bins).any())
    mod.GetNextBox()
    pt=mod.next_point(pt,nv)
    


ticks=np.zeros((64,))

for i in range(0,64):
    ticks[i]=-32.5+i

mod.GetIndex(pt)

fig, ax = pyplt.subplots()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
#ax.yaxis.grid(True, which='major')
#ax.yaxis.grid(True, which='minor')

pyplt.scatter(x, y, alpha=0.5)
pyplt.grid(True,which="both")
pyplt.show()




