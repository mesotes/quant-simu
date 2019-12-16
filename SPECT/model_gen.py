# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:40:42 2019

@author: Stefan Rummel
"""

from math import pi
import math
import numpy as np
#import cupy as np
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

from numba import jit
from numba import jitclass          # import the decorator
from numba import int32, float32    # import the types
from numba import void, int_, double

"""
Calculate upper limits on Acitivity in each bin based on number of photons in each segment and 
the assumption that each voxel contains the complete activity which is compatible with the number of photons detected

"""
spec = [
    ('L', float32),
    ('Rb', float32),    
    ('num_seg', int32),      # a simple scalar field
    ('ActivityMatrice', float32[:]),  
    ('sens_mats', float32[:]),
    ('upper_lim', float32[:]),  
    ('observed', float32[:]) # an array field
]


#@jit
class model_gen:
    #@void(int_, int_)
    def __init__(self,binsx,binsy):
        
        self.nbin_x=binsx
        self.nbin_y=binsy
        
        self.L=80.0
        self.Rb=30.0
        self.num_seg=40
        
        self.r_det=3.05 # sperical detector  
        self.time_per_seg=1800/self.num_seg ## halbe stunde
        self.rho=1.2 # g/cm3
        self.mu=2e-2
        self.rot_mats=np.zeros((self.num_seg,2,2),dtype=np.float32)
        self.angle_arr=np.linspace(0,2*pi/self.num_seg*( self.num_seg-1), self.num_seg,dtype=np.float32)
                        
        self.observed=np.zeros(self.num_seg,dtype=np.float32)
        self.simulated=np.zeros(self.num_seg,dtype=np.float32)
    
        
        self.y_min=self.x_min=-30
        self.y_max=self.x_max=30
        
        self.idx_list=[] # complete list with indices of pt_array
        self.pt_arr=np.zeros((self.nbin_x,self.nbin_y,2,1),dtype=np.float32) # Array with centers of each voxel
        self.upper_lim=np.zeros((self.nbin_x,self.nbin_y),dtype=np.float32)
        self.bin_size=np.array([[(self.x_max-self.x_min)/self.nbin_x],[(self.y_max-self.y_min)/self.nbin_y]],dtype=np.float32)
        self.sens_mats=np.zeros((self.num_seg,self.nbin_x,self.nbin_y),dtype=np.float32)
        
        self.sources=[]
        self.ActivityMatrice=np.zeros((self.nbin_x,self.nbin_y),dtype=np.float32)
        
        #np.cuda.Device(0).use()

        self.make_mats()
   
    def set_observation(self,observed):
        l=observed.shape[0]
        if l==self.num_seg:
            self.observed=observed
        else:
            print("scans do not fit, number of segments (num_seg) differ")
        for i in range(0,self.num_seg):
            if self.observed[i]==0:
                self.observed[i]=5
    
    def make_mats(self):
        for i in range(0,self.num_seg):
            self.rot_mats[i,0,0]=math.cos(self.angle_arr[i])
            self.rot_mats[i,1,0]=math.sin(self.angle_arr[i])
            self.rot_mats[i,1,1]=math.cos(self.angle_arr[i])  
            self.rot_mats[i,0,1]=-math.sin(self.angle_arr[i])
            
        self.pt_arr=np.zeros((self.nbin_x,self.nbin_y,2,1)) # Array with centers of each voxel
        self.upper_lim=np.zeros((self.nbin_x,self.nbin_y))
        self.idx_list=[]
        
        self.ActivityMatrice=np.zeros((self.nbin_x,self.nbin_y))
        
        self.bin_size=np.array([[(self.x_max-self.x_min)/self.nbin_x],[(self.y_max-self.y_min)/self.nbin_y]])
        
        for i in range(0,self.nbin_x):
            for j in range(0,self.nbin_y):
                self.pt_arr[i,j,0,0]=self.x_min+(0.5+i)*(self.x_max-self.x_min)/self.nbin_x
                self.pt_arr[i,j,1,0]=self.y_min+(0.5+j)*(self.y_max-self.y_min)/self.nbin_y
                self.idx_list.append((i,j))
                
        
    
    def make_contour_stuff(self):
        bx2=(self.x_max-self.x_min)/self.nbin_x/2
        by2=(self.y_max-self.y_min)/self.nbin_y/2
        x = np.linspace(self.x_min, self.x_max, self.nbin_x)
        y = np.linspace(self.y_min, self.y_max, self.nbin_y)

        X,Y= np.meshgrid(x, y)
        return X,Y
        
    def coll_func(self,dva):
        p0,p1,p2,p3,p4,p5,p6=1.10718,6.24301,-0.130778, 0.814905,-0.555095,-0.113148,53.1849
            
        if dva>p0:
            if dva<p1:
                return 1.0+p2*(dva-p0)
            else:
                return p6*math.exp(-p3*dva+p4*(p0-dva)+p5*(math.pow((p0-dva),2)))
        else:
            return 1
 
    def pred_algo(self,pt):
    
        xq=pt[0,0]
        yq=pt[1,0]
        
        if math.sqrt(xq*xq+yq*yq)<self.Rb:
            """
            stuff for collimator function
            view=theano.shared(np.array([[1.0],[0.0]])) #direction of line of sight
            det_pos=theano.shared(np.array([[-80],[0.0]]))  # detector position
            #angle=T.arccos(T.dot(pt-det_pos,view)/T.dot(pt-det_pos,pt-det_pos))
            """
            dva=abs((math.acos((xq+self.L)/math.sqrt(math.pow((xq+self.L),2)+yq*yq)))*180/pi)
            """
            Pointsource-Model
            """
            m2=math.pow(yq/(self.L+xq),2) # slope squared
            m2p12=math.pow((m2+1),2) # 
            sgn=-1
            if yq>=0: 
                sgn=1 
            yI=sgn*math.sqrt(self.Rb*self.Rb-math.pow((-self.L*m2/(1+m2)-math.sqrt((m2*self.Rb*self.Rb-m2*self.L*self.L+self.Rb*self.Rb)/m2p12)),2))
            xI=-self.L*m2/(1+m2)-math.sqrt((m2*(self.Rb*self.Rb-self.L*self.L)+self.Rb*self.Rb)/m2p12) ## potential sign but neg is req.
            A=math.sqrt((xI-xq)*(xI-xq)+(yI-yq)*(yI-yq))   
            dist=math.sqrt((-self.L-xq)*(-self.L-xq)+yq*yq) # distance detector source
            SA_factor=self.r_det*self.r_det*pi/(4.0*pi*dist*dist)
            
            return self.time_per_seg*SA_factor*self.coll_func(dva)*math.exp(-self.mu*A*self.rho)
    
        else: 
            return 0
      
    def gen_upper_limit(self):
        for i in range(0,self.nbin_x):
            for j in range(0,self.nbin_y):
                 pts=np.dot(self.rot_mats,self.pt_arr[i,j])
                 sense_arr = np.array([self.pred_algo(pt) for pt in pts])
                 upper_lim_arr=scipy.stats.poisson.interval(0.999,self.observed)[1]/sense_arr #[0] --> lowerlimits [1]--> upper limits
                 #print(upper_lim_arr)
                 
                 self.upper_lim[i,j]=np.min(upper_lim_arr) # getting numpy inf
             
        for i in range(0,self.nbin_x):
            for j in range(0,self.nbin_y):
                if np.linalg.norm(self.pt_arr[i,j])>self.Rb:
                    self.upper_lim[i,j]=10
        
    
    def make_sens_mats(self):
        #self.nbin_x=bins_x
        #self.nbin_y=bins_y
        #self.make_mats()
        
        sens_mats=np.zeros((self.num_seg,self.nbin_x,self.nbin_y))
        
        for k in range(0,self.num_seg):
            print("Generating Sensitivity Matrice ",k," out of ",self.num_seg)
            for i in range(0,self.nbin_x):
                for j in range(0,self.nbin_y):
                    pt=np.dot(self.rot_mats[k],self.pt_arr[i,j])
                    sens_mats[k][i,j]=self.pred_algo(pt)
    
        self.sens_mats=sens_mats
    
        return sens_mats
    
    def make_upper_limit_matrice(self,upper_limit):
        """ only entries which are inside of the barrel i.e. have somewhere nonzero weights"""
        
        upper_lim=np.zeros((self.nbin_x,self.nbin_y))
        
        for i in range(0,self.nbin_x):
            for j in range(0,self.nbin_y):  
                if np.linalg.norm(self.pt_arr[i,j])<self.Rb:
                    upper_lim[i,j]=upper_limit
                else:
                    upper_lim[i,j]=100
        return upper_lim
                
    def make_plot_limit(self):  
        z_sel=np.sort(self.upper_lim, axis=None) 
        n_discard=np.count_nonzero(z_sel==np.inf)

        levels=np.zeros((0,1))
        percentiles=np.linspace(0,100,num=10)
        for frac in percentiles:
            levels=np.append(levels,np.percentile(z_sel[0:-n_discard],frac))
        
        upper=math.log10(z_sel[-n_discard-1])
        lower=math.log10(z_sel[0]+1)
        print("upper: ",upper," lower: ",lower," lowabs=",z_sel[0]+1)
        levels=np.logspace(lower,upper,15)
        
        X,Y=self.make_contour_stuff()
        
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, z, levels=levels,norm=colors.LogNorm(), cmap=cm.Reds)
        ax.set_aspect("equal")
        circle = Circle((0, 0), 30, facecolor='none',linewidth=1,edgecolor="Red", alpha=0.5,)
        ax.add_patch(circle)
        point=np.array([[0],[-37]])
        cts_max=np.max(self.observed)
        for i in range(0,self.num_seg):
                pt=np.dot(self.rot_mats[i],point)
                r=self.observed[i]/cts_max*5+0.1
                ax.add_patch(Circle((pt[0,0], pt[1,0]),r, facecolor='Red',linewidth=1,edgecolor="Red", alpha=0.5))
        xmin=ymin=-44
        ymax=xmax=44
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        cbar = fig.colorbar(cs,ticks=levels,extend='max')
        cbar.ax.set_yticklabels(levels.astype(int))
        ax.set_ylabel('y [cm]')
        ax.set_xlabel('x [cm]')
        ax.grid(True,alpha=0.3)
        plt.show()
        
    def plot_barrel(self,data,para):
        """ assumes data fitting to current bining, """        
        lower=math.log10(1)
        upper=math.log10(np.max(data))
        levels=np.logspace(lower,upper,15)
        X,Y=self.make_contour_stuff()        
        fig, ax = plt.subplots()
        
        if para=="LIN":
            cs = ax.contourf(X, Y, data, cmap=cm.Reds)
        if para=="LOG":
            cs = ax.contourf(X, Y, data, levels=levels,norm=colors.LogNorm(), cmap=cm.Reds)
        ax.set_aspect("equal")
        circle = Circle((0, 0), 30, facecolor='none',linewidth=1,edgecolor="Red", alpha=0.5,)
        ax.add_patch(circle)
        xmin=ymin=-32
        ymax=xmax=32
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        cbar = fig.colorbar(cs,ticks=levels,extend='max')
        cbar.ax.set_yticklabels(levels.astype(int))
        ax.set_ylabel('y [cm]')
        ax.set_xlabel('x [cm]')
        ax.grid(True,alpha=0.3)
        plt.show()
        
    
    def generate_activity_matrice(self,sources):
      
        #self.make_sens_mats(bins_x,bins_y) # makte sens mats updates mats,too
        self.sources=sources # complete list with indices of pt_array
        for src in sources:
            if src=="POINT":
                self.make_point_source(sources["POINT"])
            if src=="ZYLINDER":
                self.make_zylinder_source(sources["ZYLINDER"])
            
    
           
    @staticmethod
    @jit(nopython=True)  
    def make_zylinder_source(self,param):
        x=param["X"]
        y=param["Y"]
        r=param["R"]
        A=param["A"]
        print("Cyl-Source: at: ",x,"/",y,"R: ",r,"A: ",A)
        A_den=A/(r*r*pi)
        A_den=A_den*self.bin_size[0,0]*self.bin_size[1,0]
        print("Activity density: ",A_den,"Bq/cm2")
        pt_zyl=np.array([[x],[y]])     
        tot=0
        num_pix=0
        for idx in self.idx_list:
            pt=self.pt_arr[idx]
            if np.linalg.norm((pt_zyl-pt))<r:
                self.ActivityMatrice[idx]=self.ActivityMatrice[idx]+A_den
                tot=tot+A_den
                num_pix=num_pix+1
        print("distributed: ",tot,"Bq, Dev.: ",(tot-A)/A*100.0,"%")
        
    def make_point_source(self,param):
        x=param["X"]
        y=param["Y"]
        A=param["A"]
        pt_pt=np.array([[x],[y]])
        bin_size=np.array([[(self.x_max-self.x_min)/self.nbin_x],[(self.y_max-self.y_min)/self.nbin_y]])
        num_pts=0
        for idx in self.idx_list:
            pt=self.pt_arr[idx]
            if np.abs((pt_pt-pt))<bin_size/2:
                num_pts=num_pts+1
                self.ActivityMatrice[idx]=self.ActivityMatrice[idx]+A    
        print("Point source points: ",num_pts," only one is correct!")
        
    def reset_act_mat(self):
        for idx in self.idx_list:
            self.ActivityMatrice[idx]=0         
        
    def simulate_prediction(self):
        """ prediction should run on fine grid, homogenious mu"""
        #source_dict={"POINT":{"X":10,"Y":11,"A":1e1} , "ZYLINDER":{"X":10,"Y":11,"R":5,"A":1e1}} 
        measured_data=np.zeros(self.num_seg)
 
        for i in range(0,self.num_seg):
            print("Simulating Segment ",i," out of ",self.num_seg)
            measured_data[i]=np.sum(np.multiply(self.sens_mats[i],self.ActivityMatrice))
        
        self.simulated=np.random.poisson(measured_data)
        
        param,comp_stuff,source=self.get_parameters()
        path="C:\\Users\\rum\\Desktop\\"        
        doc_str="SimDat_Src"+str(len(source))+"V2.npz" 
        outfile=open(path+doc_str,"bw")
        
        np.savez(outfile, angles=self.angle_arr, data=self.simulated,parameter=param,sources=source,calc_param=comp_stuff)

        outfile.close()

        
        return self.simulated

    def plot_simulation(self): 
        measured_data=self.simulated 
        max_cts=np.max(measured_data)
        num_seg=self.num_seg
        theta = np.linspace(0.0, 2 * np.pi, num_seg, endpoint=False)
        width = (2*np.pi) / num_seg
        
        ax = plt.subplot(111, polar=True)
        #plt.title(doc_string,loc='left')
        bars = ax.bar(theta, measured_data, width=width, bottom=0)
        ax.set_rorigin(-max_cts/2)
        
        # Use custom colors and opacity
        for r, bar in zip(measured_data, bars):
            bar.set_facecolor(plt.cm.jet(r / max_cts))
            bar.set_alpha(0.8)
        
        #ax1 = plt.subplot(111)    
        circle = Circle((-500,20), 100, facecolor='none',edgecolor="red", linewidth=1, alpha=0.5)
        ax.add_patch(circle)    
        
        plt.show()


        
    def get_parameters(self):     
        param_dict={"DetPos":self.L,"R_Barrel":self.Rb,"R_Detector":self.r_det,"NSeg": self.num_seg,"TInt":self.time_per_seg,"RhoMat": self.rho,"MuMat":self.mu}    
        computation_dict={"BinsX":self.nbin_x,"BinsY":self.nbin_y,"MaxX":self.x_max,"MinX":self.x_min,"MaxY":self.y_max,"MinY":self.y_min}
        source_list=self.sources
        return param_dict,computation_dict,source_list



    
if __name__ == '__main__':      
    

    path="C:\\Users\\rum\\Desktop\\"
    file_name="sim_dat_A1.0e+05_NSeg40Int30minmu2.0e-02_Si_xy_(15_-15)(-15_-15).npz"
    file_name="sim_dat_A1.0e+05_NSeg40Int30minmu5.0e-02_Si_xy_(3_-10)(3_-10).npz"
    datfile=np.load(path+file_name,"br")
    observed=datfile['data']  
         
    observed=datfile['data']
    bins_x=bins_y=300
    m=model_gen(300,300)
    #m.set_observation(observed)
    m.make_sens_mats()
    #m.make_mats()
    
    #m.gen_upper_limit()
    #z=m.upper_lim
    #m.make_plot()  
    #source_dict={"POINT":{"X":10,"Y":11,"A":1e1} , "ZYLINDER":{"X":10,"Y":11,"R":5,"A":1e1}} 
    source_dict={"ZYLINDER":{"X":15,"Y":15,"R":8,"A":1e5}} 
    #m.make_sens_mats(bins_x,bins_y)
    m.generate_activity_matrice(source_dict)
    m.simulate_prediction()
    m.plot_simulation()

    