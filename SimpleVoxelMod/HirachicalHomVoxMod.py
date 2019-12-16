import numpy as np
#import matplotlib as plt
import math
from math import sqrt
 
import matplotlib.pyplot as pyplt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from PlotVoxel import PlotVoxel

"""
Center of barrel is at (0,0,0) similar to root sim

rectangular box, voxel outside of volume are defined i.e. one can loop till one reach the edge
outside voxels mu is set to 0

important: grids contain the center coordinate of voxel

Tracking in Hirachy: 
 get_next_box:
    if refined==False:
        do stuff - as implemented - no new things to do
    if refined==True:
        while refined==True: # go down in hirachy
            go dow          # find finest degree of refinement....
        while in subvolume:
            tracking, next box
         go up:
        while in subvolume 
             tracking, next box
        go up

"""       

class hom_voxel:
    def __init__(self,bins,boarders):       
        """ Top level stuff
        bins=Tuple!! (binsx,binsy,binsz)
        """     
        self.starts=boarders[0] #np.array([-30.0,-30.0,-40.0])
        self.ends=boarders[1] #np.array([30.0,30.0,40.0])
        self.bin_size=np.zeros((3,))     
        self.num_bins=np.array(bins)
        
        self.calc_bin_size() # important to do it immediatelly at creation
        
        """ Data Structure """
        self.mu_data=np.zeros((bins))
        
        """ ''''''''''''''''''''''''''''''''''''''''''''''''''''"""
        """ Refinement Information """
        self.is_refined=np.zeros((bins),dtype=bool) # is refined
        self.is_refined.fill(False)
        """ contains the list(!) index of the voxel below in self.hirachy"""
        self.hIDX=np.zeros((bins),dtype=int)
        """ Hirachy contains the information towards the next level """
        self.hirachy= []
        """ ''''''''''''''''''''''''''''''''''''''''''''''''''''"""
        
        # variables for tracking
        self.pt=np.zeros((3,))
        self.boarders=np.zeros((2,3))
        self.next_pt=np.zeros((3,))
        self.next_box=np.zeros((3,))
        self.norm_vec=np.zeros((3,))
        self.direction=np.zeros((3,))
        self.points=[] 
        self.refinement_pts=[]# collects points over hirachy
        """drawing stuff"""
        
        self.boxes = []
      
        
    def info(self):
        print("Binning: ",self.num_bins)
        print("Starts: ",self.starts)
        print("Ends: ",self.ends)
        
    def refine_voxel(self,pt): # assume 4x refinement
        idx=self.GetIndex(pt)
        """if pixel is not refined --> refine it"""
        if self.is_refined[tuple(idx)]==False: 
            self.is_refined[tuple(idx)]=True
            self.add_voxel(pt)
        else: # pixel is already refined---> Refine pt below
            idx_next=self.hIDX[tuple(idx)]
            self.hirachy[idx_next].refine_voxel(pt)
            
    def is_refined(self,pt):
        """ help function which returns refinement state of voxel containing pt"""
        idx=self.GetIndex(pt)
        return self.is_refined[tuple(idx)]
            
    def add_voxel(self,pt):
        idx=self.GetIndex(pt)
        additional_bins=(2,2,2)
        boarders=self.get_boarders(pt)  # get index is executed twice
        br=np.zeros((2,3))
        br[0]=boarders[1]
        br[1]=boarders[0]
        self.hirachy.append(hom_voxel(additional_bins,br))
        """ blocks easy removal of """ 
        self.hIDX[tuple(idx)]=len(self.hirachy)-1 
        
    def calc_bin_size(self):
        self.bin_size=(self.ends-self.starts)/self.num_bins

    def GetIndex(self,pt):        
        starts=np.where(self.starts==pt)
        pt[starts[0]]=pt[starts[0]]+1e-12
        
        #print("starts: ",range_check1)   
        ends=np.where(self.ends==pt)
        pt[ends[0]]=pt[ends[0]]-1e-12
        
        return  np.floor(((pt-self.starts)/self.bin_size)).astype(int)
    
    def SetIndex(self,pt):
        self.next_box=self.GetIndex(pt)

    def get_boarders(self,pt): # fine for points within box 
        bin_coord=self.GetIndex(pt)
        left= bin_coord*self.bin_size+self.starts+self.bin_size
        right= bin_coord*self.bin_size+self.starts#-self.bin_size/2+
        #self.boarders=np.stack((left,right),axis=0)
        return np.stack((left,right),axis=0)
    
    def get_boarders_next_box(self):
        bin_coord=self.next_box
        left= bin_coord*self.bin_size+self.starts+self.bin_size
        right= bin_coord*self.bin_size+self.starts
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
        #print(lambdas)
        return np.amin(lambdas[np.where(lambdas>1e-12)])
                
    def GetNextBox(self,norm): # norm can falsify next value but does not predict it
        #edge_check=(self.next_pt==self.boarders)
        edge_check=np.isclose(self.boarders-self.next_pt,0,atol=1e-12,rtol=1e-10 )
        check_inc=np.where(norm>0)
        check_dec=np.where(norm<0)
        # [0]--> Dekrement [1]-> Increment
        inc=np.where(edge_check[0]==True) # list of indizes
        dec=np.where(edge_check[1]==True)
        
        norm_check1=np.where(np.isin(inc,check_inc)==True) # tuple
        norm_check2=np.where(np.isin(dec,check_dec)==True)
        
        if norm_check1[0].shape[0]!=inc[0].shape[0]:
            print("wrong box selected while incrementing")
        if norm_check2[0].shape[0]!=dec[0].shape[0]:
            print("wrong box selected while decrementing")
    
        #print("Increment: ",inc," Decrement: ",dec)
        #old=self.next_box
        #print(self.next_box)
        self.next_box[inc]=self.next_box[inc]+1
        self.next_box[dec]=self.next_box[dec]-1
        return self.next_box
        #update=self.next_box
        #print(update-old)
        #return old,update
     
    def SetStartPt(self,pt,norm):
        self.clean_tracking()
        # make sure that pt has proper shape
        self.norm_vec=norm
        self.next_pt=pt
        for i in range(0,3):
            if norm[i]>0:        
                self.direction[i]=1
            if norm[i]<0:
                self.direction[i]=-1 # vector is initalized to zero rest ist zero
        
    def clean_tracking(self):
        self.pt=np.zeros((3,))
        self.boarders=np.zeros((2,3))
        self.next_pt=np.zeros((3,))
        self.next_box=np.zeros((3,))
        self.norm_vec=np.zeros((3,))
        self.direction=np.zeros((3,))
        self.points=[] 
        self.refinement_pts=[]# collects points over hirachy
        self.next_pt=np.zeros((3,))
     
    def is_in_volume(self,pt):
        """ checks if point is within current voxel"""
        # first point is on boarder, too!!! --> norm will help       
        range_check1=(self.starts<pt).all() # check for true expression
        #print("starts: ",range_check1)   
        range_check2=(self.ends>pt).all()
        #print("ends: ",range_check2)
        #return not (s1 or s2)
        return  (range_check1 and range_check2)
    
    def is_in_volume_boarder_inc(self,pt):
        """ checks if point is within current voxel"""
        #is_on_boarder=self.is_on_boarder(pt)
        
        does_exit=self.does_exit(pt)
              
        # first point is on boarder, too!!! --> norm will help       
        range_check1=(self.starts<=pt).all() # check for true expression
        #print("starts: ",range_check1)   
        range_check2=(self.ends>=pt).all()
        #print("ends: ",range_check2)
        #return not (s1 or s2)
        
        if does_exit==True:
            return False
        else:  
            return  (range_check1 and range_check2)
        
    def is_on_boarder(self,pt):
        boarder_check1=np.isclose(self.starts-pt,0,atol=1e-12,rtol=1e-10 ).any()
        boarder_check2=np.isclose(self.ends-pt,0,atol=1e-12,rtol=1e-10 ).any()
        return boarder_check1 or boarder_check2
    
    def does_exit(self,pt):
        """ checks if track exits volume i.e. point is on boarder AND direction points outwards"""
        boarder_check1=np.isclose(self.starts-pt,0,atol=1e-12,rtol=1e-10 )
        dir_check1=self.direction==-1
        s1=np.logical_and(dir_check1,boarder_check1).any()==True
        
        boarder_check2=np.isclose(self.ends-pt,0,atol=1e-12,rtol=1e-10 )
        dir_check2=self.direction==1
         # goes out of volume       
        s2=np.logical_and(dir_check2,boarder_check2).any()==True
        
        return s1 or s2
       
    def print_info(self,stuff):
        """ help function to debug lower braches"""
        if self.num_bins[0]==2:
            print("News from da under! My binsize is: ",self.bin_size[0])
            print(stuff)
            
    def track(self,pt,norm): ## pt norm is passed through hirachy 
        print("calculate track pt:",pt," direction: ",norm)
        self.clean_tracking()
        self.SetStartPt(pt,norm) # set next_pt to start i.e. pt
        self.next_box=self.GetIndex(pt) ## initialsing
        #self.print_info("start box: "+str(self.next_box))
        #self.print_info(str(self.starts)+str(self.ends))
        loc_pts=[]
        #self.print_info(self.is_in_volume_boarder_inc(self.next_pt))
        ctr=0
        while self.is_in_volume_boarder_inc(self.next_pt)==True: # and self.does_exit(self.next_pt)==False:
            idx=self.GetIndex(self.next_pt)
            #self.print_info(self.next_pt)
            #self.print_info(self.next_box)
            #self.print_info("Counter: "+str(ctr))
            if self.is_refined[tuple(self.next_box)]==True:
                
                br=self.hirachy[self.hIDX[tuple(self.next_box)]]   #br=self.hirachy[self.hIDX[tuple(idx)]]
                br.print_info("")
                """
                print("*************")
                print(self.next_pt)
                print("IDX: ",tuple(idx))
                print("NextBox: ",tuple(self.next_box))
                #br.info()
                print("*************")
                """
                pts=br.track(self.next_pt,norm)
                #print("points from refinement: ",len(pts))
                loc_pts.extend(pts)
                print(pts)
                          
                
                if len(pts)!=0:
                    print("")
                    #self.next_pt=pts[-1] # new point is set--> next box must be calculated
                    #self.next_box=self.GetIndex(self.next_pt)
                    # sub voxel is tracked -->    
                else: 
                    print("No points received!! from lower box")
                
           
            #print(tuple(idx))
            #print("IsRefined: ",self.is_refined[tuple(idx)])
            #if self.is_refined[tuple(self.next_box)]==False:
            borders=self.get_boarders(self.next_pt)
            idx_next_box=self.next_box
            #self.print_info(idx_next_box)
            
            self.next_point(self.next_pt,self.norm_vec)
            
            box=self.GetNextBox(self.norm_vec)
            borders=self.get_boarders_next_box()                
            ctr=ctr+1
            loc_pts.append(self.next_pt)
        
        #self.print_info("Last point: "+str(loc_pts[-1]))
        self.points=loc_pts
        return loc_pts
    
    def next_point(self,pt,norm):
        lamb=self.calc_intercept(pt,norm)
        #print(lamb)
        update=norm*lamb
        length=np.abs(update)
        correction=1
        if (length>self.bin_size).any()==True:
            print("wrong step length")
            correction=np.amax(length/self.bin_size)       
        self.next_pt=pt +update #*correction
            #self.next_box=self.GetIndex(self.next_pt) # initialiseren 
        #print(lamb)
        return self.next_pt
        
    def plot_voxel(self,ULim,LLim):
        """ Draws list of boxes """
        x_min,y_min,zmin=LLim
        x_max,y_max,zmax=ULim
    
        edgecolor='Black'
        facecolor='None'
        alpha=0.5
        
        fig, ax = plt.subplots()
        plt.axis('equal')        
        self.boxes = self.MakeBoxes(ULim,LLim,'Black')        
        pc = PatchCollection(self.boxes, facecolor=facecolor, alpha=alpha)
        ax.add_collection(pc)
        ax.set_xlim(x_min-self.bin_size[1], x_max+self.bin_size[1])
        ax.set_ylim(y_min-self.bin_size[2],y_max+self.bin_size[2])
        plt.show()
    
    def draw_track(self):
        pt=np.array(self.points)
        x=pt[:,1]
        y=pt[:,2]
        edgecolor='Black'
        facecolor='None'
        alpha=0.5
        UpperLim=np.array([5.3,5.3,5.3])
        LowerLim=np.array([-5.3,-5.3,-5.3])
        x_min,y_min,zmin=LowerLim
        x_max,y_max,zmax=UpperLim
        fig, ax = plt.subplots()
        plt.axis('equal')        
        self.boxes = self.MakeBoxes(UpperLim,LowerLim,'Black')        
        pc = PatchCollection(self.boxes, facecolor=facecolor, alpha=alpha)
        ax.add_collection(pc)
        plt.scatter(x, y, alpha=0.5)
        ax.set_xlim(x_min-self.bin_size[1], x_max+self.bin_size[1])
        ax.set_ylim(y_min-self.bin_size[2],y_max+self.bin_size[2])
        plt.show()
         
         
        
    def MakeBoxes(self, UpLim,LowLim,color):
        """ Generates list of boxes within UpLim and LowLim for drawing purpose"""
        LowerLimit=np.maximum(self.starts,LowLim.reshape((3,)))
        UpperLimit=np.minimum(self.ends,UpLim.reshape((3,)))
        x_min,y_min=LowerLimit[1],LowerLimit[2]
        x_max,y_max=UpperLimit[1],UpperLimit[2]
        pt=np.array([0.2,x_min,y_min])
        boxes=[]
        #print(pt," Max: ",x_max," ",y_max)
        while pt[1]<x_max:    
            while pt[2]<y_max:
                idx=self.GetIndex(pt)
                #print(tuple(idx))
                borders=self.get_boarders(pt)
                #print(borders)
                center=0.5*(borders[1]+borders[0]) # mean of upper and lower boarder
                #print(center)
                width= self.bin_size[1]
                height=self.bin_size[2]
                rect = Rectangle((center[1]-width/2,center[2]-height/2),width, height,edgecolor=color,linestyle="-")
                boxes.append(rect)
                if self.is_refined[tuple(idx)]==True:
                    #print("Is Refined")
                    br=self.hirachy[self.hIDX[tuple(idx)]]
                    #print("SubVoxel Boarders: ",br.starts," ",br.ends)
                    sub_boxes=br.MakeBoxes(br.ends,br.starts,'Blue')
                    #print(len(sub_boxes))
                    boxes.extend(sub_boxes)
                pt[2]=pt[2]+self.bin_size[2]
            pt[2]=y_min
            pt[1]=pt[1]+self.bin_size[1]
        return boxes
                
        
def get_rect(boarders,idx1,idx2):
    center=0.5*(borders[1]+borders[0]) # mean of upper and lower boarder
    size=(borders[1]-borders[0]) # upper - lower boarders
    coord=center-size/2
    rect = Rectangle((coord[idx1],coord[idx2]),size[idx1], size[idx2],linestyle="-",edgecolor="Red")
    return rect
    
def axis_tuple(a,b):
    lst=[a,b]
    lst.sort()
    ax_dict={"X":(1,0,0),"Y":(0,1,0),"Z":(0,0,1)}
    ax_idx={"X":0,"Y":1,"Z":2}
    return ax_dict[lst[0]],ax_dict[lst[1]],ax_idx[lst[0]],ax_idx[lst[1]]
        
        
if __name__ == '__main__':          
    """
    next_box contains idx for current box
    
    GetIndex
    GetNextBox() (given the next(!) point)
     --> Checks where next pt sits on current box boarder 
     --> increments index on those edges (can be one two or even 3 )
    
    Length check: sum over individual pieces should be equal to |end-start| 
    
    
    """
    
    bins=(60,60,80)
    upper=np.array([30,30,40])
    lower=np.array([-30,-30,-40])
    
    mod=hom_voxel(bins,np.stack((lower,upper),axis=0))
    
    pt=np.array([0.01,0.01,0.01])
    pt1=np.array([30.01,0.01,0.01])
    y=np.array([0,1,0])
    z=np.array([0,0,1])
    for i in range(-2,3):
        for j in range(-2,3):
            mod.refine_voxel(pt+i*y+j*z) ## 1 --> 0.5
    
    for i in range(0,2):
        for j in range(0,2):
            print(pt+i*y*0.25+j*z*0.25)
            mod.refine_voxel(pt+i*y*0.5+j*z*0.5)
    
    
    upper=np.array([5.3,5.3,5.3])
    lower=np.array([-5.3,-5.3,-5.3])
    
    #mod.plot_voxel(upper,lower)
     
    pt=np.array([0.2,-4.5,1.9]) # start
    #pt=np.array([0.2,-4.5,1.3])
    phi=2*math.pi*(344.1/360) # 30 deg
    
    nv=np.array([0,math.cos(phi),math.sin(phi)]) # normalvedtor
    tup1,tup2,idx1,idx2=axis_tuple("Y","Z")
    print(tup1,tup2," ",idx1," ",idx2)
    
    out_glob_box=False
    
    x=np.empty((0,))
    y=np.empty((0,))
    
    pts=mod.track(pt,nv)
    #print("Points: ",pts)
    pt=np.array(pts) 
    mod.draw_track()
    for i in range(0,pt.shape[0]-1):
        print(pt[i]-pt[i+1])
        
        
    """
    
    mod.next_box=mod.GetIndex(pt) # initialiseren 
    
    bin_size=mod.bin_size
    rect_lst=[]
    
    print("do tracking")
    
    while out_glob_box==False:
        #borders=mod.get_boarders(pt)
        print("***********************************")
        x=np.append(x,[pt[idx1]])
        y=np.append(y,[pt[idx2]])
        borders=mod.get_boarders(pt)
        
        idx=mod.next_box
        box=mod.GetNextBox(nv)
        borders=mod.get_boarders_next_box() 
        rect_lst.append(get_rect(borders,idx1,idx2))
        pt=mod.next_point(pt,nv)
        print(pt)
        print("***********************************")
        out_glob_box=((idx<0).any() or (idx>=mod.num_bins).any())
     
    pt=np.array([0,4.1,-9.1])
    start_b=mod.get_boarders(pt)
    
    ticks1=np.zeros((32,))
    ticks2=np.zeros((32,))
    for i in range(0,32):
        ticks1[i]=start_b[0,1]+bin_size[1]*i
        ticks2[i]=start_b[0,2]-bin_size[2]*i
        
    mod.GetIndex(pt)
    
    
    print("make plot with box")
    fig, ax = pyplt.subplots()
    ax.set_yticks(ticks2)
    ax.set_xticks(ticks1)
    pc = PatchCollection(rect_lst,alpha=.5,edgecolor="Red")
    ax.add_collection(pc)
    
    #ax.yaxis.grid(True, which='major')
    #ax.yaxis.grid(True, which='minor')
    
    pyplt.scatter(x, y, alpha=0.5)
    pyplt.grid(True,which="both")
    pyplt.show()

    """
    

