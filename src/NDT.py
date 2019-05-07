#!/usr/bin/env python 
import numpy as np
class ndt():
    def __init__(self,last_scan, new_scan,T0 = [0,0,0],res = 1.0):
        xmin, xmax, ymin, ymax = np.min(last_scan[:,0]), np.max(last_scan[:,0]), np.min(last_scan[:,1]), np.max(last_scan[:,1])
        xbin = int((xmax-xmin)//res)
        ybin = int((ymax-ymin)//res)
        H = np.histogram2d(last_scan[:,0],last_scan[:,1],bins = [xbin,ybin],range=[[xmin, xmax], [ymin, ymax]])
        self.B = np.argwhere(H>3)
        self.Mu = []
        self.Sigma = []
        for ii in range(len(B)):
            idxs = (last_scan[:,0]<=self.B[ii,0]*res + xmin) + (last_scan[:,0]>(self.B[ii,0]-1)*res + xmin) + (last_scan[:,1]<=self.B[ii,1]*res + ymin) + (last_scan[:,1]>(self.B[ii,1]-1)*res + ymin)
            self.Mu.append(np.mean(last_scan[idxs],axis=0)) 
            self.Sigma.append(np.cov(last_scan[idxs])) 
        
        new_scan_trans = self.transform(new_scan,T0)
        p = likelihood(new_scan_trans)

    def transform(self,scan, T):
        x, y = scan[:,0], scan[:,1] 
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y]) + T[0:2]
        return m



    


    

