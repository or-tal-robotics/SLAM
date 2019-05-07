#!/usr/bin/env python 
import numpy as np
from scipy.stats import multivariate_normal

class ndt():
    def __init__(self,last_scan, new_scan,T0 = [0,0,0],res = 1.0, Niter = 10, th = 0.00001):
        self.res = res
        self.Niter = Niter
        self.T = np.array(T0)
        self.th = th
        self.new_scan = new_scan
        self.xmin, self.xmax, self.ymin, self.ymax = np.min(last_scan[:,0]), np.max(last_scan[:,0]), np.min(last_scan[:,1]), np.max(last_scan[:,1])
        xbin = int((self.xmax-self.xmin)//res)
        ybin = int((self.ymax-self.ymin)//res)
        H = np.histogram2d(last_scan[:,0],last_scan[:,1],bins = [xbin,ybin],range=[[self.xmin, self.xmax], [self.ymin, self.ymax]])
        self.B = np.argwhere(H>3)
        self.Mu = []
        self.Sigma = []
        for ii in range(len(self.B)):
            idxs = (last_scan[:,0]<=self.B[ii,0]*res + self.xmin) + (last_scan[:,0]>(self.B[ii,0]-1)*res + self.xmin) + (last_scan[:,1]<=self.B[ii,1]*res + self.ymin) + (last_scan[:,1]>(self.B[ii,1]-1)*res + self.ymin)
            self.Mu.append(np.mean(last_scan[idxs],axis=0)) 
            self.Sigma.append(np.cov(last_scan[idxs])) 
        
    def calc_t(self):
        for kk in range(self.Niter):
            new_scan_trans = self.transform(self.new_scan,self.T)
            q, new_scan_trans_filtered, covs = self.calc_residual(new_scan_trans)
            dT = 0.0
            for ii in range(len(new_scan_trans_filtered)):
                J = self.calc_Jacobi(new_scan_trans_filtered[ii])
                qq = q[ii].T.dot(np.matmul( np.linalg.inv(covs[ii]),J))
                dq = self.calc_sd(new_scan_trans_filtered[ii])
                dtemp = np.zeros((3,3))
                dtemp[2,2] = q[ii].T.dot(np.linalg.inv(covs[ii]).dot(dq))
                g = qq * np.exp(-0.5*q[ii].T.dot(np.linalg.inv(covs[ii]).dot(q[ii])))
                H = -np.exp(-0.5*q[ii].T.dot(np.linalg.inv(covs[ii]).dot(q[ii]))) * (np.outer(-qq,-qq) + dtemp - np.matmul(J.T,np.matmul(np.linalg.inv(covs[ii]),J)))
                dT += -np.linalg.inv(H).dot(g)
            self.T += dT
            if np.sum(np.abs(dT)) < self.th:
                break
        return self.T
            

    def transform(self,scan, T):
        x, y = scan[:,0], scan[:,1] 
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y]) + T[0:2]
        return m
    
    def Likelihood(self,scan):
        p = 0.0
        for ii in range(len(self.B)):
            idxs = (scan[:,0]<=self.B[ii,0]*self.res + self.xmin) + (scan[:,0]>(self.B[ii,0]-1)*self.res + self.xmin) + (scan[:,1]<=self.B[ii,1]*self.res + self.ymin) + (scan[:,1]>(self.B[ii,1]-1)*self.res + self.ymin)
            if len(idxs) > 1:
                p += np.sum(multivariate_normal.pdf(scan[idxs], mean=self.Mu[ii], cov=self.Sigma[ii]))
        return p

    def calc_residual(self,scan):
        q = []
        new_scans = []
        covs = []
        for ii in range(len(self.B)):
            idxs = (scan[:,0]<=self.B[ii,0]*self.res + self.xmin) + (scan[:,0]>(self.B[ii,0]-1)*self.res + self.xmin) + (scan[:,1]<=self.B[ii,1]*self.res + self.ymin) + (scan[:,1]>(self.B[ii,1]-1)*self.res + self.ymin)
            if len(idxs) > 1:
                q.append(np.array(scan[idxs]-self.Mu[ii]))
                new_scans.append(np.array(scan[idxs]))
                covs.append(self.Sigma[ii])
        return q, new_scans, covs


    def calc_Jacobi(self,x):
        return np.array([[1,0,-x[0]*np.sin(x[2])-x[1]*np.cos(x[2])], [0,1,x[0]*np.cos(x[2])-x[1]*np.sin(x[2])]])
    def calc_sd(self,x):
        return np.array([[-x[0]*np.cos(x[2])+x[1]*np.sin(x[2])], [-x[0]*np.sin(x[2])-x[1]*np.cos(x[2])]])
    




    


    

