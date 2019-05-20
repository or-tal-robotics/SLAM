#!/usr/bin/env python 
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class ndt():
    def __init__(self,last_scan, new_scan,T0 = [0,0,0],res = 0.5, Niter = 10, th = 0.00001):
        self.res = res
        self.Niter = Niter
        self.T = np.array(T0).astype(np.float)
        self.th = th
        self.new_scan = new_scan
        self.xmin, self.xmax, self.ymin, self.ymax = np.min(last_scan[0,:])- res, np.max(last_scan[0,:])+res, np.min(last_scan[1,:])-res, np.max(last_scan[1,:])+res
        self.xbin = int((self.xmax-self.xmin)//res) + 1
        self.ybin = int((self.ymax-self.ymin)//res) + 1
        self.Mu = {}
        self.Sigma = {}
        self.scan_arange = {}
        self.R = {}
        for ii in range(self.xbin):
            for jj in range(self.ybin):
                idxs = (last_scan[:,0]>=ii*res + self.xmin) * (last_scan[:,0]<(ii+1)*res + self.xmin) * (last_scan[:,1]>=(jj)*res + self.ymin) * (last_scan[:,1]<(jj+1)*res + self.ymin)
                if sum(idxs) > 2:
                    self.Mu[ii][jj] = np.mean(last_scan[idxs,:],axis=0)
                    self.Sigma[ii][jj] = np.cov(last_scan[idxs,:]) 
                    new_idxs = (new_scan[:,0]>=ii*res + self.xmin) * (new_scan[:,0]<(ii+1)*res + self.xmin) * (new_scan[:,1]>=(jj)*res + self.ymin) * (new_scan[:,1]<(jj+1)*res + self.ymin)
                    if sum(new_idxs) > 0:
                        self.scan_arange[ii][jj] = new_scan[new_idxs]
                        self.R[ii][jj] = new_scan[new_idxs] - self.Mu[ii][jj]
                        
    def predict
                        
                    
        self.H = self.H.T
        self.B = np.argwhere(self.H>=3)
        self.Mu = []
        self.Sigma = []
        for ii in range(len(self.B)):
            idxs = (last_scan[0,:]>=self.B[ii,0]*res + self.xmin) * (last_scan[0,:]<(self.B[ii,0]+1)*res + self.xmin) * (last_scan[1,:]>=(self.ybin-self.B[ii,1])*res + self.ymin) * (last_scan[1,:]<(self.ybin-self.B[ii,1]+1)*res + self.ymin)
            print(idxs)
            self.Mu.append(np.mean(last_scan[:,idxs],axis=1)) 
            self.Sigma.append(np.cov(last_scan[:,idxs])) 
      
            
        
    def calc_t(self):
        for kk in range(self.Niter):
            new_scan_trans = self.transform(self.new_scan,self.T)
            #print(new_scan_trans)
            q, new_scan_trans_filtered, covs = self.calc_residual(new_scan_trans)
            print(q, new_scan_trans_filtered, covs)
            dT = 0.0
            for ii in range(len(new_scan_trans_filtered)):
                J = self.calc_Jacobi(self.T,new_scan_trans_filtered[ii])
                qq = q[ii].T.dot(np.matmul( np.linalg.inv(covs[ii]),J))
                dq = self.calc_sd(self.T,new_scan_trans_filtered[ii])
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
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, scan).T + T[0:2]
        return np.array(m)
    
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
            idxs = (scan[:,0]<=self.B[ii,0]*self.res + self.xmin) * (scan[:,0]>(self.B[ii,0]-1)*self.res + self.xmin) * (scan[:,1]<=(self.ybin-self.B[ii,1])*self.res + self.ymin) * (scan[:,1]>(self.ybin-self.B[ii,1]-1)*self.res + self.ymin)
            if sum(idxs) > 1:
                #print(idxs)
                #print(scan[idxs],self.Mu[ii])
                q.append(np.array(scan[idxs]-self.Mu[ii]))
                new_scans.append(np.array(scan[idxs]))
                covs.append(self.Sigma[ii])
        return q, new_scans, covs


    def calc_Jacobi(self,T,X):
        return np.array([[1,0,-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])],
                         [0,1,X[0]*np.cos(T[2])-X[1]*np.sin(T[2])]])
    def calc_sd(self,T,X):
        return np.array([[-X[0]*np.cos(T[2])+X[1]*np.sin(T[2])],
                         [-Z[0]*np.sin(T[2])-X[1]*np.cos(T[2])]])


def main():
    last_scan = np.load('test/last_scan.npy')
    new_scan = np.load('test/new_scan.npy')
    plt.scatter(last_scan[0,:],last_scan[1,:])
    plt.scatter(new_scan[0,:],new_scan[1,:])
    Ndt = ndt(last_scan,new_scan)
    plt.imshow(Ndt.H)
    T = Ndt.calc_t()
    res = 0.5
    ii=1

    idxs = (last_scan[0,:]>=Ndt.B[ii,0]*res + Ndt.xmin) * (last_scan[0,:]<(Ndt.B[ii,0]+1)*res + Ndt.xmin) * (last_scan[1,:]>=(Ndt.B[ii,1]*res + Ndt.ymin)) * (last_scan[1,:]<((Ndt.B[ii,1]+1)*res + Ndt.ymin))
            print(idxs)
    
if __name__ == "__main__":
    main()
    




    


    

