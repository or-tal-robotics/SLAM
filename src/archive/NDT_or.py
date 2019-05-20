#!/usr/bin/env python 
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class ndt():
    def __init__(self,last_scan, new_scan,T0 = [0,0,0],max_T = [0.5,0.5,0.5], Niter = 100, th = 1e-10, sigma = 2.5):
        self.sigma = sigma
        self.Niter = Niter
        self.T = np.array(T0).astype(np.float)
        self.th = th
        self.dx = 0.00001
        self.dy = 0.00001
        self.dtheta = 0.0000001
        self.Nl = 1
        self.max_T = max_T
        self.fit(last_scan.T)
        self.Ts = []
        self.Score = []
        for ii in range(10):
            rx = np.random.uniform(-max_T[0],max_T[0])
            ry = np.random.uniform(-max_T[1],max_T[1])
            rtheta = np.random.uniform(-max_T[2],max_T[2])
            self.T = np.array(T0).astype(np.float) + np.array([rx,ry,rtheta])
            self.Ts.append(self.optimize(self.T,new_scan.T))
            self.Score.append(self.likelihood(self.T,new_scan.T))
        best = np.argmax(self.Score)
        self.T = self.Ts[best]
        
        
    def optimize(self, T0,scan):
        T = np.array(T0).astype(np.float)
        for i in range(self.Niter):
            T[2]+=self.grad_theta(T,scan)*0.05/np.sqrt(i+1)
            if np.abs(T[2]) > self.max_T[2]:
                break
            
        for i in range(self.Niter):
            T[0]+=self.grad_x(T,scan)*0.05/np.sqrt(i+1)
            if np.abs(T[0]) > self.max_T[0]:
                break
            
        for i in range(self.Niter):
            T[1]+=self.grad_y(T,scan)*0.05/np.sqrt(i+1)
            if np.abs(T[1]) > self.max_T[1]:
                break
            
        return T
        
    def grad_y(self,T,scan):
        p0 = self.likelihood(T,scan)
        Ty = T
        Ty[1] += self.dy
        py = self.likelihood(Ty,scan)
        grady = (py-p0)/self.dy
        return grady
    
    def grad_x(self,T,scan):
        p0 = self.likelihood(T,scan)
        Tx = T
        Tx[0] += self.dx
        px = self.likelihood(Tx,scan)
        gradx = (px-p0)/self.dx
        return gradx
    
    def grad_theta(self,T,scan):
        p0 = self.likelihood(T,scan)
        Ttheta = T
        Ttheta[2] += self.dtheta
        ptheta = self.likelihood(Ttheta,scan)
        gradtheta = (ptheta-p0)/self.dtheta
        return gradtheta

    def fit(self, scan):
       self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(scan)
       self.Nl = self.likelihood([0,0,0],scan)
       
    def likelihood(self,T,scan):
       scan = self.transform(scan,T)
       distances, _ = self.nbrs.kneighbors(scan)
       L = np.sum(np.exp(-self.sigma*distances))
       return L/self.Nl
       
 
    def transform(self,scan, T):
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, scan.T).T + T[0:2]
        return np.array(m)
    
    
def main():
    last_scan = np.load('test/last_scan.npy')
    new_scan = np.load('test/new_scan.npy')
    plt.scatter(last_scan[0,:],last_scan[1,:])
    plt.scatter(new_scan[0,:],new_scan[1,:])
    
    Ndt = ndt(last_scan,new_scan)
    
    new_scan_tranformed = Ndt.transform(new_scan.T,Ndt.T).T
    
    plt.scatter(last_scan[0,:],last_scan[1,:],c='k')
    plt.scatter(new_scan[0,:],new_scan[1,:],c='r')
    plt.scatter(new_scan_tranformed[0,:],new_scan_tranformed[1,:],c='b')
    
    
    plt.imshow(Ndt.H)
    T = Ndt.calc_t()
    res = 0.5
    ii=1

    idxs = (last_scan[0,:]>=Ndt.B[ii,0]*res + Ndt.xmin) * (last_scan[0,:]<(Ndt.B[ii,0]+1)*res + Ndt.xmin) * (last_scan[1,:]>=(Ndt.B[ii,1]*res + Ndt.ymin)) * (last_scan[1,:]<((Ndt.B[ii,1]+1)*res + Ndt.ymin))
            print(idxs)
    
if __name__ == "__main__":
    main()
    




    


    

