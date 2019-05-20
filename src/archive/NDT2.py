#!/usr/bin/env python 
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class ndt():
    def __init__(self,last_scan, new_scan,T0 = [0.01,0.01,0.01],res = 1.0, Niter = 10, th = 0.00001):
        self.res = res
        self.Niter = Niter
        self.T = np.array(T0).astype(np.float)
        self.th = th
        self.new_scan = new_scan
        self.xmin, self.xmax, self.ymin, self.ymax = np.min(last_scan[:,0])- res, np.max(last_scan[:,0])+res, np.min(last_scan[:,1])-res, np.max(last_scan[:,1])+res
        self.xbin = int((self.xmax-self.xmin)//res) + 1
        self.ybin = int((self.ymax-self.ymin)//res) + 1
        self.Mu = {}
        self.Sigma = {}
        self.Mul = []
        self.scan_arange = {}
        self.R = {}
        self.new_scan = new_scan
        for ii in range(self.xbin):
            for jj in range(self.ybin):
                idxs = (last_scan[:,0]>=ii*res + self.xmin) * (last_scan[:,0]<(ii+1)*res + self.xmin) * (last_scan[:,1]>=(jj)*res + self.ymin) * (last_scan[:,1]<(jj+1)*res + self.ymin)
                if sum(idxs) > 2:
                    self.Mu[ii,jj] = np.mean(last_scan[idxs,:],axis=0)
                    self.Mul.append(np.mean(last_scan[idxs,:],axis=0))
                    self.Sigma[ii,jj] = np.cov(last_scan[idxs,:].T) + (1e-20)*np.eye(2)
                    new_idxs = (new_scan[:,0]>=ii*res + self.xmin) * (new_scan[:,0]<(ii+1)*res + self.xmin) * (new_scan[:,1]>=(jj)*res + self.ymin) * (new_scan[:,1]<(jj+1)*res + self.ymin)
                    if sum(new_idxs) > 0:
                        self.scan_arange[ii,jj] = new_scan[new_idxs]
                        self.R[ii,jj] = new_scan[new_idxs] - self.Mu[ii,jj]
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.array(self.Mul))
                        
    def predict(self):
        for jj in range(self.Niter):
            dT = 0.0
            delta = [0.0,0.0]
            for ii in range(len(self.new_scan)):
                g, h = self.calc_Hessian(self.T,self.new_scan[ii].T,delta)
                if g is not None:
                    dT = (np.linalg.inv(h)).dot(g)
                    print(dT)
                    self.T -= dT
            
            dT = 0.0
            delta = [-self.res*0.2,0.0]
            for ii in range(len(self.new_scan)):
                g, h = self.calc_Hessian(self.T,self.new_scan[ii].T,delta)
                if g is not None:
                    dT =(np.linalg.inv(h)).dot(g)
                    self.T -= dT
            dT = 0.0
            delta = [0.0,-self.res*0.2]
            for ii in range(len(self.new_scan)):
                g, h = self.calc_Hessian(self.T,self.new_scan[ii].T,delta)
                if g is not None:
                    dT =(np.linalg.inv(h)).dot(g)
                    self.T -= dT
            dT = 0.0
            delta = [-self.res*0.2,-self.res*0.2]
            for ii in range(len(self.new_scan)):
                g, h = self.calc_Hessian(self.T,self.new_scan[ii].T,delta)
                if g is not None:
                    dT = (np.linalg.inv(h)).dot(g)
                    self.T -= dT
            
            
        return self.T
                        
                        
                    
                        
    def calc_Jacobi(self,T,X):
        return np.array([[1,0,-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])],
                         [0,1,X[0]*np.cos(T[2])-X[1]*np.sin(T[2])]])
    
    def calc_sd(self,T,X):
        return np.array([[-X[0]*np.cos(T[2])+X[1]*np.sin(T[2])],
                         [-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])]])
    
    def transform(self,T, X):
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, X.T) + T[0:2]
        return np.array(m).reshape(2)
    
    def transform_vec(self,T, X):
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, X.T).T + T[0:2].T
        return m
    
    
    
    def calc_Resudual(self,T,X,delta):
        Xt = self.transform(T,X)
        iXt = int(np.ceil((Xt[0] - self.xmin + delta[0])/self.res))
        jXt = int(np.ceil((Xt[1] - self.ymin + delta[1])/self.res))
        if self.Mu.get((iXt,jXt)) is not None:
            q = Xt - self.Mu[iXt,jXt]
            c = self.Sigma[iXt,jXt]
            return q, c
        else:
            return None, None
        
    def calc_Resudual2(self,T,X,delta):
        Xt = self.transform(T,X)
        
        q, indices = self.nbrs.kneighbors(Xt.reshape((1,2)))
        q = Xt - np.array(self.Mul[int(indices)])
        c = 0.1*np.eye(2)
        print(q, c)
        return q, c
    
        
    def calc_Hessian(self,T,X,delta):
        q,c = self.calc_Resudual(T,X,delta)
        Xt = self.transform(T,X)
        if q is not None:
            e = -(np.exp(-0.5*q.T.dot(np.linalg.inv(c).dot(q))) +1e-20)
            J = self.calc_Jacobi(T,Xt)
            qq = q.T.dot(np.matmul( np.linalg.inv(c),J))
            dtemp = np.zeros((3,3))
            dq = self.calc_sd(T,Xt)
            dtemp[2,2] = -q.T.dot(np.linalg.inv(c).dot(dq))
            r = np.outer(qq,qq) + dtemp - np.matmul(J.T,np.matmul(np.linalg.inv(c),J))
            H = r*e
            G = qq * e
            #print(H[2,2],G[2],e)
            return G, H
        else:
            return None, None
    
    
                        
    

def main():
    last_scan = np.load('test/last_scan.npy')
    new_scan = np.load('test/new_scan.npy')
    plt.scatter(last_scan[0,:],last_scan[1,:])
    plt.scatter(new_scan[0,:],new_scan[1,:])
    
    Ndt = ndt(last_scan.T,new_scan.T)
    T = Ndt.predict()
    
    new_scan_tranformed =np.array(Ndt.transform_vec(T,new_scan.T).T)
    
    plt.scatter(last_scan[0,:],last_scan[1,:],c='k')
    plt.scatter(new_scan[0,:],new_scan[1,:],c='r')
    plt.scatter(new_scan_tranformed[0,:],new_scan_tranformed[1,:],c='b')
    
    
if __name__ == "__main__":
    main()
    




    


    

