#!/usr/bin/env python 
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class ndt():
    def __init__(self,last_scan, new_scan,T0 = [0,0,0],K=6, Niter = 1000, th = 1e-10):
        self.Niter = Niter
        self.T = np.array(T0).astype(np.float)
        self.th = th
        self.new_scan = new_scan.T
        self.last_scan = last_scan.T
        self.K = K
            
       
            
    def fit(self):
        self.kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.last_scan)
        self.Mu = []
        self.Sigma = []
        self.kernel_idxs = []
        labels = self.kmeans.predict(self.last_scan)
        for ii in range(self.K):
            if len(self.last_scan[labels==ii])>=3:
                self.kernel_idxs.append(ii)
                self.Mu.append(self.kmeans.cluster_centers_[ii])
                self.Sigma.append(np.cov(self.last_scan[labels==ii].T)  + 0.5*np.eye(2))
               
                
    def predict(self):
        for kk in range(self.Niter):
            new_scan_trans = self.transform(self.new_scan,self.T)
            #print(new_scan_trans)
            self.q, self.new_scan_trans_filtered, self.covs = self.calc_residual(new_scan_trans)
            
            #print(q.shape)
            dT = np.array([0.0, 0.0, 0.0])
            for ii in range(len(self.new_scan_trans_filtered)):
                self.J = self.calc_Jacobi(self.T,self.new_scan_trans_filtered[ii])
                qq = self.q[ii].dot(np.matmul( np.linalg.inv(self.covs[ii]),self.J))
                #print(qq)
                dq = self.calc_sd(self.T,self.new_scan_trans_filtered[ii])
                dtemp = np.zeros((3,3))
                #print(q[ii],np.linalg.inv(covs[ii]),dq)
                dtemp[2,2] = self.q[ii].dot(np.linalg.inv(self.covs[ii]).dot(dq))
                self.g = qq * np.exp(-0.5*self.q[ii].dot(np.linalg.inv(self.covs[ii]).dot(self.q[ii].T)))
                self.H = -np.exp(-0.5*self.q[ii].dot(np.linalg.inv(self.covs[ii]).dot(self.q[ii].T))) * (np.outer(-qq,-qq) - dtemp - np.matmul(self.J.T,np.matmul(np.linalg.inv(self.covs[ii]),self.J))+ 0.00001*np.eye(3)) 

                dT += -np.linalg.inv(self.H).dot(self.g)
                #dT[0] *= -1
                #dT[1] *= -1
            self.T += 0.5*dT/np.float(len(self.new_scan_trans_filtered))
            '''if np.sum(np.abs(self.g)) < self.th:
                print('Converged after '+str(kk)+' iterations')
                break'''
        return self.T
            
            

    def transform(self,scan, T):
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, scan.T).T + T[0:2]
        return np.array(m)
    
    

    def calc_residual(self,scan):
        q = []
        new_scans = []
        covs = []
        labels = self.kmeans.predict(scan)
        for cc, ii in enumerate(self.kernel_idxs):
            if len(scan[labels==ii]) > 2:
                new_scans.append(np.array(scan[labels==ii]))
                for jj in range(len(np.array(scan[labels==ii]))):
                    covs.append(self.Sigma[cc])
                q.append(np.array(scan[labels==ii]-self.Mu[cc]))
        return np.concatenate(q), np.concatenate(np.array(new_scans)), covs


    def calc_Jacobi(self,T,X):
        return np.array([[1,0,-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])],
                         [0,1,X[0]*np.cos(T[2])-X[1]*np.sin(T[2])]])
    def calc_sd(self,T,X):
        return np.array([[-X[0]*np.cos(T[2])+X[1]*np.sin(T[2])],
                         [-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])]])

def main():
    last_scan = np.load('test/last_scan.npy')
    new_scan = np.load('test/new_scan.npy')
    plt.scatter(last_scan[0,:],last_scan[1,:])
    plt.scatter(new_scan[0,:],new_scan[1,:])
    
    Ndt = ndt(last_scan,new_scan)
    Ndt.fit()
    T = Ndt.predict()
    new_scan_tranformed = Ndt.transform(new_scan.T,T).T
    
    labels = Ndt.kmeans.predict(last_scan.T)
    plt.scatter(last_scan[0,:],last_scan[1,:],c=labels)
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
    




    


    

