#DE-NDT by Or Tslil
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import time

class dendt():
    def __init__(self,last_scan,new_scan,bounds,n = 10, reg = 0.0005, nn = 5, maxiter=200,popsize=6,tol=0.0001):
        self.new_scan = new_scan
        self.Mus, self.Sigmas = self.ndt2(last_scan.T,n,reg, nn)
        self.result = differential_evolution(self.func, bounds,maxiter=maxiter,popsize=popsize,tol=tol)
        self.T = self.result.x
        
    def ndt2(self,X,n = 10, reg = 0.0005, nn = 5):
        #idxs = np.random.choice(len(X),n)
        idxs = np.linspace(0,len(X)-1,n).astype(int)
        Mus = X[idxs]
        nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(Mus)
        Sigmas = []
        for ii in range(n):
            idxs = nbrs.kneighbors(Mus[ii].reshape((1,2)),n_neighbors = nn, return_distance=False)
            c = np.cov(Mus[idxs].reshape(Mus[idxs].shape[1],Mus[idxs].shape[2]).T) + reg*np.eye(2)
            Sigmas.append(c)
        return np.array(Mus), Sigmas
    
    def transform(self,X,T):
        T = np.array(T).reshape(3)
        c, s = np.cos(T[2]), np.sin(T[2])
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, X.T).T + T[0:2].T
        return m
    
    def likelihood(self,X, T):
        X = self.transform(X,T)
        p = np.zeros(len(X))
        for ii in range(len(self.Mus)):
            p += multivariate_normal.pdf(X, mean = self.Mus[ii], cov = self.Sigmas[ii])
        return p
    
    def func(self,T):
        X = self.new_scan.T
        return -np.prod(self.likelihood(X, T))



if __name__ == "__main__":
    last_scan = np.load('test/last_scan.npy')
    new_scan = np.load('test/new_scan.npy')

    bounds = [(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)]
    t0 = time.time()
    Dendt = dendt(last_scan=last_scan,new_scan=new_scan,bounds=bounds,maxiter=10,popsize=6,tol=0.0001,n=15)
    dt = (time.time() - t0)
    print(dt)
    new_scan_T = np.array(Dendt.transform(new_scan.T,Dendt.T)).T
        
    plt.scatter(last_scan[0,:],last_scan[1,:],c='k')
    plt.scatter(new_scan[0,:],new_scan[1,:],c='r')
    plt.scatter(new_scan_T[0,:],new_scan_T[1,:],c='b')


