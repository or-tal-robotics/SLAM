#!/usr/bin/env python 
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import scipy.optimize as optimize

def transform(X,T):
    T = np.array(T).reshape(3)
    c, s = np.cos(T[2]), np.sin(T[2])
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, X.T).T + T[0:2].T
    return m

def ndt(X, res = 1.0, phase = [0.0, 0.0], reg = 1e-10):
    dx, dy = phase[0], phase[1]
    xmin, xmax, ymin, ymax = np.min(X[:,0]), np.max(X[:,0]), np.min(X[:,1]), np.max(X[:,1])
    xbin = int((xmax-xmin)//res) + 1
    ybin = int((ymax-ymin)//res) + 1
    Mus = []
    Sigmas = []
    for ii in range(xbin):
            for jj in range(ybin):
                idxs = (X[:,0]>=ii*res + xmin + dx*res) * (X[:,0]<(ii+1)*res + xmin + dx*res) * (X[:,1]>=(jj)*res + ymin + dy*res) * (X[:,1]<(jj+1)*res + ymin + dy*res)
                if sum(idxs) > 2:
                    Mus.append(np.mean(X[idxs,:],axis=0))
                    Sigmas.append(np.cov(X[idxs,:].T) + reg*np.eye(2)*10)

    return np.array(Mus), Sigmas

def ndt2(X,n = 50, reg = 0.0005, nn = 5):
    idxs = np.random.choice(len(X),n)
    Mus = X[idxs]
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(Mus)
    Sigmas = []
    for ii in range(n):
        idxs = nbrs.kneighbors(Mus[ii].reshape((1,2)),n_neighbors = nn, return_distance=False)
        c = np.cov(Mus[idxs].reshape(Mus[idxs].shape[1],Mus[idxs].shape[2]).T) + reg*np.eye(2)
        Sigmas.append(c)
    return np.array(Mus), Sigmas

def likelihood(X, params):
    Mus = params[0]
    Sigmas = params[1]
    p = np.zeros(len(X))
    for ii in range(len(Mus)):
        p += multivariate_normal.pdf(X, mean = Mus[ii], cov = Sigmas[ii])
    return p

def Jaccobian(X,T):
    T = np.array(T).reshape(3)
    return np.array([[1,0,-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])],
                     [0,1,X[0]*np.cos(T[2])-X[1]*np.sin(T[2])]])
    
def DJaccobian(X,T):
    T = np.array(T).reshape(3)
    return np.array([[-X[0]*np.cos(T[2])+X[1]*np.sin(T[2])],
                     [-X[0]*np.sin(T[2])-X[1]*np.cos(T[2])]])    
     

def grad(X, T, params):
    Mus = params[0]
    Sigmas = params[1]
    Xt = transform(X,T)
    p = likelihood(Xt, params)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Mus)
    idxs = nbrs.kneighbors(Xt, return_distance=False)
    q = Xt - np.array(Mus[idxs]).reshape(Xt.shape)
    dT = []
    for ii in range(len(X)):
        dT.append((p[ii] * q[ii].dot(np.matmul(np.linalg.inv(Sigmas[int(idxs[ii])]),Jaccobian(X[ii],T)))))
    return dT

def Hessian(X, T, params, reg = 1e-40):
    Mus = params[0]
    Sigmas = params[1]
    Xt = transform(X,T)
    p = likelihood(Xt, params)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(Mus)
    idxs = nbrs.kneighbors(Xt, return_distance=False)
    q = Xt - np.array(Mus[idxs]).reshape(Xt.shape)
    H = []
    for ii in range(len(X)):
        dq = q[ii].dot(np.matmul(np.linalg.inv(Sigmas[int(idxs[ii])]),Jaccobian(X[ii],T)))
        dJ = np.zeros((3,3))
        dJ[2,2] = -q[ii].dot(np.linalg.inv(Sigmas[int(idxs[ii])])).dot(DJaccobian(X[ii],T))
        dD = -np.matmul(Jaccobian(X[ii],T).T,np.matmul(np.linalg.inv(Sigmas[int(idxs[ii])]),Jaccobian(X[ii],T)))
        Hes = p[ii] * ( np.outer(-dq,-dq) + dJ + dD) + np.eye(3)*(reg)
        #D,_ = np.linalg.eig(Hes)
        #dmin = np.min(D)
        #if dmin < 0:
        #    Hes = Hes + np.eye(3)*(reg-dmin)
        #    print(Hes)
        H.append(Hes)
        
    return H


def ndt_optimizer(Xr, Xn, T0, iterations = 1):
    T = np.array(T0)
    #Mus, Sigmas = ndt2(Xr,n = 10,reg = 0.05, nn = 10)
    Mus, Sigmas = ndt2(Xr)
    for it in range(iterations):
        g = grad(Xn, T, (Mus, Sigmas))
        h = Hessian(Xn, T, (Mus, Sigmas))
        Gm = np.mean(g,axis=0)
        Hm = np.mean(h,axis=0)
        dT = np.zeros(3)
        for ii in range(len(g)):
            dT = (np.linalg.inv(h[ii]).dot(g[ii].T)).reshape(dT.shape) + dT
        #T = -0.5*dT.T.reshape(T.shape)/len(g) + T
        T = T - (np.linalg.inv(Hm).dot(Gm.T)).reshape(T.shape)*0.01
        Xn = np.array(transform(Xn,T))
    return T

def main():
    last_scan = np.load('test/last_scan.npy')
    new_scan = np.load('test/new_scan.npy')
    plt.scatter(last_scan[0,:],last_scan[1,:])
    plt.scatter(new_scan[0,:],new_scan[1,:])
    T = [0.5 ,0.0 ,-0.2]
    T = ndt_optimizer(last_scan.T, new_scan.T, T, iterations=1000)
    
    new_scan_T = np.array(transform(new_scan.T,T).T)
    
    plt.scatter(last_scan[0,:],last_scan[1,:],c='k')
    plt.scatter(new_scan[0,:],new_scan[1,:],c='r')
    plt.scatter(new_scan_T[0,:],new_scan_T[1,:],c='b')
    Mus1, Sigmas = ndt2(last_scan.T)
    plt.scatter(Mus1[:,0],Mus1[:,1],c='y')
    #p = likelihood(new_scan.T, (Mus1, Sigmas))
    #g = grad(new_scan.T, T, (Mus1, Sigmas))
    #h = Hessian(new_scan.T, T, (Mus1, Sigmas))
    Mus2, Sigmas = ndt2(last_scan.T)
    
    plt.scatter(last_scan[0,:],last_scan[1,:],c='k')
    plt.scatter(Mus1[:,0],Mus1[:,1],c='b')
    plt.scatter(new_scan[0,:],new_scan[1,:],c='r')
    plt.scatter(Mus2[:,0],Mus2[:,1],c='r')
    
    
if __name__ == "__main__":
    main()

    

