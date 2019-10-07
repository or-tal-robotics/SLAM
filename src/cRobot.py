#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:54:38 2019

@author: matan

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# from scipy.optimize import rosen, differential_evolution

class rot(object):
    
    # define 'rot' to be the class of the rotation for resamplimg filter

    def __init__(self , theta , dist):
        
         self.theta = np.radians(theta)
         self.dist = dist
         self.w = []
         
    def weight(self , oMap , tMap):
        
        var = 0.16
        
        # initial KNN with the original map 
        nbrs = NearestNeighbors(n_neighbors= 1, algorithm='ball_tree').fit(np.transpose(oMap))
        # fit data of map 2 to map 1  
        distances, indices = nbrs.kneighbors(tMap)
        # find the propability 
        prob = (1/(np.sqrt(2*np.pi*var)))*np.exp(-np.power(distances,2)/(2*var)) 
        # returm the 'weight' of this transformation
        wiegth =np.sum((prob)/np.prod(distances.shape)) #np.sum((prob)/np.prod(distances.shape)) 
        
        print(wiegth)
        
        self.w.append(wiegth)

    def det(self):
        
        r= self.dist
        theta = np.radians(self.theta) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s,c*r), (s, c,s*r),(0 , 0 ,1))) #Rotation matrix
        
        return np.linalg.det(R)
   

##(1/sqrt(2*pi*var))*exp((-(measurements(1) - dist_par_1)^2)/(2*var))

class maps(object):

    def __init__ ( self ,  MAP = None ):
        
        size = np.shape(MAP) # Dim of map
        
        if size[0] == 2:
            self.Coordinates = np.transpose(MAP)
            
        elif size[1] == 2:
            self.Coordinates = np.MAP
            
        self.shift2origin() # shit cordintats of landmarks to origin
        self.plot() # plot coordinates of landmarks
        

    def plot(self): # plot map
        
        #plt.figure()
        plt.scatter(self.Coordinates[:,0] , self.Coordinates[:,1])
        
    def rotate(self, r , RotationAngle): #rotat map
      
        theta = np.radians(RotationAngle) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s), (s, c))) #Rotation matrix
   
        RotatedLandmarks = np.matmul( R , self.Coordinates.T ) + np.array([[r*c] , [r*s] ])# matrix multiplation (2*n)X(2*2)
        
        
        #plt.scatter(RotatedLandmarks[0,:] , RotatedLandmarks[1,:])
        #plt.scatter(self.Coordinates[:,0] , self.Coordinates[:,1])
        
        return RotatedLandmarks
        
    def rotate2(self, r , RotationAngle): #rotat map
      
        theta = np.radians(RotationAngle) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s, c*r), (s, c,s*r),(0 , 0 ,1))) #Rotation matrix
   
        #RotatedLandmarks = np.matmul(R , np.transpose(self.Coordinates))  # matrix multiplation (2*n)X(2*2)
   
        RotatedLandmarks = R.dot(self.Coordinates.T) #+ np.array([[r*c] , [r*s] , [1]])
     
        plt.scatter(RotatedLandmarks[0,:] , RotatedLandmarks[1,:])
        plt.scatter(self.Coordinates[:,0] , self.Coordinates[:,1])
        
        return RotatedLandmarks

    def shift2origin(self):  # Define 'moveToOrigin' to shift voordinates to C.M
                
        #plt.scatter(self.Coordinates[:,0] , self.Coordinates[:,1])

        #find C.M of 2D map and creat C.M vector shape(2*1) 
        cm = np.sum(np.transpose(self.Coordinates),axis=1)/len(self.Coordinates)
        self.Coordinates = self.Coordinates - cm  # shift map to center of mass
        
        #plt.scatter(self.Coordinates[:,0] , self.Coordinates[:,1])


def importmap(mapArr):
    
    lm = np.copy(mapArr[1:3,:]) 
    
    # index not to be included
    index = np.argwhere(lm[1:3,:] == 0)

    lmx = np.delete(lm[0,:] , index , 0)  # delet empty cells
    lmy = np.delete(lm[1,:] , index , 0)  # delet empty cells
    lm = np.vstack((lmx,lmy))


    return lm





