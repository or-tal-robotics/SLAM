#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Created on Thu Sep  5 10:09:21 2019

@author: Matan Samina
"""

import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class tPF():
    
    def __init__(self,Np = 100 , grid_size = [10,10], initial_pose = [5,5]):

        # creat Np first particales 
        self.Np = Np
        self.initialize_PF()

        rospy.init_node('listener', anonymous=True)
      
        # convert maps to landmarks arrays:
        self.oMap = maps("LM1") 
        self.tMap = maps("LM2")
  

    def initialize_PF( self , angles =np.linspace(0 , 2*np.pi , 100) , radius = np.linspace(-10 , 10 , 10) ):
       
        # make a list of class rot(s)
        self.Rot = []

        for i in range(len(angles)):
            for j in range(len(radius)):
                self.Rot.append(rot(angles[i] ,radius [j]))
        
        print ("initialaize PF whith") , len(self.Rot) , (" samples completed")


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

class maps:

    def __init__(self , topic_name ):
        
       # plt.plot([1,2,3], [1,2,3])
       #plt.show(block=True)
        self.started = False # indicate if msg recived
        self.map = None #landmarks array
        self.originalShift = None # the original shift of the map from origin

        rospy.Subscriber( topic_name , numpy_msg(Floats) , self.callback)
                  
    def callback(self ,data):

        # reshape array from 1D to 2D
        landmarks = np.reshape(data.data, (-1, 2))
        print landmarks.shape
        self.started = True 
        self.map = np.array(landmarks)         

    def shift2origin(self):  # Define 'shit2Origin' to shift coordinates to Center of Mass

        #find C.M of 2D map and creat C.M vector shape(2*1) 

        cm = np.sum(np.transpose(self.map),axis=1)/len(self.map)
        self.originalShift = cm
        print cm
        self.map = self.map - cm  # shift map to center of mass
    
    def plot(self):

        plt.scatter(self.map[: , 0] ,self.map[:,1])
        plt.show(block =True)

    def rotate(self, r , RotationAngle): #rotat map
      
        theta = np.radians(RotationAngle) # angles to radians
        c ,s = np.cos(theta) , np.sin(theta)
        R = np.array(((c,-s), (s, c))) #Rotation matrix
   
        RotatedLandmarks = np.matmul( R , self.Coordinates.T ) + np.array([[r*c] , [r*s] ])# matrix multiplation (2*n)X(2*2)

        self.rotated_LM = RotatedLandmarks
   

if __name__ == '__main__':
   
    print "Running"
    PFtry = tPF()
    r = rospy.Rate(0.5)
    rospy.spin()  
    