#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:09:21 2019

@author: matan
"""
import rospy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid, Path

maxL = 1000
mapsArray = np.empty([1, maxL])


def callback(data):

    global mapsArray
    global maxL

    maps = np.array(data.data , dtype = int)
    N = np.sqrt(maps.shape)[0].astype(np.int64)
    Re = np.copy(maps.reshape((N,N)))
    #print(Re)
 
    #convert to landmarks array
    scale = data.info.resolution
    landMarksArray = np.transpose(np.argwhere( Re == 100 ) * scale)
    lengthOfLM = maxL-len(landMarksArray[0 , :])

    #creat array of zeros 
    Z = np.zeros(lengthOfLM)
    LMAfixX= np.append(landMarksArray[0 , :] ,Z)
    LMAfixY = np.append(landMarksArray[1 , :] , Z)

    LMAfix = np.stack((LMAfixX , LMAfixY))
    print(LMAfix)
    
    # add current landmarks array to the list
    mapsArray = np.vstack((mapsArray ,LMAfix))
    
    #plt.scatter(LMAfixX , LMAfixY)
    #plt.show()
    
    np.save('mapsArr1.npy', mapsArray)


    

def listener():
    
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/ABot1/map", OccupancyGrid , callback)
    rospy.spin()

    

if __name__ == '__main__':
    listener()
    