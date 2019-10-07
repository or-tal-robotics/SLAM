#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Created on Thu Sep  5 10:09:21 2019

@author: matan
"""


import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
import tf_conversions as tf


newMap = ""
started = False
T = ""


def callback(data):
 
    
    print "New map recived"

    newMap = data
    pub = rospy.Publisher('/map2T', OccupancyGrid , queue_size=1000 )


    theta = input("Rotation in degrees : ")
    dist = input("dist in map units : ")

    x , y ,z , w = tf.transformations.quaternion_from_euler(0, 0, np.radians(theta))

    # change map orientation
    newMap.info.origin.orientation.x = x
    newMap.info.origin.orientation.y = y
    newMap.info.origin.orientation.z = z
    newMap.info.origin.orientation.w = w

    # change map origin 
    newMap.info.origin.position.x = newMap.info.origin.position.x + dist*np.cos(theta)
    newMap.info.origin.position.y = newMap.info.origin.position.y + dist*np.sin(theta)

   
    r = rospy.Rate(0.1) # 0.1 hz

    while not rospy.is_shutdown():

        pub.publish(newMap)
        print newMap.info.origin
        r.sleep()


def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/ABot2/map", OccupancyGrid , callback)
    rospy.spin()    

if __name__ == '__main__':

    print "Running"
    listener()
    