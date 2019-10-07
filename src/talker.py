#!/usr/bin/env python


import rospy
import numpy as np
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

def talker():
    pub = rospy.Publisher('floats', numpy_msg(Floats),queue_size=10)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(0.1) # 10hz
    while not rospy.is_shutdown():
        a = np.array([1.0, 2.1, 3.2, 4.3, 5.4, 6.5], dtype=np.float32)
        pub.publish(a)
        print "msg pass"
        r.sleep()

if __name__ == '__main__':
    talker()
