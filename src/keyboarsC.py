#!/usr/bin/env python

import sys
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def _init_(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def _call_(self): return self.impl()

class _GetchUnix:
    def _init_(self):
        import tty, sys

    def _call_(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

class _GetchWindows:
    def _init_(self):
        import msvcrt

    def _call_(self):
        import msvcrt
        return msvcrt.getch()


def printing():

    rospy.init_node('robot_cordinator', anonymous=True) 
    
    coordinates_subscriber = rospy.Subscriber('/turtle1/pose', Pose, callback)

    pose = Pose()
    print(pose.theta)
    
def callback(data):
    pose = data


if _name_ == '_main_':
    try:
        printing()
    except rospy.ROSInterruptException:
        pass