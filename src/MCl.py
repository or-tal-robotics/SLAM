#!/usr/bin/env python 

import rospy
import numpy as np
from particle_filter import ParticleFilter
from OccupancyGrid import occupancy_grid

def main():

    rospy.init_node('SLAM', anonymous = True)
    PF = ParticleFilter(Np=100)
    OG = occupancy_grid()
    OG.update_map(scan = PF.scan.z)
    PF.get_map(OG)
    r = rospy.Rate(5)
    
    while not rospy.is_shutdown():
        r.sleep()
        if PF.i_MU[0] > 0.3 or PF.i_MU[1] > 0.1:
            OG.update_map(X = np.mean(PF.particles),scan = PF.scan.z)
            PF.get_map(OG)
            PF.i_MU = [0.0,0.0]
  

    rospy.spin()


if __name__ == "__main__":
    main()