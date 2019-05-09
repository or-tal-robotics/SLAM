#!/usr/bin/env python 

import rospy
import numpy as np
from particle_filter import ParticleFilter
from OccupancyGrid import occupancy_grid

def main():

    rospy.init_node('SLAM', anonymous = True)
    PF = ParticleFilter(Np=100)
    PF.init()
    OG = occupancy_grid(max_rays = 100,grid_size = [30,30],initial_pose = [15,15])
    OG.update_map(scan = PF.scan.z, initialize=1)
    PF.get_map(OG)
    r = rospy.Rate(5)
    OG.publish_occupancy_grid()
    
    while not rospy.is_shutdown():
        OG.publish_occupancy_grid()
        print 'runing!'
        r.sleep()
        PF.pub()
        if PF.i_MU[0] > 0.05 or PF.i_MU[1] > 0.05:
            #print np.mean(PF.particles,axis = 0).shape
            OG.update_map(X = np.mean(PF.particles,axis = 0),scan = PF.scan.z, initialize=0)
            PF.get_map(OG)
            PF.i_MU = [0.0,0.0]
            print 'updated map'
            
  

    rospy.spin()


if __name__ == "__main__":
    main()