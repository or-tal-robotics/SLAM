#!/usr/bin/env python 

import rospy
import numpy as np
import random
import geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from laser_scan_get_map import MapClientLaserScanSubscriber  
import tf_conversions
import tf
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors as KNN
from scipy.stats import multivariate_normal
from matplotlib.mlab import bivariate_normal
from OccupancyGrid import occupancy_grid
class ParticleFilter(object):

    def __init__ (self,Np = 100):
        self.ctr = 1
        self.laser_tf_br = tf.TransformBroadcaster()
        self.laser_frame = rospy.get_param('~laser_frame')
        self.pub_particlecloud = rospy.Publisher('/particlecloud', PoseArray, queue_size = 60)
        self.pub_estimated_pos = rospy.Publisher('/MCL_estimated_pose', PoseWithCovarianceStamped, queue_size = 60)
        self.pub_particlecloud2fusion = rospy.Publisher('/particlecloud2fuse_out', PoseArray, queue_size = 60)
        self.scan = MapClientLaserScanSubscriber ()
        self.last_time = rospy.Time.now().to_sec()
        self.Np = Np
        self.init()   
        self.i_TH = 0.0  
        self.i_MU = [0.0 ,0.0]
        self.M_idxs = (np.linspace(0,len(self.scan.z.ranges)-1,20)).astype(np.int32)
        rospy.Subscriber('/odom', Odometry, self.get_odom) 
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.init_pose)
        self.map = occupancy_grid()
        #print self.M_idxs
        #print self.scan.obs()
        #print self.nbrs, "print"

    def get_map(self,map):
        self.map = map

    def update_particles_ndt(self,X):
        for ii in range(self.Np):
            self.particles[ii,:] = X
        
    def get_odom(self, msg):  # callback function for odom topic
        self.odom = msg
        current_time = msg.header.stamp.secs 
        self.dt = current_time - self.last_time
        self.last_time = current_time
        self.ind_map_update()
        if np.abs(self.odom.twist.twist.linear.x)>0.05 or np.abs( self.odom.twist.twist.angular.z)>0.01:
            self.prediction()
        if self.update_TH() > 0.01: #and self.ctr%1 == 0:
            self.likelihood_fild(self.map)
            self.i_TH = 0.0
            self.ctr = 1
            self.resampling()
            
            #if 1/np.sum(self.weights**2) < self.Np/5:
             #   self.resampling()

        self.ctr += 1

    def init_pose (self,msg): # callback function for /initialpose topic
        X = np.zeros(3)
        sigmas = np.zeros((3,3))
        position = np.zeros(3)
        orientation_quaternion = np.zeros(4)
        pose_stemp = msg
        position = [pose_stemp.pose.pose.position.x, pose_stemp.pose.pose.position.y, pose_stemp.pose.pose.position.z]
        orientation_quaternion = [pose_stemp.pose.pose.orientation.x, pose_stemp.pose.pose.orientation.y, pose_stemp.pose.pose.orientation.z, pose_stemp.pose.pose.orientation.w]
        orientation_euler = tf_conversions.transformations.euler_from_quaternion(orientation_quaternion)
        X = [position[0], position[1], orientation_euler[2]]
        sigmas[0,0] = pose_stemp.pose.covariance[0]
        sigmas[1,1] = pose_stemp.pose.covariance[7]
        sigmas[2,2] = pose_stemp.pose.covariance[35]

        self.init(X0 = X, P0 = sigmas)

    def init (self, X0 = [0,0,0], P0 = [[0.0001,0,0],[0,0.0001,0],[0,0,0.0001*np.pi*2]]):
        self.particles = np.random.multivariate_normal(X0, P0, self.Np)
        self.weights = np.ones (self.Np) / self.Np

    def prediction (self): #Odometry = Odometry massege. donot forget to initialize self.last_time = 
        dot = np.zeros((self.Np,3))

        dot[:,0] = self.odom.twist.twist.linear.x
        dot[:,1] = self.odom.twist.twist.linear.y
        dot[:,2] =  self.odom.twist.twist.angular.z

        sigma_x = np.sqrt(self.odom.twist.covariance[0]) + 0.02
        sigma_y = np.sqrt(self.odom.twist.covariance[7]) + 0.02
        sigma_theta = np.sqrt(self.odom.twist.covariance[35]) + 0.02
 
        #self.x_pose_cov = self.odom.pose.covariance[0] ############
        #self.y_pose_cov = self.odom.pose.covariance[7] ###########
        #self.theta_pose_cov = self.odom.pose.covariance[35] #########
        
        delta = np.zeros((self.Np,3)) 
           
        delta[:,2] = dot[:,2] * self.dt + sigma_theta * np.random.randn(self.Np) 

        self.particles[:,2] += delta[:,2] 

        delta[:,0] = (dot[:,0] + sigma_x * np.random.randn(self.Np)) * self.dt * np.cos(self.particles[:,2])  
        delta[:,1] = (dot[:,0] + sigma_y * np.random.randn(self.Np)) * self.dt * np.sin(self.particles[:,2]) 

        self.particles[:,0] += delta[:,0]
        self.particles[:,1] += delta[:,1] 

    def update_TH(self):
        self.i_TH += (self.odom.twist.twist.linear.x + self.odom.twist.twist.angular.z) * self.dt
        return np.abs(self.i_TH)


    def ind_map_update(self):
        self.i_MU[0] += np.abs(self.odom.twist.twist.linear.x * self.dt)
        self.i_MU[1] += np.abs(self.odom.twist.twist.angular.z * self.dt)

    def likelihood_fild(self,map):
        print "likelihood"
        for ii in range(self.Np):
            self.weights[ii] = map.get_Likelihood(self.particles[ii,:],self.scan.z,0.5)
        if np.sum(self.weights) == 0 or np.sum(np.isnan(self.weights))> 0:
            self.weights = np.ones(self.Np)/self.Np
        else:
            self.weights = self.weights / np.sum(self.weights)


    


    def resampling(self):
        index = np.random.choice(a = self.Np,size = self.Np,p = self.weights)
        self.particles = self.particles[index]
        self.weights = np.ones (self.Np) / self.Np
        self.particles[:,0] += 0.0001 * np.random.randn(self.Np) 
        self.particles[:,1] += 0.0001 * np.random.randn(self.Np) 
        self.particles[:,2] += 0.0001 * np.random.randn(self.Np) 
        

    def pub (self):
        particle_pose = PoseArray()
        particle_pose.header.frame_id = 'map'
        particle_pose.header.stamp = rospy.Time.now()
        particle_pose.poses = []
        estimated_pose = PoseWithCovarianceStamped()
        estimated_pose.header.frame_id = 'map'
        estimated_pose.header.stamp = rospy.Time.now()
        estimated_pose.pose.pose.position.x = np.mean(self.particles[:,0])
        estimated_pose.pose.pose.position.y = np.mean(self.particles[:,1])
        estimated_pose.pose.pose.position.z = 0.0
        quaternion = tf_conversions.transformations.quaternion_from_euler(0, 0, np.mean(self.particles[:,2]) )
        estimated_pose.pose.pose.orientation = geometry_msgs.msg.Quaternion(*quaternion)

        

        
        for ii in range(self.Np):
            pose = geometry_msgs.msg.Pose()
            point_P = (self.particles[ii,0],self.particles[ii,1],0.0)
            pose.position = geometry_msgs.msg.Point(*point_P)

            quaternion = tf_conversions.transformations.quaternion_from_euler(0, 0, self.particles[ii,2]) 
            pose.orientation = geometry_msgs.msg.Quaternion(*quaternion)

            particle_pose.poses.append(pose)
        
        self.pub_particlecloud.publish(particle_pose)
        self.pub_estimated_pos.publish(estimated_pose)
        #print(self.laser_frame)
        self.laser_tf_br.sendTransform((np.mean(self.particles[:,0]) , np.mean(self.particles[:,1]) , 0),
                            (estimated_pose.pose.pose.orientation.x,estimated_pose.pose.pose.orientation.y,estimated_pose.pose.pose.orientation.z,estimated_pose.pose.pose.orientation.w),
                            rospy.Time.now(),
                            self.laser_frame,
                            "map")
        
    def pub2fuse (self):
        particle_pose = PoseArray()
        particle_pose.header.frame_id = 'map'
        particle_pose.header.stamp = rospy.Time.now()
        particle_pose.poses = []
        
        for ii in range(self.Np):
            pose = geometry_msgs.msg.Pose()
            point_P = (self.particles[ii,0],self.particles[ii,1],0.0)
            pose.position = geometry_msgs.msg.Point(*point_P)

            quaternion = tf_conversions.transformations.quaternion_from_euler(0, 0, self.particles[ii,2]) 
            pose.orientation = geometry_msgs.msg.Quaternion(*quaternion)

            particle_pose.poses.append(pose)
        
        self.pub_particlecloud2fusion.publish(particle_pose)

if __name__ == "__main__":
    rospy.init_node('particle_filter', anonymous = True)
    PF_l = ParticleFilter()

    r = rospy.Rate(5)
    #plt.ion()
    #fig = plt.figure()
    
    while not rospy.is_shutdown():
        r.sleep()

       
        PF_l.pub()
        #mean =  np.zeros(3)
        #mean = np.mean(PF_l.particles,axis=0)

        #M = PF_l.scan.loction_based(mean)

        r.sleep()
        #plt.imshow(-M+PF_l.scan.occupancy_grid)
       # fig.canvas.draw()


    rospy.spin()
