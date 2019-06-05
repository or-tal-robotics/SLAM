#!/usr/bin/env python 
import numpy as np
import rospy
from skimage.draw import line
from nav_msgs.msg import OccupancyGrid
import tf_conversions
from sklearn.neighbors import NearestNeighbors as KNN
from std_msgs.msg import Int8MultiArray
from DENDT import dendt
from laser_scan_get_map import MapClientLaserScanSubscriber  
import random
import geometry_msgs
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
import tf
from sensor_msgs.msg import PointCloud
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.mlab import bivariate_normal
from sensor_msgs.msg import LaserScan
from nav_msgs.srv import GetMap

class ParticleFilterPm():
    def __init__(self,Np = 100, max_rays = 30):
        self.i_TH = 0.0
        self.max_rays = max_rays
        self.last_time = rospy.Time.now().to_sec()
        # ---- TF Broadcaster ---- #
        self.laser_tf_br = tf.TransformBroadcaster()
        self.laser_frame = rospy.get_param('~laser_frame')

        # ---- Publishers ---- #
        self.pub_particlecloud = rospy.Publisher('/particlecloud', PoseArray, queue_size = 60)
        self.pub_estimated_pos = rospy.Publisher('/MCL_estimated_pose', PoseWithCovarianceStamped, queue_size = 60)
        self.map_publisher = rospy.Publisher('/map', OccupancyGrid, queue_size = 60)
        self.PCL_map_publisher = rospy.Publisher('/PCL_map', PointCloud, queue_size = 20)


        # ---- Subscribers ---- #
        rospy.Subscriber('/scan',LaserScan,self.get_scan)
        rospy.Subscriber('/odom', Odometry, self.get_odom)
        self.scan = rospy.wait_for_message("/scan",LaserScan)
        self.odom = rospy.wait_for_message("/odom",Odometry)

        # ---- Particle filter params ---- #
        self.Np = Np
        self.initialize_PF()   
        

    # ---- Calbackes ---- #
    def get_scan(self, msg):
        self.scan = msg
         
    def get_odom(self, msg):  # callback function for odom topic
        self.odom = msg
        self.update_dt(msg.header.stamp.secs)
        if np.abs(self.odom.twist.twist.linear.x)>0.05 or np.abs( self.odom.twist.twist.angular.z)>0.1:
            self.prediction()
        if self.update_TH(reset=False) > 0.05: #and self.ctr%1 == 0:
            self.update_map()
            self.update_particles()
            self.update_TH(reset=True)
            self.ctr = 1
            self.resample()

    # ---- Particle filter --- #

    def prediction (self): #Odometry = Odometry massege. donot forget to initialize self.last_time = 
        dot = np.zeros((self.Np,3))
        dot[:,0] = self.odom.twist.twist.linear.x
        dot[:,1] = self.odom.twist.twist.linear.y
        dot[:,2] =  self.odom.twist.twist.angular.z
        sigma_x = np.sqrt(self.odom.twist.covariance[0]) + 0.02
        sigma_y = np.sqrt(self.odom.twist.covariance[7]) + 0.02
        sigma_theta = np.sqrt(self.odom.twist.covariance[35]) + 0.02
        delta = np.zeros((self.Np,3)) 
        delta[:,2] = dot[:,2] * self.dt + sigma_theta * np.random.randn(self.Np) 
        self.particles[:,2] += delta[:,2] 
        delta[:,0] = (dot[:,0] + sigma_x * np.random.randn(self.Np)) * self.dt * np.cos(self.particles[:,2])  
        delta[:,1] = (dot[:,0] + sigma_y * np.random.randn(self.Np)) * self.dt * np.sin(self.particles[:,2]) 
        self.particles[:,0] += delta[:,0]
        self.particles[:,1] += delta[:,1] 

    def update_TH(self,reset = False):
        if reset:
            self.i_TH = 0.0
        else:
            self.i_TH += (self.odom.twist.twist.linear.x + self.odom.twist.twist.angular.z) * self.dt
        return np.abs(self.i_TH)

    def update_dt(self,current_time):
        self.dt = current_time - self.last_time
        self.last_time = current_time

    def initialize_PF(self, X0 = [0,0,0], P0 = [[0.0001,0,0],[0,0.0001,0],[0,0,0.0001*np.pi*2]]):
        self.particles = np.random.multivariate_normal(X0, P0, self.Np)
        self.weights = np.ones (self.Np) / self.Np

    def update_particles(self):
        for ii in range(self.Np):
            self.weights[ii] = self.get_Likelihood(self.particles[ii],s = 3.5)
            if np.isnan(self.weights[ii]) or np.isinf(self.weights[ii]):
                self.weights[ii] = 1/self.Np
        self.weights = self.weights/np.sum(self.weights)

    def resample(self):
        index = np.random.choice(a = self.Np,size = self.Np,p = self.weights)
        self.particles = self.particles[index]
        self.maps = self.maps[index]
        self.weights = np.ones (self.Np) / self.Np
        self.particles[:,0] += 0.0001 * np.random.randn(self.Np) 
        self.particles[:,1] += 0.0001 * np.random.randn(self.Np) 
        self.particles[:,2] += 0.0002 * np.random.randn(self.Np) 

    def publish_particles(self):
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
        self.laser_tf_br.sendTransform((np.mean(self.particles[:,0]) , np.mean(self.particles[:,1]) , 0),
                            (estimated_pose.pose.pose.orientation.x,estimated_pose.pose.pose.orientation.y,estimated_pose.pose.pose.orientation.z,estimated_pose.pose.pose.orientation.w),
                            rospy.Time.now(),
                            self.laser_frame,
                            "map")

    def bounds(self,T):
        max_x = np.max(self.particles[:,0]) + T[0]
        min_x = np.min(self.particles[:,0]) + T[0]
        max_y = np.max(self.particles[:,1]) + T[1]
        min_y = np.min(self.particles[:,1]) + T[1]
        max_t = np.max(self.particles[:,2]) + T[2]
        min_t = np.min(self.particles[:,2]) + T[2]
        return [(max_x,min_x),(max_y,min_y),(max_t,min_t)]
    
    # ---- Map ---- #              
    def update_map(self,initialize = 0):
        X = np.mean(self.particles,axis = 0)
        if initialize == 0:
            Z = self.scan2cart(X,self.scan,self.max_rays)
            self.new_scan = Z
            T0 = X - self.last_pose
            Dendt = dendt(last_scan=self.last_scan,new_scan=self.new_scan,bounds=self.bounds(T0),maxiter=10)
            T = Dendt.T
            X = T + self.last_pose
            self.last_scan = self.new_scan
            self.last_pose = X
            Z = Dendt.transform(self.new_scan.T,T)
            
            ranges, indices = self.nbrs.kneighbors(Z)
            print ranges>0.2
            print self.objects_map.shape,Z[ranges>0.2].shape
            self.objects_map = np.append(self.objects_map,Z[ranges>0.2].T,axis = 0)
            self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.objects_map)
            plt.scatter(self.objects_map[:,0],self.objects_map[:,1])
            

        else:
            self.objects_map = self.scan2cart([0.0,0.0,0.0],self.scan,self.max_rays)
            self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.objects_map)
            self.last_scan = self.objects_map
            self.last_pose = X
        
        
    
    def get_Likelihood(self,robot_origin,s = 1.0):
        z_star = self.scan2cart(robot_origin,self.scan,20)
        _, indices = self.nbrs.kneighbors(z_star.T)
        z = self.objects_map[indices].reshape(z_star.shape)
        return np.sum(np.exp(-s* np.linalg.norm(z_star-z,axis=1)**2))



    
    def scan2cart(self,robot_origin,scan,max_rays):
        # the robot orientation in relation to the world [m,m,rad]
        mu_x = robot_origin[0]
        mu_y = robot_origin[1]
        theta = robot_origin[2]  
        # converting the laserscan measurements from polar coordinate to cartesian and create matrix from them.
        n = len(scan.ranges); i = np.arange(len(scan.ranges))
        angle = np.zeros(n); x = np.zeros(n); y = np.zeros(n)
        rng = np.asarray(scan.ranges)
        angle = np.add(scan.angle_min, np.multiply(i, scan.angle_increment)) + theta
        idxs = (np.linspace(0,n-1,max_rays)).astype(int)
        #print idxs,rng.shape
        x = np.multiply(rng[idxs], np.cos(angle[idxs])) + mu_x
        y = np.multiply(rng[idxs], np.sin(angle[idxs])) + mu_y
        x[~np.isfinite(x)] = -1
        y[~np.isfinite(y)] = -1
        


        Y = np.stack((x,y))
        #Y = np.delete(Y,np.where(Y==-1))
        #Y = Y[~np.all(Y == -1, axis=1)]
        idx = (x==-1) + (y==-1)
        Y = np.delete(Y,np.where(idx),axis=1)

        return Y

    def publish_occupancy_grid(self):
        grid = OccupancyGrid()
        grid.data = []
        
        grid.data = self.map.ravel().astype(int).tolist()
        grid.header.stamp = rospy.Time.now()
        grid.header.frame_id = '/world'
        grid.info.height = np.float(self.map.shape[0])
        grid.info.width = np.float(self.map.shape[1])
        grid.info.resolution = self.res
        grid.info.origin.position.x = -self.y_shape*self.res+self.initial_pose[1].astype(np.float)
        grid.info.origin.position.y = -self.initial_pose[0].astype(np.float) #
        self.map_publisher.publish(grid)

    
    
        
def main():

    rospy.init_node('SLAM', anonymous = True)
    PF = ParticleFilterPm()
    PF.initialize_PF()
    PF.update_map(initialize=1)
    r = rospy.Rate(5)
    
    while not rospy.is_shutdown():
        print 'runing!'
        r.sleep()
        PF.publish_particles()
    
            
  

    rospy.spin()


if __name__ == "__main__":
    main()

