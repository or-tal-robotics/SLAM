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
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.mlab import bivariate_normal

class ParticleFilterRB():
    def __init__(self,Np = 100, res = 0.1, grid_size = [10,10], initial_pose = [5,5], max_rays = 30):
        self.laser_tf_br = tf.TransformBroadcaster()
        self.laser_frame = rospy.get_param('~laser_frame')
        self.pub_particlecloud = rospy.Publisher('/particlecloud', PoseArray, queue_size = 60)
        self.pub_estimated_pos = rospy.Publisher('/MCL_estimated_pose', PoseWithCovarianceStamped, queue_size = 60)
        self.map_publisher = rospy.Publisher('/map', OccupancyGrid, queue_size = 60)
        rospy.Subscriber('/odom', Odometry, self.get_odom)
        self.scan = MapClientLaserScanSubscriber()
        self.last_time = rospy.Time.now().to_sec()
        self.Np = Np
        self.init()   
        self.i_TH = 0.0  
        self.i_MU = [0.0 ,0.0]
        self.res, self.grid_size, self.initial_pose, self.max_rays = res, grid_size, np.array(initial_pose), max_rays
        self.x_shape = int(grid_size[0]//res)
        self.y_shape = int(grid_size[1]//res)
        self.map =-1*np.ones((self.x_shape,self.y_shape))
        
        

    def init (self, X0 = [0,0,0], P0 = [[0.0001,0,0],[0,0.0001,0],[0,0,0.0001*np.pi*2]]):
        self.particles = np.random.multivariate_normal(X0, P0, self.Np)
        self.weights = np.ones (self.Np) / self.Np
        
        
    def get_odom(self, msg):  # callback function for odom topic
        self.odom = msg
        current_time = msg.header.stamp.secs 
        self.dt = current_time - self.last_time
        self.last_time = current_time
        if np.abs(self.odom.twist.twist.linear.x)>0.05 or np.abs( self.odom.twist.twist.angular.z)>0.1:
            self.prediction()
        if self.update_TH() > 0.05: #and self.ctr%1 == 0:
            self.update_particles()
            self.i_TH = 0.0
            self.ctr = 1
            self.resampling()

    def resample(self):
        index = np.random.choice(a = self.Np,size = self.Np,p = self.weights)
        self.particles = self.particles[index]
        self.maps = self.maps[index]
        self.weights = np.ones (self.Np) / self.Np
        self.particles[:,0] += 0.0001 * np.random.randn(self.Np) 
        self.particles[:,1] += 0.0001 * np.random.randn(self.Np) 
        self.particles[:,2] += 0.0002 * np.random.randn(self.Np) 
            

    def update_particles(self):
        for ii in range(self.Np):
            self.maps[ii], self.objects_map_knn[ii], self.objects_map[ii] = self.update_map(map = self.maps[ii], scan = self.scan.z, X = self.particles[ii])
            self.weights[ii] = self.get_Likelihood(self.particles[ii],self.scan.z,self.objects_map[ii],self.objects_map_knn[ii],s = 3.5)
            if np.isnan(self.weights[ii]):
                self.weights[ii] = 1/self.Np

    
    def update_map(self,scan,X=[0,0,0],initialize = 0):
        Z = self.scan2cart(X,scan,self.max_rays)
        
        if initialize == 0:
            self.new_scan = Z
            T0 = X - self.last_pose
            bounds = [(-0.5+T0[0],0.5+T0[0]),(-0.5+T0[1],0.5+T0[1]),(-0.5+T0[2],0.5+T0[2])]
            Dendt = dendt(last_scan=self.last_scan,new_scan=self.new_scan,bounds=bounds,maxiter=10)
            T = Dendt.T
            X = T + self.last_pose
            self.last_scan = self.new_scan
            self.last_pose = X
        else:
            self.last_scan = Z
            self.last_pose = X

        idxs = self.cart2ind(Z)
        if idxs is not None:
            #print idxs.shape
            robot_idx = ((X[0:2] + self.initial_pose) // self.res).astype(int)
            robot_idx[1] = -robot_idx[1] +  self.y_shape
            for ii in range(len(idxs)):
                rr, cc = line(robot_idx[0], robot_idx[1], idxs[ii,0], idxs[ii,1])
                if len(rr)>10:
                    none_object_idxs = (np.stack((rr, cc)).astype(np.int))
                    bad_idxs = (none_object_idxs[:,0]>=self.x_shape) + (none_object_idxs[:,0]<0) +(none_object_idxs[:,1]>=self.y_shape) + (none_object_idxs[:,1]<0)
                    none_object_idxs = np.delete(none_object_idxs,np.where(bad_idxs),axis=0)
                    self.map[none_object_idxs[0], none_object_idxs[1]] += -30
                    self.map[self.map<-1] = 0
                    self.map[idxs[ii,0],idxs[ii,1]] = 100
                    #self.map[robot_idx[0],robot_idx[1]] = 100
                    #self.map[0,0] = 100
        #self.objects_map = (np.argwhere(self.map==100)*self.res) - self.initial_pose - 0.5*self.res*np.ones(2)
        self.objects_map = np.argwhere(self.map==100)
        self.objects_map[:,0] = self.objects_map[:,0]*self.res - self.initial_pose[0] 
        self.objects_map[:,1] = (self.y_shape - self.objects_map[:,1])*self.res - self.initial_pose[1]
        self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.objects_map)
        print 'Map is updated!'
        
        
    
    def get_Likelihood(self,robot_origin,scan,objects_map,objects_map_knn,s):
        z_star = self.scan2cart(robot_origin,scan,20)
        _, indices = objects_map_knn.kneighbors(z_star.T)
        z = objects_map[indices].reshape(z_star.shape)
        return np.prod(np.exp(-s* np.linalg.norm(z_star-z,axis=1)**2))



    def cart2ind(self,Z):
        #print Z.shape ,self.initial_pose.shape
        idxs = ((Z.T + self.initial_pose) // self.res).astype(int)
        idxs[:,1] = -idxs[:,1] + self.y_shape
        #print idxs[:,0]
        bad_idxs = (idxs[:,0]>=self.x_shape) + (idxs[:,0]<0) +(idxs[:,1]>=self.y_shape) + (idxs[:,1]<0)
        idxs = np.delete(idxs,np.where(bad_idxs),axis=0)
        if idxs.shape[1] == 2:
            return idxs
        else:
            print 'Bad idxs'



    
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
        
def main():

    rospy.init_node('SLAM', anonymous = True)
    PF = ParticleFilterRB(Np=100,max_rays = 100,grid_size = [30,30],initial_pose = [15,15])
    PF.init()
    PF.update_map(scan = PF.scan.z, initialize=1)
    r = rospy.Rate(5)
    PF.publish_occupancy_grid()
    
    while not rospy.is_shutdown():
        PF.publish_occupancy_grid()
        print 'runing!'
        r.sleep()
        PF.publish_particles()
        if PF.i_MU[0] > 0.01 or PF.i_MU[1] > 0.01:
            #print np.mean(PF.particles,axis = 0).shape
            PF.update_map(X = np.mean(PF.particles,axis = 0),scan = PF.scan.z, initialize=0)
            PF.i_MU = [0.0,0.0]
            print 'updated map'
            
  

    rospy.spin()


if __name__ == "__main__":
    main()

