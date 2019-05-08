#!/usr/bin/env python 
import numpy as np
import rospy
from skimage.draw import line
from nav_msgs.msg import OccupancyGrid
import tf_conversions
from sklearn.neighbors import NearestNeighbors as KNN
from std_msgs.msg import Int8MultiArray
from NDT_or import ndt


class occupancy_grid():
    def __init__(self, res = 0.1, grid_size = [10,10], initial_pose = [5,5], max_rays = 30):
        self.res, self.grid_size, self.initial_pose, self.max_rays = res, grid_size, np.array(initial_pose), max_rays
        self.x_shape = int(grid_size[0]//res)
        self.y_shape = int(grid_size[1]//res)
        self.map =-1*np.ones((self.x_shape,self.y_shape))
        self.map_publisher = rospy.Publisher('/map', OccupancyGrid, queue_size = 60)

    def update_map(self,scan,X=[0,0,0],initialize = 0):
        Z = self.scan2cart(X,scan,self.max_rays)
        
        if initialize == 0:
            self.new_scan = Z
            T0 = X - self.last_pose
            Ndt = ndt(self.last_scan,self.new_scan,T0)
            T = Ndt.T
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
    
    def get_Likelihood(self,robot_origin,scan,s):
        z_star = self.scan2cart(robot_origin,scan,20)
        _, indices = self.nbrs.kneighbors(z_star.T)
        z = self.objects_map[indices].reshape(z_star.shape)
        #print np.prod(np.exp(-s* np.linalg.norm(z_star-z,axis=1)**2))
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
        #grid.info.origin.orientation = tf_conversions.transformations.quaternion_from_euler(0, 0, 0)
        grid.info.origin.position.x = -self.y_shape*self.res+self.initial_pose[1].astype(np.float)
        grid.info.origin.position.y = -self.initial_pose[0].astype(np.float) #
        self.map_publisher.publish(grid)
