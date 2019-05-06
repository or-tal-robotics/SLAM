#!/usr/bin/env python 
import numpy as np
import rospy
from skimage.draw import line
from nav_msgs import OccupancyGrid
from tf_conversions.transformations import quaternion_from_euler


class occupancy_grid():
    def __init__(self, res = 0.1, grid_size = [10,10], initial_pose = [5,5], max_rays = 30):
        self.res, self.grid_size, self.initial_pose, self.max_rays = res, grid_size, initial_pose, max_rays
        self.x_shape = grid_size[0]//res
        self.y_shape = grid_size[1]//res
        self.map =-1*np.ones((self.x_shape,self.y_shape))
        self.map_publisher = rospy.Publisher('/map', OccupancyGrid, queue_size = 60)

    def update_map(self,X=[0,0,0],scan):
        Z = self.scan2cart(X,scan)
        idxs = self.cart2ind(Z)
        robot_idx = (X[0:2] + self.initial_pose) // self.res
        for ii in range(len(idxs)):
            rr, cc = line(robot_idx[0], robot_idx[1], idxs[ii,0], idxs[ii,1])
            self.map[rr, cc] += -30
            self.map[self.map<0] = 0
            self.map[idxs[ii,0],idxs[ii,1]] = 100

        self.objects_map = self.map[self.map==1]*self.res - self.initial_pose
        self.nbrs = KNN(n_neighbors=1, algorithm='ball_tree').fit(self.objects_map)
    
    def get_Likelihood(self,X,scan,s):
        z_star = scan2cart(X,scan)
        _, indices = self.nbrs.kneighbors(z_star)
        z = ob[indices].reshape(z_star.shape)
        return np.prod(np.exp(-s* np.linalg.norm(z_star-z,axis=1)**2))



    def cart2ind(self,Z):
        idxs = (Z + self.initial_pose) // self.res
        bad_idxs = (idxs[0,:]>self.x_shape) + (idxs[0,:]<0) +(idxs[1,:]>self.y_shape) + (idxs[1,:]<0)
        idxs = np.delete(idxs,np.where(bad_idxs),axis=1)
        return idxs



    
    def scan2cart(self,robot_origin,scan):
        # the robot orientation in relation to the world [m,m,rad]
        mu_x = robot_origin[0]
        mu_y = robot_origin[1]
        theta = robot_origin[2]  
        # converting the laserscan measurements from polar coordinate to cartesian and create matrix from them.
        n = len(scan.ranges); i = np.arange(len(scan.ranges))
        angle = np.zeros(n); x = np.zeros(n); y = np.zeros(n)
        rng = scan.ranges
        angle = np.add(scan.angle_min, np.multiply(i, scan.angle_increment)) + theta
        idxs = np.linspace(0,n-1,self.max_rays).astype(np.int)
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
        grid.data = self.map.astype(np.int8).ravel().tolist()
        grid.header.frame_id = '/world'
        grid.info.height = self.map.shape[0]
        grid.info.width = self.map.shape[1]
        grid.info.resolution = 1/self.res
        grid.info.origin.orientation = quaternion_from_euler(0, 0, 0)
        grid.info.origin.position.x = self.initial_pose[0]
        grid.info.origin.position.y = self.initial_pose[1]
        self.map_publisher.publish(grid)
