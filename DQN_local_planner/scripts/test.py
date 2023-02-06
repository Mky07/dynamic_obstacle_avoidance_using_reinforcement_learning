#!/usr/bin/env python3

import rospy
import time
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

from math import sqrt, atan2
import numpy as np
import tf2_ros
# import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Test:
    def __init__(self):
        ## tf2
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.scan = LaserScan()
        rospy.Subscriber("odom", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("scan", LaserScan, self.scan_cb, queue_size=1)
    
        self.goal = PoseStamped()
    def scan_cb(self, msg):
        self.scan = msg

    def odom_cb(self, msg):
        path = self.create_gobal_plan()
        print("**path:**\n{}\n".format(path.poses[:10]))
        # start = PoseStamped()
        # start.pose.position = Point(-4.0, 0.0, 0.0)

        # goal = PoseStamped()
        # goal.pose.position = Point(3.0, 0.0, 0.0)  

        # poses = []
        # poses.append(start)
        # poses.append(goal)

        # path = self.create_path(poses, 0.2)

        # x, y, theta = self.get_pose_wrt_relative_coordinate(path, msg, 0.5)
        
        # get discrete x, y, theta
        # dist_resolution = 0.2
        # angle_resolution = 0.3
        # discrete_robot_x, discrete_robot_y, discrete_robot_theta = self.to_discrete(x, dist_resolution), self.to_discrete(y, dist_resolution), self.to_discrete(theta, angle_resolution)

        # print("discrete pose: [{}, {}, {}]".format(discrete_robot_x, discrete_robot_y, discrete_robot_theta))

        ss = self.discrete_scan(self.scan,  30)

    def to_discrete(self, var, resolution):
        return int(var/resolution)
    

    def get_pose_wrt_relative_coordinate(self, path:Path, robot_pose:Odometry, look_ahead_dist):
        """
        pure pursuit den esinlenilmistir.
        """
        # transform robot_pose to path's frame
        robot_pose_stamped = PoseStamped()
        robot_pose_stamped.header = robot_pose.header
        robot_pose_stamped.pose = robot_pose.pose.pose
        transformed_robot_pose = robot_pose_stamped

        try: 
            transform = self.tf_buffer.lookup_transform(path.header.frame_id, robot_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            # transformed_robot_pose = tf2_geometry_msgs.do_transform_pose(robot_pose_stamped, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("TF error.")

        # find robot's x,y, theta
        robot_x, robot_y = transformed_robot_pose.pose.position.x, transformed_robot_pose.pose.position.y 
        (_,_,robot_theta) = euler_from_quaternion([transformed_robot_pose.pose.orientation.x, transformed_robot_pose.pose.orientation.y, transformed_robot_pose.pose.orientation.z, transformed_robot_pose.pose.orientation.w]) 

        # find closed point
        closed_point = Point()
        closed_point_idx = 0
        min_dist = float('inf')

        for i, pose in enumerate(path.poses):
            dist = sqrt((pose.pose.position.x - robot_x)**2 + (pose.pose.position.y - robot_y)**2)
            if dist < min_dist:
                min_dist = dist
                closed_point = pose.pose.position
                closed_point_idx = i
        
        # find look ahead dist point
        dist = 0
        look_ahead_point = Point()
        idx = closed_point_idx
        print("idx start:{}".format(idx))
        while(dist<= look_ahead_dist and idx< len(path.poses)):
            print("idx1:{}".format(idx))
            dist = sqrt((path.poses[closed_point_idx].pose.position.x - path.poses[idx].pose.position.x)**2 + (path.poses[closed_point_idx].pose.position.y - path.poses[idx].pose.position.y)**2)
            print("dist: {}".format(dist))
            if (dist<=look_ahead_dist):
                print("idx2:{}".format(idx))
                look_ahead_point = path.poses[idx].pose.position
            idx+=1

        print("look ahead point: {}".format(look_ahead_point))

        # create coordinate
        look_ahead_theta =  atan2(look_ahead_point.y - robot_y, look_ahead_point.x - robot_x) # range: [-pi, pi]
        rospy.loginfo("robot theta: {} look ahead theta:{}".format(robot_theta, look_ahead_theta))

        # get (x,y,theta) w.r.t. this coordinate system
        x, y, theta = robot_x - look_ahead_point.x , robot_y - look_ahead_point.y, robot_theta - look_ahead_theta

        return x, y, theta


    def get_discrete_range_map(self, range_max):
        """
        min_dist - max_dist -> resolution 
        0 - 0.2 -> 0.05
        0.2 - 1 -> 0.1
        1 - 5 -> 0.5
        5 - 15 -> 1
        15 - max_range -> 5
        """

        # max range, resolution, start index
        discrete_range_map = [[0.4, 0.05], [1.0, 0.1], [5.0, 0.5], [15.0, 1.0], [range_max, 5.0]]
        
        for i, m in enumerate(discrete_range_map):
            if i == 0:
                m.append(0)
            else:
                m.append(self.to_discrete(discrete_range_map[i-1][0], discrete_range_map[i-1][1]) + discrete_range_map[i-1][2])

        state_size = discrete_range_map[-1][2] + self.to_discrete(discrete_range_map[-1][0], discrete_range_map[-1][1])    
        return discrete_range_map, state_size

    def discrete_scan(self, scan:LaserScan, sector_resolution):
        discrete_range_map, state_size = self.get_discrete_range_map(scan.range_max)
        print("map: {}".format(discrete_range_map))
        scan_state = [state_size]*self.to_discrete(len(scan.ranges), sector_resolution)
        
        min_range = scan.range_max
        for i, var in enumerate(scan.ranges):
            idx = self.to_discrete(i, sector_resolution)

            if var < min_range:
                min_range = var

                for j, rm in enumerate(discrete_range_map):
                    if min_range <= rm[0]:
                        scan_state[idx] = self.to_discrete(min_range, rm[1]) + rm[2]
                        break
            
            if i%sector_resolution == 0:
                min_range = scan.range_max

        print("state size:{} scan state:{}".format(state_size, scan_state))
        return scan_state


    def create_gobal_plan(self):
        global_plan_resolution = 0.1

        start = PoseStamped()
        # self.odom = self.get_odom()
        start.pose.position.x = -3.9
        start.pose.position.x = 0.0

        print("****************** start ********************\n{}".format(start))
        
        # goal rastgele seçiliyor. Buradaki parametreleri farklı bir yerden çağırmayı unutma!!!
        self.goal.pose.position = Point(np.random.uniform(2, 5), np.random.uniform(-10, 10), 0.0)  

        poses = []
        poses.append(start)
        poses.append(self.goal)

        return self.create_path(poses, global_plan_resolution)


    def get_plan(self, p1:PoseStamped, p2:PoseStamped, resolution):
        """
        Linear Bezier Curve
        B(t) = (1-t)*P1 + t* P2, 0<= t <= 1 
        """
        
        t = resolution / sqrt((p2.pose.position.x - p1.pose.position.x)**2 + (p2.pose.position.y - p1.pose.position.y)**2)
        
        plan = []

        i=0
        while i<=1+t:
            p = PoseStamped()
            p.pose.position.x = (1-i)*p1.pose.position.x + i*p2.pose.position.x
            p.pose.position.y = (1-i)*p1.pose.position.y + i*p2.pose.position.y
            plan.append(p)

            i+= t
        return plan

    def create_path(self, poses, resolution = 0.2):

        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = rospy.Time.now()

        for i in range(len(poses)-1):
            path.poses += self.get_plan(poses[i], poses[i+1], resolution) 

        return path

    def create_gobal_plan(self):
        global_plan_resolution = 0.1

        start = PoseStamped()
        # self.odom = self.get_odom()
        start.pose.position.x = -3.9
        start.pose.position.y = 0.0
        print("****************** start ********************\n{}".format(start))

        start.pose.position
        
        # goal rastgele seçiliyor. Buradaki parametreleri farklı bir yerden çağırmayı unutma!!!
        self.goal.pose.position = Point(np.random.uniform(2, 5), np.random.uniform(-10, 10), 0.0)
        print("** start **\n{}".format(start))  
        print("** goal: **\n{}".format(self.goal.pose.position))
        
        poses = []
        poses.append(start)
        poses.append(self.goal)

        return self.create_path(poses, global_plan_resolution)

def listener():
    rospy.init_node('listener', anonymous=True)

    t = Test()
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
