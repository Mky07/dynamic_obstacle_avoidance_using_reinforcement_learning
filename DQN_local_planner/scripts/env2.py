#!/usr/bin/env python3

import rospy
import time
import numpy as np
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import *
from gazebo_msgs.msg import ModelState
from math import sqrt, atan2, pi, ceil
import xml.etree.ElementTree as ET

from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse

import tf2_ros
# import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler

timestep_limit_per_episode = 100000 # Can be any Value

register(
        id='LocalPlannerWorld-v1',
        entry_point='env2:LocalPlannerWorld',
        max_episode_steps=timestep_limit_per_episode,
    )


class LocalPlannerWorld(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        # check subs and pubs
        self.odom = self._check_odom_ready()
        self.scan = self._check_laser_scan_ready()

        # action spaces
        number_actions = 3
        self.action_space = spaces.Discrete(number_actions)
        self.nsteps = 300

        # state spaces
        #x, y
        max_dist = 3 #m
        self.dist_resolution = 0.2
        self.angle_resolution = 0.3490658504 # 20 deg
        self.sector_resolution = 30
        discrete_range_map, sector_size = self.get_discrete_range_map(range_max=30)

        s1_h = np.full(1, self.to_discrete(max_dist, self.dist_resolution)) # min_dist 
        s2_h = np.full(1, self.to_discrete(pi + self.angle_resolution, self.angle_resolution)) # theta
        s3_h = np.full(self.to_discrete(len(self.scan.ranges),self.sector_resolution), sector_size) # scan

        s1_l = np.full(1 , self.to_discrete(-max_dist, self.dist_resolution)) # min_dist
        s2_l = np.full(1, self.to_discrete(-pi - self.angle_resolution, self.angle_resolution)) # theta
        s3_l = np.full(self.to_discrete(len(self.scan.ranges), self.sector_resolution), 0) # scan

        high = np.concatenate((s1_h, s2_h, s3_h))
        low = np.concatenate((s1_l, s2_l, s3_l))
        self.observation_space = spaces.Box(low, high)
        print("observation spaces: {}".format(self.observation_space))

        ## tf2
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # self.goal_point_xml = ET.tostring(ET.parse("/home/muharrem/model_editor_models/Untitled/model.sdf").getroot(), encoding='unicode', method='xml')
        # self.path_point_green_xml = ET.tostring(ET.parse("/home/mky/model_editor_models/path_point_green/model.sdf").getroot(), encoding='unicode', method='xml')
        # self.path_point_yellow_xml = ET.tostring(ET.parse("/home/mky/model_editor_models/path_point_yellow/model.sdf").getroot(), encoding='unicode', method='xml')
        
        # create global plan
        self.goal = PoseStamped()
        self.global_plan = self.create_gobal_plan()
        # print("GLobal plan: {}".format(self.global_plan))
        
        self.look_ahead_dist = 1.0
        self.goal_th = 0.2
        self.over_dist = 2.5
        self.over_angle = 2.4434609528 # 90 deg
        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.is_over_dist = False
        self.is_over_angle = False
        self.is_goal_reached = False
        self.nsteps_done = False
        self.is_collision_detected=False

        # rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/set_model_state")

        # tmp = Pose()
        # tmp.orientation.w = 1.0
        # self.spawn_model("closed_pose", tmp, self.path_point_yellow_xml)


        super(LocalPlannerWorld, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        # self.set_model_state("waffle", -5, 2, -5, 5)

        self.move_base( 0.0,
                        0.0,
                        epsilon=0.05,
                        update_rate=10)

        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """

        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0
        
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.is_over_dist = False
        self.is_over_angle = False
        self.is_goal_reached = False
        self.nsteps_done = False
        self.is_collision_detected=False

        self.global_plan = self.create_gobal_plan()

        # start pose
        # self.set_model_state("waffle", -5, 0, -5, 5)
        # hedef nokta 

        goal_x, goal_y =self.global_plan.poses[-1].pose.position.x, self.global_plan.poses[-1].pose.position.y
        self.set_model_state("Untitled",goal_x, goal_x, goal_y, goal_y)        

        # engeli rastgele baslat
        self.set_model_state("unit_box_red", -0.5, 0.5, -2, 2)

        # for i, pose in enumerate(self.global_plan.poses):
        #     if i%5 == 0:
        #         self.spawn_model("pose_" + str(i), pose.pose, self.path_point_yellow_xml)
    
        self.set_model_state("unit_box_red_0", 150, 150, 150, 150)
        self.set_model_state("unit_box_red_1", 150, 150, 150, 150)
        self.set_model_state("unit_box_red_2", 150, 150, 150, 150)
        self.set_model_state("unit_box_red_3", 150, 150, 150, 150)

        # This is necessary to give the laser sensors to refresh in the new reseted position.
        rospy.logwarn("Waiting...")
        time.sleep(0.5)
        rospy.logwarn("END Waiting...")        

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))

        _linear_speed = 0.0
        _angular_speed = 0.0

        if action == 0: # i+1, j
            _linear_speed = 1.2
            self.last_action = "front"

        elif action == 1: # i-1, j
            _linear_speed = 0.3
            _angular_speed = 0.5
            self.last_action = "left"

        elif action == 2: # i, j+1
            _linear_speed = 0.3
            _angular_speed = -1* 0.5
            self.last_action = "right"

        self.move_base(_linear_speed, _angular_speed)

        # We tell TurtleBot2 the linear and angular speed to set to execute
        # 0.2 sn uyuyor
        # self.move_base(_linear_speed, _angular_speed, epsilon=0.05, update_rate=10)
        rospy.loginfo("end action")
        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        self.odom = self.get_odom()
        min_dist, theta = self.get_pose_wrt_relative_coordinate(self.global_plan, self.odom, self.look_ahead_dist)
        
        # continuous space to discrete space
        discrete_min_dist, discrete_robot_theta = self.to_discrete(min_dist, self.dist_resolution), self.to_discrete_ceil(theta - self.angle_resolution/2.0, self.angle_resolution)
        
        self.scan = self.get_laser_scan()
        scan_state = self.discrete_scan(self.scan, self.sector_resolution)

        self.observations = [discrete_min_dist, discrete_robot_theta] + scan_state

        print("observations: {}".format(self.observations))
        return self.observations

                                                             
    def _is_done(self, observations):
        self._episode_done = False

        # collision detected
        if min(observations[2:])<= 6: # 6 = 0.6 cm. calculated from range_map
            print("min scan done")
            self.is_collision_detected = True
            self._episode_done = True

        # goal has reached        
        odom = self.get_odom()
        
        dist = sqrt((self.goal.pose.position.x-odom.pose.pose.position.x)**2 + (self.goal.pose.position.y-odom.pose.pose.position.y)**2)
        
        if dist< self.goal_th:
            print("goal has reached")
            self.is_goal_reached = True
            self._episode_done = True

        # when it's too far from global plan
        if observations[0] > self.to_discrete(self.over_dist, self.dist_resolution):
            print("too far from global plan")
            self.is_over_dist = True
            self._episode_done = True

        # robot acısı çok fazla ise işlemi sonlandır.
        if abs(observations[1]) >= self.to_discrete_ceil(self.over_angle - self.angle_resolution/2.0, self.angle_resolution):
            print("robot angle too much")
            self.is_over_angle = True
            self._episode_done = True


        if self.cumulated_steps == self.nsteps-1:
            print("steps end")
            self._episode_done = True
            self.nsteps_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0

        # en yakın noktadan çok uzaktaysa
        if observations[0] == 0:
            reward += 0.5 
        elif observations[0] == 1:
            reward += 0.3 
        else:
            reward+= 0.3/observations[0]

        # look ahead e göre robot açısı az ise ödül ver
        if observations[1] == 0:
            reward += 0.2 
        elif abs(observations[1]) == 1:
            reward += 0.1 
        else:
            reward+= 0.1/abs(observations[1])

        if done:
            if self.is_collision_detected:
                reward-= 100
            if self.is_goal_reached:
                reward+=200
            if self.is_over_dist:
                reward-=50
            if self.is_over_angle:
                reward-=20

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        print("cumulated reward: {}, reward: {}".format(self.cumulated_reward,reward))
        return reward


    def to_discrete_ceil(self, var, resolution):
        return ceil(var/resolution)


    def to_discrete(self, var, resolution):
        return int(var/resolution)

    def create_gobal_plan(self):
        global_plan_resolution = 0.05

        start = PoseStamped()
        self.odom = self.get_odom()
        start.pose.position.x = self.odom.pose.pose.position.x
        start.pose.position.y = self.odom.pose.pose.position.y

        
        # goal rastgele seçiliyor. Buradaki parametreleri farklı bir yerden çağırmayı unutma!!!
        self.goal.pose.position = Point(np.random.uniform(2, 5), np.random.uniform(-4, 4), 0.0)  
        print("** goal: **\n{}".format(self.goal.pose.position))

        poses = []
        poses.append(start)
        poses.append(self.goal)

        return self.create_path(poses, global_plan_resolution)


    def create_path(self, poses, resolution = 0.2):

        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = rospy.Time.now()

        for i in range(len(poses)-1):
            path.poses += self.get_plan(poses[i], poses[i+1], resolution) 

        # print("global plan start:{} \n\n goal:{}".format(path.poses[0], path.poses[-1]))
        return path

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

    def get_pose_wrt_relative_coordinate(self, path:Path, robot_pose:Odometry, look_ahead_dist):
        """
        pure pursuit den esinlenilmistir.
        """
        # transform robot_pose to path's frame
        robot_pose_stamped = PoseStamped()
        robot_pose_stamped.header = robot_pose.header
        robot_pose_stamped.pose = robot_pose.pose.pose
        transformed_robot_pose = robot_pose_stamped

        # print("path start:{}\n robot_pose:{}".format(path.poses[0].pose.position, robot_pose.pose.pose.position))

        try: 
            transform = self.tf_buffer.lookup_transform(path.header.frame_id, robot_pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            
            # Kütüphane eklenemeddi. Bu yüzden geçici süreliğine yorum olarak kaldı.
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
        
        # self.set_model_state("closed_pose",closed_point.x, closed_point.x, closed_point.y, closed_point.y)

        # find look ahead dist point
        dist = 0
        look_ahead_point = Point()
        idx = closed_point_idx

        while(dist<= look_ahead_dist and idx< len(path.poses)):
            dist = sqrt((path.poses[closed_point_idx].pose.position.x - path.poses[idx].pose.position.x)**2 + (path.poses[closed_point_idx].pose.position.y - path.poses[idx].pose.position.y)**2)
            if (dist<=look_ahead_dist):
                look_ahead_point = path.poses[idx].pose.position
            idx+=1
        
        print("look ahead point:{}".format(look_ahead_point))
        # print("path: {}".format(path.poses[:5]))
        # print("path end: {}".format(path.poses[-5:]))
        # print("robot_pose:{}\n closed_pose:{}\n look ahead point:{}\n".format(robot_pose.pose.pose.position, closed_point, look_ahead_point))
        
        # create coordinate
        look_ahead_theta =  atan2(look_ahead_point.y - robot_y, look_ahead_point.x - robot_x) # range: [-pi, pi]

        # get (x,y,theta) w.r.t. this coordinate system
        theta_diff = robot_theta - look_ahead_theta
        return [min_dist, theta_diff]

    def discrete_scan(self, scan:LaserScan, sector_resolution):

        # robot açı çözünürlüklerini robot bakışındakilerin çözünürlüklerini arttır.
        discrete_range_map, state_size = self.get_discrete_range_map(scan.range_max)
        # print("map: {}".format(discrete_range_map))
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

        # print("state size:{} scan state:{}".format(state_size, scan_state))
        return scan_state

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

    def set_model_state(self, model_name, min_x, max_x, min_y, max_y):

        res = SetModelStateResponse()
        while not res.success:
            # random pose for obstacle
            try:
                set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                req = SetModelStateRequest()
                req.model_state.model_name = model_name
                req.model_state.pose.position.x = np.random.uniform(min_x, max_x)
                req.model_state.pose.position.y = np.random.uniform(min_y, max_y)
                res = set_model_state(req)
                rospy.logwarn(res)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

    def spawn_model(self, model_name, initial_pose, model_xml, reference_frame = "world"):
        res = SpawnModelResponse()
        # random pose for obstacle
        try:
            set_spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            req = SpawnModelRequest()
            req.model_name = model_name
            req.initial_pose = initial_pose
            req.model_xml = model_xml
            req.reference_frame = reference_frame
            res = set_spawn_model(req)
            rospy.logwarn(res)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)     
  

# koordinat düzelemini yeniden oluştur. # min_dist alındı. x,y kullanılmıyor.
# look_ahead açısını -10 +10 arası yap # eklendi
# x, y farklı ödül # min_dist alındı. x,y kullanılmıyor.
# theta acısı için ceza eklenebilir mi?