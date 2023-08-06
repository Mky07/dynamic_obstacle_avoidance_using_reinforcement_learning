#!/usr/bin/env python3

"""
Yapilacaklar
* [+] scan verisinin ilk değerlerini en sona ekle
* [+] robot açisinin limitini kaldir
* [+] aci arttikca negatif ödül ver.
* [+] aksiyon uzaya (0,0) ve (0, theta) açilarini ekle
"""

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

from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse

from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse
from utils import create_action_spaces
from preprocessing import ScanPreProcessing, RobotPreProcessing

timestep_limit_per_episode = 100000 # Can be any Value

register(
        id='LocalPlannerWorld-v4',
        entry_point='env3:LocalPlannerWorld',
        max_episode_steps=timestep_limit_per_episode,
    )


class LocalPlannerWorld(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        self.nsteps = 9999
        self.scan_ranges = 360
        self.scan_padding = 25
        n_scan_states = self.scan_ranges + self.scan_padding

        max_range = 30 #m
        max_dist = 2 # m
        self.look_ahead_dist = 3 #m
        self.closed_obstacle_dist = 0.2
        # Limits
        self.goal_th = 0.2
        self.dist_th = max_dist - 0.1
        # self.angle_th = 2.4434609528 # 90 deg

        # action spaces
        self.action_spaces_value = create_action_spaces(1.0, 0.4, 20, 10)
        number_actions = len(self.action_spaces_value)
        self.action_space = spaces.Discrete(number_actions)
        
        # state spaces
        self.scan_preprocessing =ScanPreProcessing(n_scan_states,max_range, self.scan_padding)
        self.robot_preprocessing = RobotPreProcessing(self.look_ahead_dist)

        s1_l = np.full(1, -max_dist)
        s2_l = np.full(1, -pi)
        s3_l = np.full(n_scan_states, 0.0)

        s1_h = np.full(1, max_dist)
        s2_h = np.full(1, pi)
        s3_h = np.full(n_scan_states, max_range)
        
        high = np.concatenate((s1_h, s2_h, s3_h))
        low = np.concatenate((s1_l, s2_l, s3_l))
        self.observation_space = spaces.Box(low, high)
        print("observation spaces: {}".format(self.observation_space))
        # state spaces end

        # check subs and pubs
        self.odom = self._check_odom_ready()
        self.scan = self._check_laser_scan_ready()
        
        print(f"scan range :{len(self.scan.ranges)}")
        # rospy.wait_for_service("/gazebo/set_model_state")

        super(LocalPlannerWorld, self).__init__()

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        self.move_base( 0.0, 0.0, epsilon=0.05, update_rate=10)
        
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # This is necessary to give the laser sensors to refresh in the new reseted position.
        time.sleep(1.5)

        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.is_dist_exceed = False
        # self.is_angle_exceed = False
        self.is_goal_reached = False
        self.nsteps_done = False
        self.is_collision_detected=False

        # create global plan
        self.global_plan = Path()
        while len(self.global_plan.poses) == 0:
            self.goal = self.create_random_goal()
            self.global_plan = self.get_global_path(self.goal)

        # Only for Visualization
        # goal_x, goal_y =self.global_plan.poses[-1].pose.position.x, self.global_plan.poses[-1].pose.position.y
        # self.set_model_state("Goal_Point",goal_x, goal_x, goal_y, goal_y)        

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))

        _linear_speed = self.action_spaces_value[action][0]
        _angular_speed = self.action_spaces_value[action][1]

        print(f'velocity x:{_linear_speed} z:{_angular_speed}')
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
        self.scan = self.get_laser_scan()
        
        scan_state = self.scan_preprocessing.get_states(self.scan)
        min_dist, theta = self.robot_preprocessing.get_states(self.global_plan, self.odom.pose.pose)

        self.observations = [min_dist, theta] + scan_state

        print("observations: {}".format(self.observations))
        return self.observations

                                                             
    def _is_done(self, observations):
        self._episode_done = False

        # collision detected
        if min(observations[2:])<= self.closed_obstacle_dist:
            print("min scan done")
            self.is_collision_detected = True
            self._episode_done = True

        # goal has reached        
        dist = sqrt((self.goal.pose.position.x-self.odom.pose.pose.position.x)**2 + (self.goal.pose.position.y-self.odom.pose.pose.position.y)**2)
        
        if dist< self.goal_th:
            print("goal has reached")
            self.is_goal_reached = True
            self._episode_done = True

        # when it's too far from global plan
        if observations[0] > self.dist_th:
            print("too far from global plan")
            self.is_dist_exceed = True
            self._episode_done = True

        # robot acısı çok fazla ise işlemi sonlandır.
        # if abs(observations[1]) > self.angle_th:
        #     print("Angle has exceeded")
        #     self.is_angle_exceed = True
        #     self._episode_done = True


        if self.cumulated_steps == self.nsteps-1:
            print("steps end")
            self._episode_done = True
            self.nsteps_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0

        # en yakın noktadan çok uzaktaysa
        reward -= observations[0] * 0.02
        
        # if observations[0] == 0:
        #     reward += 0.5 
        # elif observations[0] == 1:
        #     reward += 0.3 
        # else:
        #     reward+= 0.3/observations[0]

        # look ahead e göre robot açısı az ise ödül ver
        reward-= abs(observations[1]) * 0.05
        
        # if observations[1] == 0:
        #     reward += 0.2 
        # elif abs(observations[1]) == 1:
        #     reward += 0.1 
        # else:
        #     reward+= 0.1/abs(observations[1])

        # if self.is_angle_exceed:
        #     reward-= 40
        
        if self.is_collision_detected:
            reward-= 150
        if self.is_dist_exceed:
            reward-= 70
        if self.is_goal_reached:
            reward+= 200

        if not done:
            reward-= 0.03

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        print(f'cumulated reward: {self.cumulated_reward}, reward: {reward}')
        return reward


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

    def get_global_path(self, goal:PoseStamped):
        """
        return nav_msgs/Path
        """
        rospy.wait_for_service('/move_base/make_plan')
        try:
            get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
            req = GetPlanRequest()

            start = PoseStamped()
            start.header.frame_id = "map"
            start.pose = self.odom.pose.pose            
            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.pose = goal.pose
            req.start = start
            req.goal = goal_msg
            res = get_plan(req)

            return res.plan
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            raise(e)
    
    def create_random_goal(self):
        goal = PoseStamped()
        goal.pose.position = Point(np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0.0)  
        print(f"goal position:{goal}")
        return goal