#!/usr/bin/env python3

import rospy
import time
import numpy as np
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import *
from gazebo_msgs.msg import ModelState
from math import sqrt, atan2, pi, ceil, exp

from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse

from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse
from utils import create_action_spaces
from preprocessing import ScanPreProcessing, RobotPreProcessing

timestep_limit_per_episode = 100000 # Can be any Value

register(
        id='LocalPlannerWorld-v4',
        entry_point='DDPG_env:LocalPlannerWorld',
        max_episode_steps=timestep_limit_per_episode,
    )

class LocalPlannerWorld(turtlebot2_env.TurtleBot2Env):
    def __init__(self):

        # check subs and pubs
        self.odom = self._check_odom_ready()
        self.scan = self._check_laser_scan_ready()
        
        print(f"scan range :{len(self.scan.ranges)}")

        self.nsteps = 1000
        n_scan_states = len(self.scan.ranges)

        self.max_range = 3 #m
        self.look_ahead_dist = 1.5 #m
        self.closed_obstacle_dist = 0.3
        
        # Limits
        self.goal_th = 0.3
        self.dist_th = 1.5 # m

        # action spaces
        self.max_vx = 1.0
        self.max_wz = 0.4

        number_actions = 2

        vx_low = np.full(1, 0.1)
        vx_high = np.full(1, 1.0)
        wz_low = np.full(1, -0.4)
        wz_high = np.full(1, 0.4)
        high = np.concatenate((vx_high, wz_high))
        low = np.concatenate((vx_low, wz_low))

        self.action_space = spaces.Box(low, high)
        
        self.robot_preprocessing = RobotPreProcessing(self.look_ahead_dist, self.dist_th)

        min_dist_l = np.full(1, 0)
        theta_l = np.full(1, -2*pi)
        dist_diff_l = np.full(1, 0)
        scan_l = np.full(n_scan_states, 0.0)

        min_dist_h = np.full(1, self.dist_th)
        theta_h = np.full(1, 2*pi)
        dist_diff_h = np.full(1, self.look_ahead_dist)
        scan_h = np.full(n_scan_states, self.max_range)
        
        high = np.concatenate((min_dist_h, theta_h, dist_diff_h, scan_h))
        low = np.concatenate((min_dist_l, theta_l, dist_diff_l, scan_l))

        self.observation_space = spaces.Box(low, high)
        print("observation spaces: {}".format(self.observation_space))

        super(LocalPlannerWorld, self).__init__()

    def _set_init_pose(self):
        """
        Sets the Robot in its init pose
        """
        self.move_base( 0.0, 0.0, epsilon=0.05, update_rate=5)
        
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # This is necessary to give the laser sensors to refresh in the new reseted position.
        time.sleep(2)

        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.is_dist_exceed = False
        self.is_angle_exceed = False
        self.is_goal_reached = False
        self.nsteps_done = False
        self.is_collision_detected=False

        # create global plan
        self.global_plan = Path()
        self.global_plan.header.frame_id ="map"
        self.global_plan.header.stamp = rospy.Time.now()

        while len(self.global_plan.poses) == 0:
            self.goal = self.create_random_goal()
            self.global_plan = self.get_global_path(self.goal)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        rospy.logdebug("Start Set Action ==>"+str(action))
        _linear_speed = action[0]
        _angular_speed = action[1]

        print(f'actions x:{_linear_speed} z:{_angular_speed}')
        self.move_base(_linear_speed, _angular_speed)

        rospy.logdebug("END Set Action ==>"+str(action))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """

        """
        obs: [[L, theta, hedefe olan mesafe,vx, w, action x, action w, scans]
        """

        self.odom = self.get_odom()
        self.scan = self.get_laser_scan()
        
        # scan state
        scan_state = [min(x, self.max_range) for x in self.scan.ranges]

        # other state
        dist_to_goal = sqrt((self.goal.pose.position.x-self.odom.pose.pose.position.x)**2 + (self.goal.pose.position.y-self.odom.pose.pose.position.y)**2)
        dist_to_goal = min(self.look_ahead_dist, dist_to_goal)

        min_dist, theta, _ = self.robot_preprocessing.get_states(self.global_plan, self.odom.pose.pose)
        
        self.observations = [min_dist, theta, dist_to_goal] + scan_state

        # print("observations: {}".format(self.observations))
        return self.observations
                                                             
    def _is_done(self, observations):
        self._episode_done = False

        # collision detected
        if min(observations[3:])<= self.closed_obstacle_dist:
            print("min scan done")
            self.is_collision_detected = True
            self._episode_done = True
            return self._episode_done

        # goal has reached        
        dist_to_goal = sqrt((self.goal.pose.position.x-self.odom.pose.pose.position.x)**2 + (self.goal.pose.position.y-self.odom.pose.pose.position.y)**2)

        if dist_to_goal<= self.goal_th:
            print("goal has reached")
            self.is_goal_reached = True
            self._episode_done = True
            return self._episode_done 

        # when it's too far from global plan
        if observations[0] >=self.dist_th:
            print("too far from global plan")
            self.is_dist_exceed = True
            self._episode_done = True
            return self._episode_done

        # robot acısı çok fazla ise işlemi sonlandır.
        rad150 = (5/6)*pi
        if rad150<=observations[1] or observations[1]<=-rad150:
            print("Angle has exceeded")
            self.is_angle_exceed = True
            self._episode_done = True
            return self._episode_done

        if self.cumulated_steps == self.nsteps-1:
            print("steps end")
            self._episode_done = True
            self.nsteps_done = True
            return self._episode_done

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0
        
        ## rotaya uzaklık
        # r1 = (exp(-observations[0]+1.8)-1.406) # reward range: [0.82-4.644]
        r1 = 10*(-0.5*observations[0]+0.1) # [0, 2] -> [1, -9]
        reward+= r1

        ## angle
        # r2 = (-3*(observations[1]-0.5)**2+0.75) # reward range: [0.0, 0.75]
        r2 = -observations[1]**2 + 0.27 # [-2.61, +2.61] -> [0.27, -6.6]
        reward+= r2
        
        dist_to_goal = sqrt((self.goal.pose.position.x-self.odom.pose.pose.position.x)**2 + (self.goal.pose.position.y-self.odom.pose.pose.position.y)**2)

        r3=0.0
        if observations[2]<self.look_ahead_dist and observations[2]!=0:
            r3 = 2*self.look_ahead_dist/observations[2] # reward range: [2, 13.33]
        reward+= r3

        if self.is_goal_reached:
            reward+= 400
        if self.is_collision_detected:
            reward-= 50
        if self.is_dist_exceed:
            reward-= 200
        if self.is_angle_exceed:
            reward-= 200

        # time factor
        if not done:
            reward-= 0.02 # almost disable

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        # print(f'cumulated reward: {self.cumulated_reward}, reward: {reward} r1:{r1} r2:{r2} r3:{r3}')
        return reward

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
            start.pose.position = Point(0.0, 0.0, 0.0)
            # start.pose = self.odom.pose.pose  
            # start.pose.position = Point(np.random.uniform(-2, 2.0), np.random.uniform(-12.0, -8.0), 0.0)
            start.pose.orientation.w = 1.0

            goal_msg = PoseStamped()
            goal_msg.header.frame_id = "map"
            goal_msg.pose = goal.pose
            goal_msg.pose.orientation.w = 1.0
            req.start = start
            req.goal = goal_msg
            res = get_plan(req)

            return res.plan
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            raise(e)
    
    def create_random_goal(self):
        goal = PoseStamped()
        # goal.pose.position = Point(8.0, 0.0, 0.0)  # 1314 adım eğitildi 
        # goal.pose.position = Point(13.0, 0.0, 0.0)  # * adım eğitildi c3 1. adım 
        # goal.pose.position = Point(13.0, 0.0, 0.0)  # * adım eğitildi c3 1. adım 
        goal.pose.position = Point(13.0, 0.0, 0.0)  # * ddpg test 7 
        # goal.pose.position = Point(np.random.uniform(1.0, 4.0), np.random.uniform(-4.0, 4.0), 0.0) # 755 adım eğitildi.
        # goal.pose.position = Point(np.random.uniform(7.0, 9.0), np.random.uniform(-8.0, 8.0), 0.0) # 755 adım eğitildi.
        # print(f"goal position:{goal}")
        return goal