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
        entry_point='testenv:LocalPlannerWorld',
        max_episode_steps=timestep_limit_per_episode,
    )


class LocalPlannerWorld(turtlebot2_env.TurtleBot2Env):
    def __init__(self):

        # action spaces
        self.action_spaces_value = [-0.5, 0.5]

        number_actions = len(self.action_spaces_value)
        self.action_space = spaces.Discrete(number_actions)
        
        # state spaces
        low = np.full(1, -50)
        high = np.full(1, 50)
        
        self.observation_space = spaces.Box(low, high)
        print("observation spaces: {}".format(self.observation_space))
        
        # check subs and pubs
        self.odom = self._check_odom_ready()

        self.nsteps = 25
        
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
        time.sleep(1.0)

        # For Info Purposes
        self.cumulated_reward = 0.0
        self.cumulated_steps = 0.0

        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.is_dist_exceed = False
        self.is_goal_reached = False
        self.nsteps_done = False

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        
        rospy.logdebug("Start Set Action ==>"+str(action))

        _linear_speed = self.action_spaces_value[action]

        print(f'actions x:{_linear_speed}')
        self.move_base(_linear_speed, 0)

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
             
        self.observations = [self.odom.pose.pose.position.x]

        print("observations: {}".format(self.observations))
        return self.observations
                                                             
    def _is_done(self, observations):
        self._episode_done = False

        # goal has reached        
        dist = abs(5.0-self.odom.pose.pose.position.x)
        
        if dist< 0.5:
            print("goal has reached")
            self.is_goal_reached = True
            self._episode_done = True

        # when it's too far from global plan
        if 25 <= observations[0] or observations[0] <= -25:
            print("too far from global plan")
            self.is_dist_exceed = True
            self._episode_done = True

        if self.cumulated_steps == self.nsteps-1:
            print("steps end")
            self._episode_done = True
            self.nsteps_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0

        if self.is_goal_reached:
            reward+= 200

        dist = abs(5.0-self.odom.pose.pose.position.x)

        reward-=dist*0.1

        # time factor
        if not done:
            reward-= 0.2

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        print(f'cumulated reward: {self.cumulated_reward}, reward: {reward}')
        return reward