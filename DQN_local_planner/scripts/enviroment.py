#!/usr/bin/env python3

import rospy
import time
import numpy as np
from numpy import sqrt
from gym import spaces
from openai_ros.robot_envs import turtlebot2_env
from gym.envs.registration import register
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry

from gazebo_msgs.srv import *
from gazebo_msgs.msg import ModelState
import math
from tf.transformations import euler_from_quaternion

timestep_limit_per_episode = 100000 # Can be any Value

register(
        id='LocalPlannerWorld-v0',
        entry_point='enviroment:LocalPlannerWorld',
        max_episode_steps=timestep_limit_per_episode,
    )


class LocalPlannerWorld(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        # check subs and pubs
        self.odom = self._check_odom_ready()
        self.laser_scan = self._check_laser_scan_ready()

        # action spaces
        number_actions = 3
        self.action_space = spaces.Discrete(number_actions)
        self.nsteps = 300

        # create global plan
        self.resolution = 0.5
        self.initial_point = [-3.9, 0.0]
        self.goal_point = [3.0, 0.0]
        self.global_plan1 = self.create_global_plan(self.initial_point, [0.0, 3.0], self.resolution)
        self.global_plan2 = self.create_global_plan([0.0, 3.0], self.goal_point, self.resolution)
        self.global_plan = self.global_plan1 + self.global_plan2

        print("GLobal plan: {}".format(self.global_plan))
        
        # other params
        self.goal_th = 0.5
        self.over_dist = 10
        self.max_range = 30
        
        # models input = global plan [size = look ahead dist] | sector_size  => total state = 3 + 36
        
        # global plan state
        self.look_ahead_dist_index = 3
        closed_dist_index,_ = self.closed_dist_index()

        robot_to_plan_dist = self.robot_to_plan_dist()
        partial_dist_list = self.get_partial_robot_to_plan_dist(robot_to_plan_dist,closed_dist_index)

        # partial_dist_list = self.get_partial_robot_to_plan_dist(self.robot_to_plan_dist(),closed_dist_index)

        # scan state
        self.sector_size = 36
        self.scan_range_resolution = 0.5
        scan_state = self.adjust_scan_state()


        # concatenate states
        high = np.concatenate((np.full(len(partial_dist_list), 100), np.full(self.sector_size, self.max_range/self.scan_range_resolution)))
        low = np.concatenate((np.full(len(partial_dist_list), -1), np.full(self.sector_size, 0)))
        self.observation_space = spaces.Box(low, high)

        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.cumulated_steps = 0.0
        self.is_over_dist = False
        self.is_goal_reached = False
        self.nsteps_done = False
        self.is_collision_detected=False


        print("observation spaces: {}".format(self.observation_space))

        super(LocalPlannerWorld, self).__init__()

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
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
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False
        self.cumulated_steps = 0.0
        self.is_over_dist = False
        self.is_goal_reached = False
        self.nsteps_done = False
        self.is_collision_detected=False

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
        closed_dist_index, _ = self.closed_dist_index()
        robot_to_plan_dist = self.robot_to_plan_dist()
        partial_dist_list = self.get_partial_robot_to_plan_dist(robot_to_plan_dist,closed_dist_index)
        
        scan_state = self.adjust_scan_state()

        self.observations = partial_dist_list + scan_state
        return self.observations

                                                             
    def _is_done(self, observations):
        self._episode_done = False

        # collision detected
        if min(observations[self.sector_size:])<= 2:
            print("min scan done")
            self.is_collision_detected = True
            self._episode_done = True

        # goal has reached  
        odom = self.get_odom()
        dist = math.sqrt((self.goal_point[0]-odom.pose.pose.position.x)**2 + (self.goal_point[1]-odom.pose.pose.position.y)**2)
        
        if dist< self.goal_th:
            print("goal has reached")
            self.is_goal_reached = True
            self._episode_done = True

        # when it's too far from global plan        
        if observations[0] > self.over_dist:
            print("too far from global plan")
            self.is_over_dist = True
            self._episode_done = True

        if self.cumulated_steps == self.nsteps-1:
            print("steps end")
            self._episode_done = True
            self.nsteps_done = True

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0

        dist_state = observations[:self.look_ahead_dist_index] 
        sum_dist = 0
        count_dist = 0
        for i, dist in enumerate(dist_state):
            if dist == -1:
                continue
            sum_dist+= dist
            count_dist+=1

        if sum_dist==0 or count_dist==0:
            pass
        else:
            reward-= 0.01*(sum_dist/count_dist)

        if done:
            if self.is_collision_detected:
                reward-= 100
            if self.is_goal_reached:
                reward+=300
            if self.is_over_dist:
                reward-=50

        self.cumulated_reward += reward
        self.cumulated_steps += 1
        
        print("cumulated reward: {}, reward: {}".format(self.cumulated_reward,reward))
        return reward


    def create_global_plan(self, initial_point, goal_point, resolution):
        """
        Linear Bezier Curve
        B(t) = (1-t)*P0 + t* P1, 0<= t <= 1 
        """
        t = resolution / math.sqrt((goal_point[0] - initial_point[0])**2 + (goal_point[1] - initial_point[1])**2)
        
        plan = []

        i=0
        while i<=1:
            px = (1-i)*initial_point[0] + i*goal_point[0]
            py = (1-i)*initial_point[1] + i*goal_point[1]
            plan.append([px, py])

            i+= t
        
        return plan
 
    def get_partial_robot_to_plan_dist(self, dist_list, closed_dist_index):
        """
        robota en yakin nokta bulunmustu.
        bu noktadan itibaren n tane nokta seciliyor.
        Bu noktalar state olarak ekleniyor.
        hedefe yaklastigimizda yeterli nokta sayisi kalmadigindan '-1' ekliyoruz.
        """
        print("closed_dist_index: {}".format(closed_dist_index))
        partial_dist_list = dist_list[closed_dist_index:(closed_dist_index+self.look_ahead_dist_index)]
        
        if len(partial_dist_list) < self.look_ahead_dist_index:        
            for i in range(self.look_ahead_dist_index - len(partial_dist_list)):
                partial_dist_list.append(-1)
        
        print("partial dist list:{}".format(partial_dist_list))
        return partial_dist_list
    
    def robot_to_plan_dist(self):
        """
        robot ile global rota arasindaki mesafeler bulunuyor.
        Bunun avantaji global rotayi (x,y) gibi lokal degiskenden kurtarmaktir.
        """
        odom = self.get_odom()    

        plan_dist = []
        for i, plan in enumerate(self.global_plan):
            dist = math.sqrt((plan[0]-odom.pose.pose.position.x)**2 + (plan[1]-odom.pose.pose.position.y)**2)
            plan_dist.append(round(dist/self.resolution))
        print("plan dist list:{}".format(plan_dist))
        return plan_dist

    def closed_dist_index(self):
        """
        global rotanin robota en yakin noktasi bulunuyor.
        bu noktanin indexi donduruluyor
        """
        min_dist = float("inf")

        robot_to_plan_dist = self.robot_to_plan_dist()

        closed_index = 0
        for i, dist in enumerate(robot_to_plan_dist):
            if dist < min_dist:
                closed_index = i
                min_dist = dist

        print("closed index:{} min_dist:{}".format(closed_index, min_dist))
        return [closed_index, min_dist]

    
    def adjust_scan_state(self):
        """
        scan verisi ilk olarak sektorlere ayrildi.
        her sektordeki en yakin mesafe bulundu
        mesafe cozunurluge bolundu.
        """

        scan = self.get_laser_scan()
        scan_len = int((scan.angle_max - scan.angle_min)/ scan.angle_increment)

        scan_sector_resolution = int(scan_len/self.sector_size)
        
        scan_state = [self.max_range/self.scan_range_resolution] * self.sector_size

        min_scan = self.max_range

        sector_count = 0
        idx = 0
        for i, var in enumerate(scan.ranges):
            # find min scan
            if var <= min_scan and idx < self.sector_size:
                scan_state[idx] = int(var/self.scan_range_resolution)
                min_scan = var

            sector_count+=1
            if sector_count == scan_sector_resolution:
                idx+=1
                sector_count = 0
                min_scan = self.max_range


        print("scan state:{}".format(scan_state))
        return scan_state