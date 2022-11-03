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

timestep_limit_per_episode = 100000 # Can be any Value

register(
        id='ObstacleAvoidance-v0',
        entry_point='obstacle_avoidance:ObstacleAvoidance',
        max_episode_steps=timestep_limit_per_episode,
    )

class ObstacleAvoidance(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        """
        This task environment is designed for the Turtlebot2
        to visit waypoints
        """

        # subscribe robot pose
        self.robot_pose = Odometry()
        rospy.Subscriber("/odom", Odometry, self.odom_cb)

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)

        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-np.inf, np.inf)

        self.obstacle_pose_radius = rospy.get_param("/turtlebot2/obstacle_pose_radius", 0)
        self.num_of_sector = rospy.get_param("/turtlebot2/num_of_sector", 1)
        self.sensitiviy_distances = rospy.get_param("/turtlebot2/sensitiviy_distances")
       
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()

        obs_size = int(len(self.laser_scan.ranges)/ self.num_of_sector)
        high = np.full((obs_size), len(self.sensitiviy_distances))
        low = np.full((obs_size), 0)
        
        # We only use two integers
        self.observation_space = spaces.Box(low, high)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.velocity = rospy.get_param('/turtlebot2/velocity')
        self.robot_movement_area = rospy.get_param('/turtlebot2/robot_movement_area')
        self.goal_point = rospy.get_param('/turtlebot2/goal_point')
        self.nsteps = rospy.get_param('/turtlebot2/nsteps')

        # Rewards
        self.area_violation_reward = rospy.get_param('turtlebot2/area_violation_reward')
        self.goal_reached_reward = rospy.get_param('turtlebot2/goal_reached_reward')
        self.collsion_detected_reward = rospy.get_param('turtlebot2/collsion_detected_reward')

        self.cumulated_steps = 0.0
        self.is_area_violated = False
        self.is_collision_detected = False
        self.is_goal_reached = False
        self._nsteps_done = False

        # Here we will add any init functions prior to starting the MyRobotEnv
        super(ObstacleAvoidance, self).__init__()

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
        self.is_area_violated = False
        self.is_collision_detected = False
        self.is_goal_reached = False
        self._nsteps_done = False
        
        # set obtacle pose randomly
        # self.set_model_state("unit_box_red",(0, 0) ,self.obstacle_pose_radius)
        self.set_model_state("unit_box_red_0",(8, -8), 0.5)
        self.set_model_state("unit_box_red_1",(9, 6), 0.5)
        self.set_model_state("unit_box_red_2",(-5, 9), 0.5)
        self.set_model_state("unit_box_red_3",(-7, -7), 0.5)

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
        # We convert the actions to speed movements to send to the parent class CubeSingleDiskEnv
        
        _linear_speed = 0.0
        _angular_speed = 0.0

        if action == 0: #FORWARD
            _linear_speed = self.velocity['linear']['x']
            _angular_speed = 0.0
            self.last_action = "FORWARD"

        # if action == 1: #STOP
        #     _linear_speed = 0.0
        #     _angular_speed = 0.0
        #     self.last_action = "STOP"

        if action == 1: #TURN LEFT
            _linear_speed = 0.0
            _angular_speed = self.velocity['angular']['z']
            self.last_action = "TURN_LEFT"

        if action == 2: #TURN RIGHT
            _linear_speed = 0.0
            _angular_speed = -1 * self.velocity['angular']['z']
            self.last_action = "TURN_RIGHT"

        # if action == 4: #BACKWARD
        #     _linear_speed = -1 * self.velocity['linear']['x']
        #     _angular_speed = 0.0
        #     self.last_action = "BACKWARD"
        

        # We tell TurtleBot2 the linear and angular speed to set to execute
        self.move_base(_linear_speed, _angular_speed, epsilon=0.05, update_rate=10)
        
        rospy.logdebug("END Set Action ==>"+str(action))
   
    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        TurtleBot2Env API DOCS
        :return:
        """
        rospy.logdebug("Start Get Observation ==>")
        
        scan = self.get_laser_scan()
        
        self.observations = self.discretize_observation(scan)
        
        rospy.logdebug("Observations==>"+str(self.observations))
        rospy.logdebug("END Get Observation ==>")
        
        return self.observations

                                                             
    def _is_done(self, observations):
        # nothing learned
        if self.cumulated_steps == self.nsteps -1:
            self._episode_done = True
            self._nsteps_done = True
        # print("robot pose: {}".format(self.robot_pose))

        # goal has reached                   
        distance_to_goal  = np.sqrt(pow(self.robot_pose.pose.pose.position.x - self.goal_point['x'], 2) + pow(self.robot_pose.pose.pose.position.y - self.goal_point['y'], 2))
        if distance_to_goal < self.sensitiviy_distances[len(self.sensitiviy_distances) -2]:
            self._episode_done = True
            self.is_goal_reached = True

        # collision detected
        for i, item in enumerate(self.observations):
            if item == len(self.sensitiviy_distances) -2:
                self._episode_done = True
                self.is_collision_detected = True
                
        # area violation
        if self.robot_pose.pose.pose.position.x > self.robot_movement_area['max']['x'] or self.robot_pose.pose.pose.position.y > self.robot_movement_area['max']['y'] or self.robot_pose.pose.pose.position.x < self.robot_movement_area['min']['x'] or self.robot_pose.pose.pose.position.y < self.robot_movement_area['min']['y']:
            self._episode_done = True
            self.is_area_violated = True
        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0
        distance_to_goal  = np.sqrt(pow(self.robot_pose.pose.pose.position.x - self.goal_point['x'], 2) + pow(self.robot_pose.pose.pose.position.y - self.goal_point['y'], 2))
        
        if self.is_area_violated:
            reward += self.area_violation_reward *distance_to_goal * 0.05

        if self.is_collision_detected:
            reward+= self.collsion_detected_reward

        if self.is_goal_reached:
            reward+= self.goal_reached_reward

        # time reward
        reward -= 0.2
         
        rospy.logdebug("reward=" + str(reward))
        self.cumulated_reward += reward
        rospy.logdebug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        rospy.logdebug("Cumulated_steps=" + str(self.cumulated_steps))
        return reward

    def odom_cb(self, msg):
        self.robot_pose = msg

    def get_random_x_y(self, radius):
        x = np.random.uniform(-radius, radius)
        y = sqrt(pow(radius, 2) - pow(x,2))
        return x, y

    def set_model_state(self, model_name, center, radius):
        rospy.wait_for_service("/gazebo/set_model_state")
        res = SetModelStateResponse()
        while not res.success:
            # random pose for obstacle
            rand_x, rand_y = self.get_random_x_y(radius)
            try:
                set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                req = SetModelStateRequest()
                req.model_state.model_name = model_name
                req.model_state.pose.position.x = center[0] + rand_x
                req.model_state.pose.position.y = center[1] + rand_y
                res = set_model_state(req)
                rospy.logwarn(res)
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)

    def discretize_observation(self, scan):
        """
        Discards all the laser readings that are not multiple in index of number_of_sectors
        value.
        """

        obs_size = int(len(scan.ranges)/ self.num_of_sector)
        
        current_sector = -1
        
        obs = [0] * self.num_of_sector
        
        min_dist = float('inf')
        
        for i, item in enumerate(scan.ranges):
            if i%obs_size == 0:
                current_sector+= 1
                min_dist = float('inf')

            if item < min_dist:
                min_dist = item
                for j, sensitiviy_distance  in enumerate(self.sensitiviy_distances):
                    if item >= sensitiviy_distance:
                        # print("sector: {} item: {}".format(current_sector, j))
                        obs[current_sector] = j
                        break

        return obs