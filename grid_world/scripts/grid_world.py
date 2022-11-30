
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
        id='GridWorld-v0',
        entry_point='grid_world:GridWorld',
        max_episode_steps=timestep_limit_per_episode,
    )

class GridWorld(turtlebot2_env.TurtleBot2Env):
    def __init__(self):
        """
        Explain what you do
        """

        # Only variable needed to be set here
        number_actions = rospy.get_param('/turtlebot2/n_actions')
        self.action_space = spaces.Discrete(number_actions)
       
        # We create two arrays based on the binary values that will be assigned
        # In the discretization method.
        laser_scan = self._check_laser_scan_ready()

        self.grid_size = Point() 
        self.grid_size.x = self.grid_size.y = 4

        self.resolution = 0.5
        self.angle_resolution = 10 # degree
        # We only use two integers [x, y, yaw]
        high = np.concatenate((np.full((2), self.grid_size.x/self.resolution), np.full((1), 360/self.angle_resolution+1)))
        low = np.concatenate((np.full((2), -self.grid_size.x/self.resolution), np.full((1), -360/self.angle_resolution)))
        
        self.observation_space = spaces.Box(low, high)
        
        rospy.logdebug("ACTION SPACES TYPE===>"+str(self.action_space))
        rospy.logdebug("OBSERVATION SPACES TYPE===>"+str(self.observation_space))

        self.velocity = rospy.get_param('/turtlebot2/velocity')
        self.goal_point = rospy.get_param('/turtlebot2/goal_point')
        self.nsteps = rospy.get_param('/turtlebot2/nsteps')

        self.goal_grid = [int((self.goal_point['x'])/self.resolution), int((self.goal_point['y'])/self.resolution)]
        
        # Rewards
        self.area_violation_reward = rospy.get_param('turtlebot2/area_violation_reward')
        self.goal_reached_reward = rospy.get_param('turtlebot2/goal_reached_reward')
        self.collsion_detected_reward = rospy.get_param('turtlebot2/collsion_detected_reward')

        self.cumulated_steps = 0.0
        self.is_area_violated = False
        self.is_collision_detected = False
        self.is_goal_reached = False
        self.nsteps_done = False

        self._check_odom_ready()
        self._robot_pose = self.get_odom()


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(GridWorld, self).__init__()

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
        self.nsteps_done = False
        
        # set obtacle pose randomly
        # self.set_model_state("unit_box_red",(0, 0) ,self.obstacle_pose_radius)
        self.set_model_state("unit_box_red_0",(20, -20), 0.5)
        self.set_model_state("unit_box_red_1",(20, 20), 0.5)
        self.set_model_state("unit_box_red_2",(-20, 20), 0.5)
        self.set_model_state("unit_box_red_3",(-20, -20), 0.5)

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

        if action == 0: # i+1, j
            _linear_speed = self.velocity['linear']['x']
            self.last_action = "i+1, j"

        elif action == 1: # i-1, j
            _angular_speed = self.velocity['angular']['z']
            self.last_action = "i-1, j"

        elif action == 2: # i, j+1
            _angular_speed = -1* self.velocity['angular']['z']
            self.last_action = "i, j+1"
        
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
        # rospy.logdebug("Start Get Observation ==>")
        odom = self.get_odom()
        quat = [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]
        _, _, yaw = euler_from_quaternion(quat)
        yaw = int(math.degrees(yaw)/self.angle_resolution)
        x, y = self.odom_to_grid(odom)
        self.observations = [x, y, yaw]

        return self.observations

                                                             
    def _is_done(self, observations):
        self._episode_done = False

        # goal has reached                   
        if observations == self.goal_grid:
            print("goal has reached")
            self.is_goal_reached = True
            self._episode_done = True

        # collision detected
        scan = self.get_laser_scan()
        if min(scan.ranges)<= 0.4:
            print("min scan done")
            self.is_collision_detected = True
            self._episode_done = True

        # area violation
        if observations[0]<=int(-1*self.grid_size.x/self.resolution) or observations[0]>= int(self.grid_size.x/self.resolution) or observations[1]<=int(-1*self.grid_size.y/self.resolution) or observations[1]>= int(self.grid_size.y/self.resolution):
            print("area violation done")
            self.is_area_violated = True
            self._episode_done = True

        if self.cumulated_steps == 299:
            print("step end")
            self._episode_done = True
            self.nsteps_done = True
        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0
        if self.is_goal_reached:
            reward+= 100
        if self.is_collision_detected:
            reward-= 50
        if self.is_area_violated:
            reward-=20
        
        dist = math.sqrt((observations[0] - self.goal_grid[0])**2 + (observations[1] - self.goal_grid[1])**2)
        
        if dist!=0:
            reward+= 5.0/dist
        else:
            reward+=5.0
        
        reward-=0.2
         
        self.cumulated_reward += reward
        self.cumulated_steps += 1
        return reward


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
    
    def odom_to_grid(self, odom):
        x = int((odom.pose.pose.position.x)/self.resolution)
        y = int((odom.pose.pose.position.y)/self.resolution)
        return [x, y]