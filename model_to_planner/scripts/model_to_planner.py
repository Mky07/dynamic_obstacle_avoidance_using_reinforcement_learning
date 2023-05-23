from keras.models import load_model

import rospy
from rospkg import RosPack
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from math import sqrt, atan2, pi, ceil
from std_msgs.msg import String

import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

class ModelToPlanner:
    def __init__(self, model_name):

        #parameters
        self._scan = LaserScan()
        self._odom = Odometry()
        self._global_plan = Path()
        self._goal = PoseStamped()

        self.dist_resolution = 0.2
        self.angle_resolution = 0.3490658504 # 20 deg
        self.sector_resolution = 30
        self._look_ahead_dist = 1.0
        self.goal_th = 0.2
        self.over_dist = 2.5
        self.over_angle = 2.4434609528 # 90 deg

        ## tf2
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        ## keras ##
        parent_path = RosPack().get_path('model_to_planner') + '/models/'        
        
        # load model
        self._model = load_model(parent_path + model_name)

        # Subscribers
        rospy.Subscriber("scan", LaserScan, self.scan_cb, queue_size=10)
        rospy.Subscriber("odom", Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber("dqn_global_path", Path, self.path_cb, queue_size=10)
        rospy.Subscriber("move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=10)

        rospy.sleep(5)
        
        # Publishers
        self._cmd_vel_pub = rospy.Publisher("dqn_cmd_vel", Twist, queue_size=10)
        self._robot_status_pub = rospy.Publisher("dqn_robot_status", String, queue_size=10)
        
    def scan_cb(self, msg):
        self._scan = msg

    def odom_cb(self, msg):
        self._odom = msg

    def path_cb(self, msg):
        self._global_plan = msg

    def goal_cb(self, msg):
        self._goal = msg

    def robot_status(self):
        """
        DRIVING | GOAL_REACHED | COLLISON_DETECTED | OVER_LIMIT (ANGLE | DISTANCE) 
        """
        observations = self.get_state()
        print("obs: {}".format(observations))
        if len(observations)>=2:
            if self.is_goal_reached():
                return 'GOAL_REACHED'

            if self.is_collision_detected(observations):
                return 'COLLISON_DETECTED'
            
            # if self.is_over_limit(observations):
            #     return 'OVER_LIMIT'

        return 'DRIVING'

    def is_collision_detected(self, observations):
        if observations[2:] and min(observations[2:])<= 6: # 6 = 0.6 cm. calculated from range_map
            print("min scan done")
            return True
        return False

    def is_goal_reached(self):
        dist = sqrt((self._goal.pose.position.x-self._odom.pose.pose.position.x)**2 + (self._goal.pose.position.y-self._odom.pose.pose.position.y)**2)
        print("is goal reached dist: {}".format(dist))
        if dist< self.goal_th:
            print("goal has reached")
            return True
        return False

    def is_over_limit(self, observations):
        is_over_dist = observations[0] > self.to_discrete(self.over_dist, self.dist_resolution)
        is_over_angle = abs(observations[1]) >= self.to_discrete_ceil(self.over_angle - self.angle_resolution/2.0, self.angle_resolution)  

        if is_over_angle or is_over_dist:
            return True
        return False


    def get_state(self):
        scan_state = self.discrete_scan(self._scan, self.sector_resolution)

        min_dist, theta = self.get_pose_wrt_relative_coordinate(self._global_plan, self._odom, self._look_ahead_dist)
        
        # continuous space to discrete space
        discrete_min_dist, discrete_robot_theta = self.to_discrete(min_dist, self.dist_resolution), self.to_discrete_ceil(theta - self.angle_resolution/2.0, self.angle_resolution)

        state = [discrete_min_dist, discrete_robot_theta] + scan_state
        return state

    def action_space(self):
        move_forward = {
            'linear_speed': 1.2,
            'angular_speed': 0.0
        }

        turn_left =  {
            'linear_speed': 0.3,
            'angular_speed': 0.5
        }

        turn_right =  {
            'linear_speed': 0.3,
            'angular_speed': -0.5
        }
        return {0: move_forward, 1:turn_left, 2:turn_right}

    def to_velocity(self,action):
        cmd_vel = Twist()

        a_space = self.action_space() 
        tmp = a_space.get(action, None)
        cmd_vel.linear.x = tmp.get('linear_speed')
        cmd_vel.angular.z = tmp.get('angular_speed')
        return cmd_vel

    def get_velocity(self):
        if len(self._global_plan.poses)!=0:
            action = self.predict()
            print("action: {}".format(action))
            return self.to_velocity(action)
        return False

    def predict(self):
        state = self.get_state()
        print("state size: {}".format(len(state)))
        state = np.reshape(state, [1, len(state)])
        act_values = self._model.predict(state)
        return np.argmax(act_values[0])

    def publish(self):
        vel = self.get_velocity()
        # print("vel:{}".format(vel))
        if vel != False:
            self._cmd_vel_pub.publish(vel)

        if len(self._global_plan.poses)!=0:
            msg = String()
            msg.data = self.robot_status()
            self._robot_status_pub.publish(msg)
        else:
            msg = String()
            msg.data = "IDLE"
            self._robot_status_pub.publish(msg)

    def discrete_scan(self, scan, sector_resolution):
        print("scan: {}".format(scan.range_max))    
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

    def to_discrete_ceil(self, var, resolution):
        return ceil(var/resolution)

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

if __name__ == '__main__':
    try:
        rospy.init_node('model_to_planner', anonymous=True)
        rate = rospy.Rate(10) # 10hz
        # model name
        model_name = "model.h5"

        model_to_planner = ModelToPlanner(model_name)

        while not rospy.is_shutdown():
            model_to_planner.publish()
            rate.sleep()

        pass
    except rospy.ROSInterruptException:
        pass