import math

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import Pose, Point, PointStamped

from tf.transformations import euler_from_quaternion
from math import sqrt, atan2


class ScanPreProcessing(): 
    def __init__(self, sample_size = 400, max_range = 20, padding_size=50):
        self.sample_size = sample_size
        self.max_range = max_range
        self.padding_size = padding_size

        self.scan_info = LaserScan()

    def downsample(self, scan:LaserScan):
        """
        downsample
        """

        if self.sample_size == -1:
            return scan.ranges
            
        increment = len(scan.ranges)/self.sample_size
        idx = increment/2
        samples = []
        while len(samples) <self.sample_size:
            samples.append(scan.ranges[int(idx)])
            idx+=increment
        return samples

    def fill_scan_info(self,scan):
        self.scan_info = scan

    def padding(self, scan:LaserScan):
        res = LaserScan()
        res.header = scan.header
        res.angle_min = scan.angle_min
        res.angle_max = scan.angle_max
        res.angle_increment = scan.angle_increment
        res.time_increment = scan.time_increment
        res.scan_time = scan.scan_time
        res.range_min = scan.range_min
        res.range_max = scan.range_max
        res.ranges = scan.ranges + scan.ranges[:self.padding_size] 
        res.intensities = scan.intensities + scan.intensities[:self.padding_size]
        return res

    def max_filter(self, arr):
        return [min(x, self.max_range) for x in arr]

    def extended_max_filter(self, arr):
        # süreksizliği ortadan kaldır.
        rad45 = 0.7853981634
        rad135 = 2.3561944902
        rad225 = 3.926990817
        rad315 = 5.4977871438
        rad360 = 6.2831853072
        front_max_dist = 10
        side_max_dist = 3
        back_max_dist = 1
        
        scan = self.scan_info

        angle_padding = - scan.angle_min

        arr = []
        
        for i, r in enumerate(scan.ranges):
            # calculate angle
            angle = scan.angle_min + scan.angle_increment * i
            angle+=angle_padding

            if (rad45 < angle and angle < rad135) or (rad225 < angle and angle < rad315):
                arr.append(min(r, side_max_dist))
            elif rad135 < angle and angle < rad225:
                arr.append(min(r, back_max_dist))
            else:
                arr.append(min(r, front_max_dist))
        return arr
        
    def round_filter(self, arr):
        return [round(x, 1) for x in arr]

    def min_max_arr_normalize(self, val, min_val, max_val):
        normalized_data = [(x - min_val) / (max_val - min_val) for x in val]
        return normalized_data

    def normalize(self, val, min_val, max_val):
        normalized_data = (val - min_val) / (max_val - min_val)
        return normalized_data

    def get_states(self, scan:LaserScan):
        extended_scan = self.padding(scan)
        self.fill_scan_info(extended_scan)
        samples = self.downsample(extended_scan)
        # samples = self.extended_max_filter(samples)
        # samples = self.max_filter(samples)
        samples = self.min_max_arr_normalize(samples, 0.0, self.max_range)
        # samples = self.round_filter(samples)
        return samples


class RobotPreProcessing():
    def __init__(self, look_ahead_dist, max_dist):
        self.look_ahead_dist = look_ahead_dist
        self.max_dist = max_dist

    def round_filter(self, data):
        return round(data, 1)

    def closed_path_pose_info(self, path:Path, robot_pose:Pose):
        """
        find closest path pose wrt to robot
        Note: path and robot_pose should be the same frame
        """

        closed_pose = Pose()
        min_dist = float('inf')
        closed_point_idx = 0
        for i, pose in enumerate(path.poses):
            dist = sqrt((pose.pose.position.x - robot_pose.position.x)**2 + (pose.pose.position.y - robot_pose.position.y)**2)
            if dist < min_dist:
                min_dist = dist
                closed_point_idx = i
                closed_pose = pose.pose

        return closed_point_idx, closed_pose, min_dist

    def look_ahead_point(self, path:Path, closed_point_idx):
        """
        find look ahead point
        """
        dist = 0
        lap = Point()
        idx = closed_point_idx

        while(dist<= self.look_ahead_dist and idx< len(path.poses)):
            dist = sqrt((path.poses[closed_point_idx].pose.position.x - path.poses[idx].pose.position.x)**2 + (path.poses[closed_point_idx].pose.position.y - path.poses[idx].pose.position.y)**2)
            if (dist<=self.look_ahead_dist):
                lap = path.poses[idx].pose.position
            idx+=1
        
        return lap, dist
        
    def theta_wrt_look_ahead_point(self, look_ahead_point:Point, robot_pose:Pose):
        """
        find theta
        """
        # create coordinate
        look_ahead_theta =  atan2(look_ahead_point.y - robot_pose.position.y, look_ahead_point.x - robot_pose.position.x) # range: [-pi, pi]

        (_,_,robot_theta) = euler_from_quaternion([robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w]) 

        # get (x,y,theta) w.r.t. this coordinate system
        return robot_theta - look_ahead_theta

    def min_max_normalize(self, val, min_val, max_val):
        normalized_data = (val - min_val) / (max_val - min_val)
        return normalized_data

    def get_states(self, path:Path, robot_pose:Pose):
        closed_point_idx, _, min_dist = self.closed_path_pose_info(path, robot_pose)
        lap, dist_diff = self.look_ahead_point(path, closed_point_idx)
        theta = self.theta_wrt_look_ahead_point(lap, robot_pose)
        # theta = self.round_filter(theta)

        # norm_dist = self.min_max_normalize(self.round_filter(min_dist),0, self.max_dist)
        # norm_theta = self.min_max_normalize(self.round_filter(theta), -math.pi, math.pi)
        # dist_diff = self.min_max_normalize(self.round_filter(dist_diff), 0.0, self.look_ahead_dist)
        return [min_dist, theta, dist_diff]