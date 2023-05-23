import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point, Twist, Pose, PoseStamped
from actionlib_msgs.msg import GoalStatusArray, GoalStatus
from math import sqrt

class Utils:
    def __init__(self):
        
        self._global_plan = Path()
        self._goal = PoseStamped()

        rospy.Subscriber("odom", Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber("move_base/NavfnROS/plan", Path, self.path_cb, queue_size=10)
        rospy.Subscriber("move_base/status", GoalStatusArray, self.status_cb, queue_size=10)

        self._total_min_dist = 0
        self._prev_odom = Odometry()

        self._total_execution_time = 0
        
        self._prev_time = rospy.get_time()
        self._move_base_status = GoalStatus()

        self._over_path_dist = 0.2
        self.closed_path_pose_to_obstacle = Pose()
        self.is_on_path = True

    def odom_cb(self, msg:Odometry):
        if self._move_base_status == GoalStatus().ACTIVE:
            #find total dist differences
            min_dist = float('inf')
            tmp_pose = Pose()
            for i, var in enumerate(self._global_plan.poses):
                diff = sqrt((var.pose.position.x - msg.pose.pose.position.x)**2 + (var.pose.position.y - msg.pose.pose.position.y)**2 )
                if diff< min_dist:
                    min_dist = diff
            
            odom_diff = sqrt((self._prev_odom.pose.pose.position.x - msg.pose.pose.position.x)**2 + (self._prev_odom.pose.pose.position.y - msg.pose.pose.position.y)**2 )        
            if min_dist !=float('inf'):
                self._total_min_dist+= min_dist * odom_diff
                tmp_pose = var

        self._prev_odom = msg

        # find distance between obstacle and path point where robot connect to path
    
    def path_cb(self, msg):
        self._global_plan = msg

    def status_cb(self, msg:GoalStatusArray):

        if(len(msg.status_list)!=0):
            self._move_base_status = msg.status_list[0].status
            
            if (msg.status_list[0].status == GoalStatus().ACTIVE):
                self._total_execution_time = rospy.get_time() - self._prev_time            
            else:
                self._prev_time = rospy.get_time()
        else:
            self._move_base_status = GoalStatus()
            self._prev_time = rospy.get_time()

    def print(self):

        print("Total min dist: {} Total execution time: {}".format(self._total_min_dist, self._total_execution_time))

if __name__ == '__main__':
    try:
        rospy.init_node('dqn_utils', anonymous=True)
        rate = rospy.Rate(1)
        u = Utils()
        while not rospy.is_shutdown():
            u.print()
            rate.sleep()
        pass
    except rospy.ROSInterruptException:
        pass