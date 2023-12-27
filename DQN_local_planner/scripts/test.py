#!/usr/bin/env python3

import rospy

from nav_msgs.msg import Odometry, Path

from preprocessing import RobotPreProcessing
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse
from math import pi

robot_preprocessing = RobotPreProcessing(1.5, 3)

odom = Odometry()


def get_global_path():
    """
    return nav_msgs/Path
    """
    rospy.wait_for_service('/move_base/make_plan')
    try:
        get_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        req = GetPlanRequest()

        start = PoseStamped()
        start.header.frame_id = "map"
        start.pose.position = Point(0.0, -11.0, 0.0)
        start.pose.orientation.w = 1.0

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position = Point(4.0, -11.0, 0.0)
        goal_msg.pose.orientation.w = 1.0

        req.start = start
        req.goal = goal_msg
        res = get_plan(req)

        return res.plan
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)
        raise(e)



def odom_cb(msg):
    global_plan = get_global_path()
    min_dist, theta = robot_preprocessing.get_states(global_plan, msg.pose.pose)
    print(f"min dist:{min_dist} theta:{theta} angle: {theta*180/pi}")

def main():
    rospy.init_node('test', anonymous=True)
    rospy.sleep(1)

    rospy.Subscriber("odom", Odometry, odom_cb)

    rospy.spin()


if __name__ == '__main__':
    main()