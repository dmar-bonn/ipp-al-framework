from typing import Tuple

import numpy as np
import quadrotor_msgs.msg as quadrotor_msgs
import rospy
import std_msgs.msg as std_msgs
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty


class FlightMareBridge:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.state = 0

        self.cv_bridge = CvBridge()
        self.waypoint_pub = rospy.Publisher("/hummingbird/autopilot/pose_command", PoseStamped, queue_size=1)
        self.pose_sub = rospy.Subscriber(
            "/hummingbird/autopilot/feedback", quadrotor_msgs.AutopilotFeedback, self.autopilot_callback
        )
        self.image_sub = rospy.Subscriber("/rgb", Image, self.image_callback)

    @property
    def pose(self) -> np.array:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def arm():
        arm_bridge_pub = rospy.Publisher("hummingbird/bridge/arm", std_msgs.Bool, queue_size=1)
        arm_message = std_msgs.Bool(True)

        rospy.sleep(2)
        arm_bridge_pub.publish(arm_message)

    @staticmethod
    def start():
        start_pub = rospy.Publisher("/hummingbird/autopilot/start", Empty, queue_size=1)
        start_message = Empty()

        rospy.sleep(2)
        start_pub.publish(start_message)

    def autopilot_callback(self, data):
        self.state = data.autopilot_state
        self.x = data.state_estimate.pose.pose.position.x
        self.y = data.state_estimate.pose.pose.position.y
        self.z = data.state_estimate.pose.pose.position.z

    def image_callback(self, data: Image):
        pass

    @staticmethod
    def build_waypoint_msg(waypoint: np.array) -> PoseStamped:
        waypoint_msg = PoseStamped()

        waypoint_msg.header.seq = 1
        waypoint_msg.header.stamp = rospy.Time.now()
        waypoint_msg.header.frame_id = ""

        waypoint_msg.pose.position.x = waypoint[0]
        waypoint_msg.pose.position.y = waypoint[1]
        waypoint_msg.pose.position.z = waypoint[2]

        waypoint_msg.pose.orientation.x = 0.0
        waypoint_msg.pose.orientation.y = 0.0
        waypoint_msg.pose.orientation.z = 0.0
        waypoint_msg.pose.orientation.w = 0.0

        return waypoint_msg

    def wait_util_goal_pose_reached(self, goal_pose: np.array):
        dist_to_goal = np.linalg.norm(goal_pose - self.pose, ord=2)
        while dist_to_goal > 0.1:
            dist_to_goal = np.linalg.norm(goal_pose - self.pose, ord=2)
            rospy.sleep(1)

    def wait_until_hover(self):
        while self.state != 2:
            rospy.sleep(1)

    def move_to_next_waypoint(self, goal_pose: np.array):
        self.waypoint_pub.publish(self.build_waypoint_msg(goal_pose))
        self.wait_util_goal_pose_reached(goal_pose)
        self.wait_until_hover()

    def get_image_and_segmentation(self) -> Tuple[np.array, np.array]:
        data_rgb = rospy.wait_for_message("/rgb", Image, timeout=None)
        data_seg = rospy.wait_for_message("/segmentation", Image, timeout=None)

        return self.cv_bridge.imgmsg_to_cv2(data_rgb), self.cv_bridge.imgmsg_to_cv2(data_seg)
