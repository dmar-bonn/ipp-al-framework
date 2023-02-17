import numpy as np
import pyproj
import rospy
from bonn_msgs.msg import Battery, FlightMode, Heading, Position, Waypoint
from cv_bridge import CvBridge


class PoseWGS84:
    def __init__(self, longitude=None, latitude=None, height_enu=None, heading=None):
        self.longitude = longitude
        self.latitude = latitude
        self.height_enu = height_enu
        self.heading = heading

    def update_position(self, longitude, latitude, height_enu):
        self.longitude, self.latitude, self.height_enu = longitude, latitude, height_enu

    def update_heading(self, heading):
        self.heading = heading


class PoseUTM:
    def __init__(self, x=None, y=None, height_enu=None, heading=None):
        self.x = x
        self.y = y
        self.height_enu = height_enu
        self.heading = heading

    def update_position(self, x, y, height_enu):
        self.x, self.y, self.height_enu = x, y, height_enu

    def update_heading(self, heading):
        self.heading = heading


class BatteryState:
    def __init__(self, voltage=None, percentage=None):
        self.voltage = voltage
        self.percentage = percentage

    def update(self, voltage, percentage):
        self.voltage = voltage
        self.percentage = percentage

    @property
    def is_low(self):
        return self.percentage <= 25.0


class BonnDroneState:

    TO_FLIGHT_MODE = {
        "NOT_INITIALIZED": 0,
        "ATTITUDE_MANUAL": 1,
        "POSITION_MANUAL": 2,
        "POSITION_EXTERNAL": 3,
    }

    FROM_FLIGHT_MODE = {
        0: "NOT_INITIALIZED",
        1: "ATTITUDE_MANUAL",
        2: "POSITION_MANUAL",
        3: "POSITION_EXTERNAL",
    }

    def __init__(self):
        self.initial_ros_timestamp = None

        self.position_initialized = False
        self.heading_initialized = False

        self.position_fix = False
        self.mission_started = False

        self.initial_pose_wgs84 = PoseWGS84()
        self.initial_pose_utm = PoseUTM()

        self.current_pose_wgs84 = PoseWGS84()
        self.current_pose_utm = PoseUTM()

        self.flight_mode = 0
        self.battery_state = BatteryState()

        self.wgs84_to_utm_proj = pyproj.Proj(proj="utm", zone=32, ellps="WGS84")

    def set_initial_timestamp(self, msg):
        self.initial_ros_timestamp = rospy.Time.now() - msg.header.stamp
        print("INITIAL ROS TIME: {}s, {}ns".format(self.initial_ros_timestamp.secs, self.initial_ros_timestamp.nsecs))

    def update_battery_state(self, battery_msg):
        self.battery_state.update(battery_msg.voltage, battery_msg.percentage)

    def update_flight_mode(self, flight_mode_msg):
        if self.flight_mode != flight_mode_msg.flight_mode:
            print("FLIGHT MODE CHANGED TO '{}'.".format(self.FROM_FLIGHT_MODE[flight_mode_msg.flight_mode]))

        self.flight_mode = flight_mode_msg.flight_mode

    def update_position(self, position_msg):
        if not self.position_initialized:
            self.set_initial_position(position_msg)
            self.set_initial_timestamp(position_msg)

        self.position_fix = True if position_msg.position_fix == 1 else False
        if self.mission_started and not self.position_fix:
            print("LOST GPS SIGNAL. POSITION FIX: {}.".format(self.position_fix))

        self.current_pose_wgs84.update_position(position_msg.longitude, position_msg.latitude, position_msg.height_ENU)
        self.current_pose_utm.update_position(*self.wgs84_to_utm_pose(self.current_pose_wgs84))

    def update_heading(self, heading_msg):
        if not self.heading_initialized:
            self.set_initial_heading(heading_msg)

    def set_initial_position(self, position_msg):
        print("SET INITIAL POSITION.")
        self.position_initialized = True
        self.initial_pose_wgs84.update_position(position_msg.longitude, position_msg.latitude, position_msg.height_ENU)
        self.initial_pose_utm.update_position(*self.wgs84_to_utm_pose(self.initial_pose_wgs84))

    def set_initial_heading(self, heading_msg):
        print("SET INITIAL HEADING.")
        self.heading_initialized = True
        self.initial_pose_wgs84.update_heading(heading_msg.heading)
        self.initial_pose_utm.update_heading(heading_msg.heading)

    @property
    def position(self):
        x = self.initial_pose_utm.x - self.current_pose_utm.x
        y = self.initial_pose_utm.y - self.current_pose_utm.y
        return np.array([x, y, self.current_pose_utm.height_enu])

    @property
    def heading(self):
        return self.current_pose_utm.heading

    def wgs84_to_utm_pose(self, pose_wgs84):
        x, y = self.wgs84_to_utm_proj(pose_wgs84.latitude, pose_wgs84.longitude)
        return x, y, pose_wgs84.height_enu

    def utm_to_wgs84_pose(self, pose_utm):
        latitude, longitude = self.wgs84_to_utm_proj(pose_utm.x, pose_utm.y, inverse=True)
        return latitude, longitude, pose_utm.height_enu

    @property
    def autonomous_mode_activated(self):
        return (
            self.flight_mode == self.TO_FLIGHT_MODE["POSITION_EXTERNAL"]
            and self.position_initialized
            and self.heading_initialized
            and self.position_fix
            and not self.battery_state.is_low
        )


class BonnDroneBridge:
    def __init__(self):
        self.state = BonnDroneState()

        self.cv_bridge = CvBridge()
        self.waypoint_pub = rospy.Publisher("/bonn_mavlink/jp_waypoint", Waypoint, queue_size=1)
        self.position_sub = rospy.Subscriber(
            "/bonn_mavlink/pj_position", Position, self.position_callback, queue_size=1
        )
        self.heading_sub = rospy.Subscriber("/bonn_mavlink/pj_heading", Heading, self.heading_callback, queue_size=1)
        self.flight_mode_sub = rospy.Subscriber(
            "/bonn_mavlink/pj_flight_mode", FlightMode, self.flight_mode_callback, queue_size=1
        )
        self.battery_sub = rospy.Subscriber("/bonn_mavlink/pj_battery", Battery, self.battery_callback, queue_size=1)

    def flight_mode_callback(self, flight_mode_msg):
        self.state.update_flight_mode(flight_mode_msg)

    def position_callback(self, pose_msg):
        self.state.update_position(pose_msg)

    def heading_callback(self, heading_msg):
        self.state.update_heading(heading_msg)

    def battery_callback(self, battery_msg):
        self.state.update_battery_state(battery_msg)

    def build_waypoint_msg(self, waypoint):
        waypoint_utm_x = self.state.initial_pose_utm.x + waypoint[0]
        waypoint_utm_y = self.state.initial_pose_utm.y + waypoint[1]
        latitude, longitude, _ = self.state.utm_to_wgs84_pose(PoseUTM(waypoint_utm_x, waypoint_utm_y, waypoint[2]))

        waypoint_msg = Waypoint()
        waypoint_msg.header.stamp = rospy.Time.now() - self.state.initial_ros_timestamp
        waypoint_msg.longitude = longitude
        waypoint_msg.latitude = latitude
        waypoint_msg.height_ENU = waypoint[2]
        waypoint_msg.heading = waypoint[3]

        return waypoint_msg

    def position_reached(self, goal_position):
        return np.all(np.abs(goal_position - self.state.position) > np.array([0.1, 0.1, 0.5]))

    def heading_reached(self, goal_heading):
        return ((goal_heading - self.state.heading + 180) % 360 - 180) < 2.0

    def wait_util_goal_pose_reached(self, goal_pose):
        while not self.position_reached(goal_pose[:3]) or not self.heading_reached(goal_pose[3]):
            rospy.sleep(1)

    def start_mission(self):
        self.state.mission_started = True

    @staticmethod
    def waypoint_within_geofence(position):
        radius = np.linalg.norm(position, ord=2)
        return radius < 500 and position[2] > 1

    def move_to_next_waypoint(self, goal_pose):
        if not self.state.mission_started:
            print("START AUTONOMOUS MISSION FIRST BEFORE SENDING WAYPOINTS.")
            return

        if not self.state.autonomous_mode_activated:
            print("CANNOT SEND WAYPOINTS. AUTONOMOUS MODE DEACTIVATED.")
            print(
                "FLIGHT MODE: {}, POSITION FIX: {}, POSITION/HEADING INITIALIZED: {}/{}, BATTERY LEVEL: {}%".format(
                    self.state.FROM_FLIGHT_MODE[self.state.flight_mode],
                    self.state.position_fix,
                    self.state.position_initialized,
                    self.state.heading_initialized,
                    round(self.state.battery_state.percentage, 2),
                )
            )

        if goal_pose.shape[0] != 4:
            print("WAYPOINT HAS LENGTH {}. EXPECTED TO HAVE LENGTH 4 (x, y, z, heading).".format(goal_pose.shape[0]))
            return

        if not self.waypoint_within_geofence(goal_pose[:3]):
            print("WAYPOINT {} NOT WITHIN GEOFENCE.".format(goal_pose[:3]))
            return

        if np.abs(goal_pose[3]) > 180:
            goal_pose[3] = np.clip(goal_pose[3], -180, 180)
            print("HEADING NOT IN BETWEEN -180° and 180°. HEADING CLIPPED TO {}°.".format(goal_pose[4]))

        self.waypoint_pub.publish(self.build_waypoint_msg(goal_pose))
        self.wait_util_goal_pose_reached(goal_pose)
