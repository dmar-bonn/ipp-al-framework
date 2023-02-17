from typing import Dict

import cv2
import numpy as np
import rospy

from simulator import Simulator
from simulator.flightmare_bridges import FlightMareBridge
from utils import utils


class FlightmareSimulator(Simulator):
    def __init__(self, cfg: Dict):
        super(FlightmareSimulator, self).__init__(cfg)

        self.simulator_name = "flightmare-simulator"
        self.flightmare_bridge = FlightMareBridge()

    def get_measurement(self, map_pose: np.array, is_train_data: bool, mission_id: int) -> Dict:
        fov_info = utils.get_fov(map_pose, self.sensor_angle, self.gsd, self.world_range)

        fov_corner, range_list = fov_info
        gsd = [
            (np.linalg.norm(fov_corner[1] - fov_corner[0])) / self.sensor_resolution[0],
            (np.linalg.norm(fov_corner[3] - fov_corner[0])) / self.sensor_resolution[1],
        ]
        rgb_image_raw, rgb_anno_raw = self.flightmare_bridge.get_image_and_segmentation()
        rgb_image = cv2.resize(rgb_image_raw, tuple(self.sensor_resolution))
        rgb_anno = cv2.resize(rgb_anno_raw, tuple(self.sensor_resolution))

        return {
            "image": rgb_image,
            "anno": rgb_anno,
            "fov": fov_corner,
            "gsd": gsd,
            "is_train_data": is_train_data,
            "mission_id": mission_id,
            "pose": map_pose,
        }

    def start_mission(self, init_pose: np.array):
        self.flightmare_bridge.arm()
        self.flightmare_bridge.start()
        rospy.sleep(5)

        self.flightmare_bridge.move_to_next_waypoint(init_pose)

    @staticmethod
    def transform_to_world_coordinates(map_pose: np.array) -> np.array:
        coord_system_orientation = np.array([1, -1, 1])
        origin_pose = np.array([-50, 50, 0])
        return origin_pose + coord_system_orientation * map_pose

    def move_to_next_waypoint(self, map_pose: np.array):
        self.flightmare_bridge.move_to_next_waypoint(self.transform_to_world_coordinates(map_pose))
