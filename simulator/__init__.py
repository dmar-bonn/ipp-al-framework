from typing import Dict

import numpy as np


class Simulator:
    def __init__(self, cfg: Dict):
        self.simulator_name = "base-simulator"

        self.gsd = cfg["gsd"]  # m/pixel
        self.world_range = cfg["world_range"]  # pixel

        self.sensor_resolution = cfg["sensor"]["resolution"]
        self.sensor_angle = cfg["sensor"]["angle"]

    def get_measurement(self, pose: np.array, is_train_data: bool, mission_id: int) -> Dict:
        raise NotImplementedError(f"Simulator '{self.simulator_name}' does not implement 'get_measurement()' function!")

    def start_mission(self, init_pose: np.array):
        pass

    def move_to_next_waypoint(self, pose: np.array):
        pass
