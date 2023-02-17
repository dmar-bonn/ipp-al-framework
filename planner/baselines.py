from typing import List, Optional, Dict

import numpy as np

from mapper.terrain_mapper import TerrainMapper
from planner.action_spaces import ContinuousActionSpace
from planner.common import compute_flight_time, Planner


class GlobalRandomPlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        objective_fn_name: str,
        min_radius: float = 30,
        max_radius: float = 70,
    ):
        super(GlobalRandomPlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "global-random"
        self.min_radius = min_radius
        self.max_radius = max_radius

        world_size = np.array(self.mapper.map_boundary) * np.array(self.mapper.ground_resolution)
        self.action_space = ContinuousActionSpace(
            min_radius, max_radius, altitude, uav_specifications, world_size, self.sensor_angle
        )

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        sampled_action = self.action_space.sample(previous_pose, budget)

        if sampled_action is None:
            return None

        return previous_pose + sampled_action


class LocalRandomPlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        step_size: float,
        objective_fn_name: str,
    ):
        super(LocalRandomPlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "local-random"
        self.step_size = step_size

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        next_pose = previous_pose
        local_move_indices = [0, 1, 2, 3]
        while (previous_pose == next_pose).all():
            local_move_id = np.random.choice(local_move_indices)

            # Move left (x)
            if local_move_id == 0:
                next_pose = previous_pose + [-self.step_size, 0, 0]
            # Move right (x)
            elif local_move_id == 1:
                next_pose = previous_pose + [self.step_size, 0, 0]
            # Move down (y)
            elif local_move_id == 2:
                next_pose = previous_pose + [0, -self.step_size, 0]
            # Move up (y)
            elif local_move_id == 3:
                next_pose = previous_pose + [0, self.step_size, 0]

            next_pose[0] = np.clip(next_pose[0], boundary_space[0], max_x)
            next_pose[1] = np.clip(next_pose[1], boundary_space[1], max_y)

            local_move_indices.remove(local_move_id)

        if compute_flight_time(next_pose, previous_pose, self.uav_specifications) <= budget:
            return next_pose
        else:
            return None


class CoveragePlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        step_sizes: List[float],
        objective_fn_name: str,
    ):
        super(CoveragePlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "coverage-based"
        self.step_sizes = step_sizes
        self.waypoints = []
        self.step_counter = 0
        self.step_size = step_sizes[0]

    def setup(self, **kwargs):
        mission_id = kwargs["mission_id"]
        self.step_size = self.step_sizes[mission_id % len(self.step_sizes)]
        self.waypoints = self.create_coverage_pattern(flip_orientation=(mission_id % 2))

    def create_coverage_pattern(self, flip_orientation: bool = False) -> np.array:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle)) + 1
        min_y, min_x = boundary_space[1], boundary_space[0]
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        x_positions = np.linspace(min_x, max_x, int((max_x - min_x) / self.step_size) + 1)
        y_positions = np.linspace(min_y, max_y, int((max_y - min_y) / self.step_size) + 1)
        waypoints = np.zeros((len(y_positions) * len(x_positions), 3))

        if flip_orientation:
            for j, x_pos in enumerate(x_positions):
                for k, y_pos in enumerate(y_positions):
                    if j % 2 == 1:
                        y_pos = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - y_pos

                    waypoints[j * len(y_positions) + k] = np.array([x_pos, y_pos, self.altitude], dtype=np.float32)
        else:
            for j, y_pos in enumerate(y_positions):
                for k, x_pos in enumerate(x_positions):
                    if j % 2 == 1:
                        x_pos = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - x_pos

                    waypoints[j * len(x_positions) + k] = np.array([x_pos, y_pos, self.altitude], dtype=np.float32)

        return waypoints

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        if self.step_counter >= len(self.waypoints):
            return None

        pose = self.waypoints[self.step_counter, :]
        self.step_counter += 1

        if compute_flight_time(pose, previous_pose, self.uav_specifications) > budget:
            return None

        return pose
