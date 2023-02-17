from typing import Optional, Dict

import numpy as np

from mapper.terrain_mapper import TerrainMapper
from planner.common import compute_flight_time, Planner


class ImagePlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        step_size: float,
        edge_width: float,
        objective_fn_name: str,
    ):
        super(ImagePlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "image-based"
        self.step_size = step_size
        self.edge_width = edge_width

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        schematic_image = self.get_schematic_image(kwargs["uncertainty_image"], kwargs["representation_image"])

        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        # Sum uncertainty or representation score values on each image edge
        x_left_sum = np.sum(schematic_image[:, 0 : self.edge_width])
        x_right_sum = np.sum(schematic_image[:, -1 - self.edge_width : -1])
        y_bottom_sum = np.sum(schematic_image[0 : self.edge_width, :])
        y_top_sum = np.sum(schematic_image[-1 - self.edge_width : -1, :])
        schematic_values = np.array([x_left_sum, x_right_sum, y_bottom_sum, y_top_sum])

        # Sum train data count values on each image edge
        _, _, _, train_data_count_submap, _ = self.mapper.get_map_state(previous_pose)
        hit_x_left_sum = np.sum(train_data_count_submap[:, 0 : self.edge_width])
        hit_x_right_sum = np.sum(train_data_count_submap[:, -1 - self.edge_width : -1])
        hit_y_bottom_sum = np.sum(train_data_count_submap[0 : self.edge_width, :])
        hit_y_top_sum = np.sum(train_data_count_submap[-1 - self.edge_width : -1, :])
        hit_values = np.array([hit_x_left_sum, hit_x_right_sum, hit_y_bottom_sum, hit_y_top_sum])

        ind_array = np.argsort(schematic_values / hit_values)
        ind_array = np.flip(ind_array)

        i = 0
        new_pose = previous_pose

        while (previous_pose == new_pose).all():

            ind_max = ind_array[i]

            # Move left (x)
            if ind_max == 0:
                new_pose = previous_pose + [-self.step_size, 0, 0]
            # Move right (x)
            elif ind_max == 1:
                new_pose = previous_pose + [self.step_size, 0, 0]
            # Move down (y)
            elif ind_max == 2:
                new_pose = previous_pose + [0, -self.step_size, 0]
            # Move up (y)
            elif ind_max == 3:
                new_pose = previous_pose + [0, self.step_size, 0]

            new_pose[0] = np.clip(new_pose[0], boundary_space[0], max_x)
            new_pose[1] = np.clip(new_pose[1], boundary_space[1], max_y)

            i = i + 1

        if compute_flight_time(new_pose, previous_pose, self.uav_specifications) <= budget:
            return new_pose
        else:
            return None
