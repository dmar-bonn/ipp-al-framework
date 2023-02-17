from typing import Dict, Optional

import cv2
import numpy as np

from mapper.terrain_mapper import TerrainMapper
from planner.common import compute_flight_time, Planner


class FrontierPlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        frontier_step_size: float,
        objective_fn_name: str,
    ):
        super(FrontierPlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "frontier-based"
        self.frontier_step_size = frontier_step_size

    def objective_fn(self, candidate_pose: np.array) -> float:
        (
            uncertainty_submap,
            representation_submap,
            hit_submap,
            train_data_count_submap,
            fov_indices,
        ) = self.mapper.get_map_state(candidate_pose)

        uncertainty_submap[hit_submap == 0] = self.mapper.uncertainty_prior_const
        representation_submap[hit_submap == 0] = self.mapper.representation_prior_const
        score_submap = self.get_schematic_image(uncertainty_submap, representation_submap)

        return np.sum(score_submap) / (np.sum(train_data_count_submap) + 1)

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        max_y = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1]
        max_x = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0]

        hit_map_img = self.mapper.hit_map.count_map.astype(np.uint8)
        hit_map_img[hit_map_img > 1] = 1
        contours, hierarchy = cv2.findContours(hit_map_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        frontiers = contours[0][:, 0][:: self.frontier_step_size, :]

        best_pose = previous_pose
        best_frontier_value = -np.inf
        for frontier_candidate in frontiers:
            frontier_candidate = frontier_candidate * self.mapper.ground_resolution
            frontier_candidate = np.append(frontier_candidate, self.altitude)

            frontier_costs = compute_flight_time(frontier_candidate, previous_pose, self.uav_specifications)
            if frontier_costs > budget:
                continue

            frontier_candidate[0] = np.clip(frontier_candidate[0], boundary_space[0], max_x)
            frontier_candidate[1] = np.clip(frontier_candidate[1], boundary_space[1], max_y)

            if np.allclose(previous_pose, frontier_candidate):
                continue

            if compute_flight_time(previous_pose, frontier_candidate, self.uav_specifications) > budget:
                continue

            frontier_candidate_value = self.objective_fn(frontier_candidate)

            if frontier_candidate_value > best_frontier_value:
                best_frontier_value = frontier_candidate_value
                best_pose = frontier_candidate

        if np.allclose(previous_pose, best_pose):
            return None

        if compute_flight_time(best_pose, previous_pose, self.uav_specifications) > budget:
            return None

        return best_pose
