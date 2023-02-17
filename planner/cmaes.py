from typing import Dict, List, Optional, Tuple

import cma
import numpy as np

from mapper.terrain_mapper import TerrainMapper
from planner.common import compute_flight_time, Planner


class CMAESPlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        budget: float,
        max_iter: int,
        population_size: int,
        sigma0: List,
        horizon_length: int,
        lattice_step_size: float,
        objective_fn_name: str,
    ):
        super(CMAESPlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "cmaes"
        self.mission_id = 0
        self.sigma0 = sigma0
        self.max_iter = max_iter
        self.population_size = population_size
        self.horizon_length = horizon_length
        self.lattice_step_size = lattice_step_size
        self.budget = budget
        self.remaining_budget = budget

    @staticmethod
    def stacked_poses(poses: List) -> List:
        stacked_poses = []
        for i in range(len(poses) // 3):
            stacked_poses.append(np.array([poses[3 * i], poses[3 * i + 1], poses[3 * i + 2]]))

        return stacked_poses

    @staticmethod
    def flatten_poses(poses: List):
        flattened_poses = []
        for poses in poses:
            flattened_poses.extend([poses[0], poses[1], poses[2]])

        return flattened_poses

    def objective_function(self, flattened_poses: List) -> float:
        poses = self.stacked_poses(flattened_poses)
        total_score = 0
        total_hit_count = 0
        total_budget_used = 0
        simulated_hit_map = np.zeros(self.mapper.train_data_map.count_map.shape)

        for i, pose in enumerate(poses):
            if i > 0:
                total_budget_used += compute_flight_time(pose, poses[i - 1], self.uav_specifications)

            (
                uncertainty_submap,
                representation_submap,
                hit_submap,
                train_data_count_submap,
                fov_indices,
            ) = self.mapper.get_map_state(pose)
            lu_x, lu_y, rd_x, rd_y = fov_indices
            simulated_unknown_space_mask = (simulated_hit_map[lu_y:rd_y, lu_x:rd_x] == 0) & (hit_submap == 0)
            uncertainty_submap[simulated_unknown_space_mask] = self.mapper.uncertainty_prior_const
            representation_submap[simulated_unknown_space_mask] = self.mapper.representation_prior_const
            score_submap = self.get_schematic_image(uncertainty_submap, representation_submap)

            simulated_hit_map[lu_y:rd_y, lu_x:rd_x] += 1
            total_score += np.sum(score_submap)
            total_hit_count += np.sum(train_data_count_submap)

        if total_budget_used == 0 or total_budget_used > self.remaining_budget:
            return 0

        total_hit_count += np.sum(simulated_hit_map)
        return -(total_score / (total_hit_count + 1)) / ((total_budget_used + 1) / self.budget)

    def calculate_parameter_bounds_and_scales(self, num_waypoints: int) -> Tuple[List, List, List]:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        upper_x = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1] - 1
        upper_y = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0] - 1
        lower_x, lower_y = boundary_space[1] + 1, boundary_space[0] + 1
        lower_z, upper_z = self.altitude - 1.0, self.altitude + 1.0

        lower_bounds = []
        upper_bounds = []
        sigma_scales = []

        for i in range(num_waypoints):
            lower_bounds.extend([lower_y, lower_x, lower_z])
            upper_bounds.extend([upper_y, upper_x, upper_z])
            sigma_scales.extend(self.sigma0)

        return lower_bounds, upper_bounds, sigma_scales

    def cma_es_optimization(self, init_waypoints: np.array) -> List:
        lower_bounds, upper_bounds, sigma_scales = self.calculate_parameter_bounds_and_scales(self.horizon_length)
        cma_es = cma.CMAEvolutionStrategy(
            self.flatten_poses(init_waypoints),
            sigma0=1,
            inopts={
                "bounds": [lower_bounds, upper_bounds],
                "maxiter": self.max_iter,
                "popsize": self.population_size,
                "CMA_stds": sigma_scales,
                "verbose": -9,
            },
        )
        cma_es.optimize(self.objective_function)

        return self.stacked_poses(list(cma_es.result.xbest))

    def create_planning_lattice(self) -> np.array:
        boundary_space = self.altitude * np.tan(np.deg2rad(self.sensor_angle))
        upper_x = self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] - boundary_space[1] - 1
        upper_y = self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] - boundary_space[0] - 1
        lower_x, lower_y = boundary_space[1] + 1, boundary_space[0] + 1

        lattice_steps_y = int(self.mapper.map_boundary[1] * self.mapper.ground_resolution[1] / self.lattice_step_size)
        lattice_steps_x = int(self.mapper.map_boundary[0] * self.mapper.ground_resolution[0] / self.lattice_step_size)
        y_candidates = np.linspace(lower_y, upper_y, lattice_steps_y)
        x_candidates = np.linspace(lower_x, upper_x, lattice_steps_x)
        pose_candidates = np.array(np.meshgrid(y_candidates, x_candidates)).T.reshape(-1, 2)

        return np.hstack((pose_candidates, self.altitude * np.ones((len(pose_candidates), 1))))

    def greedy_optimization(self, previous_pose: np.array, budget: float) -> List:
        pose_candidates = self.create_planning_lattice()
        init_poses = []
        simulated_hit_map = np.zeros(self.mapper.train_data_map.count_map.shape)

        for i in range(self.horizon_length):
            best_pose = previous_pose
            lu_x_best, lu_y_best, rd_x_best, rd_y_best = 0, 0, 0, 0
            best_pose_value = -np.inf
            for pose_candidate in pose_candidates:
                if np.linalg.norm(pose_candidate - previous_pose, ord=2) < self.lattice_step_size:
                    continue

                pose_costs = compute_flight_time(pose_candidate, previous_pose, self.uav_specifications)
                if pose_costs > budget:
                    continue

                (
                    uncertainty_submap,
                    representation_submap,
                    hit_submap,
                    train_data_count_submap,
                    fov_indices,
                ) = self.mapper.get_map_state(pose_candidate)
                lu_x, lu_y, rd_x, rd_y = fov_indices
                simulated_unknown_space_mask = (simulated_hit_map[lu_y:rd_y, lu_x:rd_x] == 0) & (hit_submap == 0)
                uncertainty_submap[simulated_unknown_space_mask] = self.mapper.uncertainty_prior_const
                representation_submap[simulated_unknown_space_mask] = self.mapper.representation_prior_const
                score_submap = self.get_schematic_image(uncertainty_submap, representation_submap)

                simulated_hit_submap_count = np.sum(simulated_hit_map[lu_y:rd_y, lu_x:rd_x]) + np.sum(
                    train_data_count_submap
                )
                pose_value = (np.sum(score_submap) / (simulated_hit_submap_count + 1)) / (
                    (pose_costs + 1) / self.budget
                )

                if pose_value > best_pose_value:
                    best_pose = pose_candidate
                    best_pose_value = pose_value
                    lu_x_best, lu_y_best, rd_x_best, rd_y_best = lu_x, lu_y, rd_x, rd_y

            budget -= compute_flight_time(best_pose, previous_pose, self.uav_specifications)
            init_poses.append(best_pose)
            previous_pose = best_pose
            simulated_hit_map[lu_y_best:rd_y_best, lu_x_best:rd_x_best] += 1

        return init_poses

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        self.mission_id = kwargs["mission_id"]
        self.remaining_budget = budget
        greedy_poses = self.greedy_optimization(previous_pose, budget)
        fine_tuned_poses = self.cma_es_optimization(greedy_poses)

        greedy_value = -self.objective_function(self.flatten_poses(greedy_poses))
        fine_tuned_value = -self.objective_function(self.flatten_poses(fine_tuned_poses))
        if greedy_value > fine_tuned_value:
            fine_tuned_poses = greedy_poses

        next_best_pose = fine_tuned_poses[0]
        next_best_pose[2] = self.altitude

        if compute_flight_time(next_best_pose, previous_pose, self.uav_specifications) > budget:
            return None

        return next_best_pose
