from typing import Dict, List, Optional

import numpy as np

from planner.common import compute_flight_time


class ActionSpace:
    def __init__(self, altitude: float, uav_specifications: Dict, world_size: np.array, sensor_angle: np.array):
        self.action_space_name = "Base"
        self.boundary_space = altitude * np.tan(np.deg2rad(sensor_angle))
        self.world_size = world_size
        self.altitude = altitude
        self.uav_specifications = uav_specifications

    def sample(self, pose: np.array = None, budget: float = None):
        raise NotImplementedError(f"Action space '{self.action_space_name}' does not implement the 'sample' function!")


class DiscreteActionSpace(ActionSpace):
    def __init__(
        self, step_sizes: List, altitude: float, uav_specifications: Dict, world_size: np.array, sensor_angle: np.array
    ):
        super(DiscreteActionSpace, self).__init__(altitude, uav_specifications, world_size, sensor_angle)

        self.action_space_name = "Discrete"
        self.step_sizes = step_sizes
        self.n = len(self.step_sizes) * 4
        self.actions = self.create_actions()

    @property
    def directions(self) -> List:
        return [
            np.array([1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([0, -1, 0]),
        ]

    def create_actions(self) -> Dict:
        actions = {}
        for action_id in range(self.n):
            step_size_id = int(action_id / 4)
            direction_id = action_id - step_size_id * 4
            actions[action_id] = self.directions[direction_id] * np.array(
                [self.step_sizes[step_size_id], self.step_sizes[step_size_id], self.altitude]
            )

        return actions

    def get_valid_actions(self, pose: np.array, budget: float) -> List:
        valid_action_ids = []
        for action_id, action in self.actions.items():
            next_pose = pose + action
            within_budget = budget > compute_flight_time(next_pose, pose, self.uav_specifications)
            within_map = np.all(next_pose[:2] >= self.boundary_space) and np.all(
                next_pose[:2] <= self.world_size - self.boundary_space
            )
            if within_budget and within_map:
                valid_action_ids.append(action_id)

        return valid_action_ids

    def sample(self, pose: np.array = None, budget: float = None) -> int:
        if pose is None or budget is None:
            return np.random.choice(self.n)

        valid_action_ids = np.array(self.get_valid_actions(pose, budget))
        return np.random.choice(valid_action_ids)


class ContinuousActionSpace(ActionSpace):
    def __init__(
        self,
        min_radius: float,
        max_radius: float,
        altitude: float,
        uav_specifications: Dict,
        world_size: np.array,
        sensor_angle: np.array,
    ):
        super(ContinuousActionSpace, self).__init__(altitude, uav_specifications, world_size, sensor_angle)

        self.min_radius = min_radius
        self.max_radius = max_radius

    def action_valid(self, pose: np.array, budget: float, action: np.array) -> bool:
        next_pose = pose + action
        within_budget = budget > compute_flight_time(next_pose, pose, self.uav_specifications)
        within_map = np.all(next_pose[:2] >= self.boundary_space) and np.all(
            next_pose[:2] <= self.world_size - self.boundary_space
        )

        return within_budget and within_map

    def generate_uniform_sample(self) -> np.array:
        sampled_radian = np.random.uniform(low=0, high=2 * np.pi)
        sampled_unit_direction = np.round(np.array([np.sin(sampled_radian), np.cos(sampled_radian)]), 5)
        sampled_direction = np.random.uniform(low=self.min_radius, high=self.max_radius) * sampled_unit_direction
        return np.array([sampled_direction[0], sampled_direction[1], 0])

    def sample(self, pose: np.array = None, budget: float = None) -> Optional[np.array]:
        if pose is None or budget is None:
            return self.generate_uniform_sample()

        for _ in range(1000):
            sampled_action = self.generate_uniform_sample()
            if self.action_valid(pose, budget, sampled_action):
                return sampled_action

        return None
