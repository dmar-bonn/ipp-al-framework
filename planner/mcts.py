import random
from typing import Dict, List, Optional

import numpy as np

from mapper.terrain_mapper import TerrainMapper
from planner.action_spaces import DiscreteActionSpace
from planner.common import compute_flight_time, Planner
from utils import utils


class State:
    def __init__(self, remaining_budget: float, train_data_count_map: np.array, pose: np.array):
        self.pose = pose
        self.remaining_budget = remaining_budget
        self.train_data_count_map = train_data_count_map


class Node:
    def __init__(self, state: State, parent=None):
        self.parent = parent
        self.state = state
        self.value = 0
        self.visits = 0
        self.children = {}

    @staticmethod
    def uct_child(child_node, min_val: float, max_val: float, c: float = 2.0):
        if child_node.visits == 0:
            return np.inf

        exploration = c * np.sqrt(np.log(child_node.parent.visits) / child_node.visits)
        if max_val == 0:
            return child_node.value + exploration

        if max_val == min_val:
            normalized_value = (child_node.value - min_val) / max_val
        else:
            normalized_value = (child_node.value - min_val) / (max_val - min_val)

        return normalized_value + exploration

    def select_utc_child(self, c: float = 2.0):
        max_children = []
        max_action_ids = []
        max_uct = -np.inf

        children_vals = [child.value for child in self.children.values()]
        min_child_val, max_child_val = min(children_vals), max(children_vals)
        for action_id in self.children.keys():
            uct_child = self.uct_child(self.children[action_id], min_child_val, max_child_val, c=c)
            if max_uct == uct_child:
                max_children.append(self.children[action_id])
                max_action_ids.append(action_id)
            if uct_child > max_uct:
                max_uct = uct_child
                max_children = [self.children[action_id]]
                max_action_ids = [action_id]

        sampled_node_id = np.random.choice(len(max_children))
        return max_children[sampled_node_id], max_action_ids[sampled_node_id]

    def select_best_child(self):
        max_children = []
        max_value = -np.inf

        for action_id, child in self.children.items():
            if child.value > max_value:
                max_value = child.value
                max_children = [child]
            if child.value == max_value:
                max_children.append(child)

        return random.choice(max_children)


class MCTSPlanner(Planner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        budget: float,
        objective_fn_name: str,
        num_simulations: int = 100,
        gamma: float = 0.99,
        horizon_length: int = 5,
        exploration_constant: float = 2.0,
    ):
        super(MCTSPlanner, self).__init__(mapper, altitude, sensor_info, uav_specifications, objective_fn_name)

        self.planner_name = "mcts"
        self.mission_id = 0
        self.budget = budget
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.exploration_constant = exploration_constant
        self.horizon_length = horizon_length
        self.world_size = np.array(self.mapper.map_boundary) * np.array(self.mapper.ground_resolution)

    def compute_reward(self, state: State, next_state: State):
        (
            uncertainty_submap,
            representation_submap,
            hit_submap,
            _,
            fov_indices,
        ) = self.mapper.get_map_state(next_state.pose)
        lu_x, lu_y, rd_x, rd_y = fov_indices
        train_data_count_submap = state.train_data_count_map[lu_y:rd_y, lu_x:rd_x]

        unknown_space_mask = (train_data_count_submap == 0) & (hit_submap == 0)
        uncertainty_submap[unknown_space_mask] = self.mapper.uncertainty_prior_const
        representation_submap[unknown_space_mask] = self.mapper.representation_prior_const

        score_submap = self.get_schematic_image(uncertainty_submap, representation_submap)
        action_cost = compute_flight_time(next_state.pose, state.pose, self.uav_specifications)
        return np.sum(score_submap) / (np.sum(train_data_count_submap) + 1) / ((action_cost + 1) / self.budget)

    def prediction_step(self, state: State, action: np.array) -> State:
        next_pose = state.pose + action
        remaining_budget = state.remaining_budget - compute_flight_time(state.pose, next_pose, self.uav_specifications)

        fov_corners, _ = utils.get_fov(next_pose, self.sensor_angle, self.mapper.gsd, self.mapper.world_range)
        lu, _, rd, _ = fov_corners
        lu_x, lu_y = self.mapper.find_map_index(lu)
        rd_x, rd_y = self.mapper.find_map_index(rd)
        next_train_data_count_map = state.train_data_count_map
        next_train_data_count_map[lu_y:rd_y, lu_x:rd_x] += 1

        return State(remaining_budget, next_train_data_count_map, next_pose)

    def rollout(self, state: State, depth: int) -> float:
        raise NotImplementedError(f"MCTS planner '{self.planner_name}' does not implement the 'rollout' function!")

    def simulate(self, node: Node, depth: int) -> float:
        raise NotImplementedError(f"MCTS planner '{self.planner_name}' does not implement the 'simulate' function!")

    @staticmethod
    def select_best_child(root: Node) -> Node:
        best_child = root.children[0]
        for child in root.children:
            if child.value > best_child.value:
                best_child = child

        return best_child

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        self.mission_id = kwargs["mission_id"]
        root = Node(State(budget, self.mapper.train_data_map.count_map.copy(), previous_pose), parent=None)

        for i in range(self.num_simulations):
            self.simulate(root, self.horizon_length)

        if len(root.children) == 0:
            return None

        return root.select_best_child().state.pose


class DiscreteMCTSPlanner(MCTSPlanner):
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        budget: float,
        objective_fn_name: str,
        num_simulations: int = 100,
        gamma: float = 0.99,
        horizon_length: int = 5,
        exploration_constant: float = 2.0,
        eps_greedy_prob: float = 0.5,
        step_sizes: List = [30.0, 50.0, 70.0],
    ):
        super(DiscreteMCTSPlanner, self).__init__(
            mapper,
            altitude,
            sensor_info,
            uav_specifications,
            budget,
            objective_fn_name,
            num_simulations,
            gamma,
            horizon_length,
            exploration_constant,
        )

        self.planner_name = "discrete-mcts"
        self.eps_greedy_prob = eps_greedy_prob
        self.step_sizes = step_sizes
        self.action_space = DiscreteActionSpace(
            step_sizes, altitude, uav_specifications, self.world_size, self.sensor_angle
        )

    def greedy_action_id(self, state: State) -> int:
        greedy_action_id = None
        max_reward = -np.inf

        for action_id in self.action_space.get_valid_actions(state.pose, state.remaining_budget):
            next_state = self.prediction_step(state, self.action_space.actions[action_id])
            reward = self.compute_reward(state, next_state)
            if reward > max_reward:
                greedy_action_id = action_id
                max_reward = reward

        return greedy_action_id

    def eps_greedy_policy(self, state: State) -> int:
        if np.random.uniform(0, 1) < self.eps_greedy_prob:
            return self.greedy_action_id(state)

        return self.action_space.sample(state.pose, state.remaining_budget)

    def rollout(self, state: State, depth: int) -> float:
        if depth == 0 or state.remaining_budget < min(self.step_sizes):
            return 0

        sampled_action_id = self.eps_greedy_policy(state)
        next_state = self.prediction_step(state, self.action_space.actions[sampled_action_id])
        reward = self.compute_reward(state, next_state)

        return reward + self.gamma * self.rollout(next_state, depth - 1)

    def simulate(self, node: Node, depth: int) -> float:
        if depth == 0 or node.state.remaining_budget < min(self.step_sizes):
            return 0

        if node.visits == 0:
            for action_id in self.action_space.get_valid_actions(node.state.pose, node.state.remaining_budget):
                node.children[action_id] = Node(
                    self.prediction_step(node.state, self.action_space.actions[action_id]), parent=node
                )

            return self.rollout(node.state, depth)

        next_node, _ = node.select_utc_child(c=self.exploration_constant)
        reward = self.compute_reward(node.state, next_node.state)
        value = reward + self.gamma * self.simulate(next_node, depth - 1)

        node.visits += 1
        next_node.visits += 1
        next_node.value += (value - next_node.value) / next_node.visits

        return value
