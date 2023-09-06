import copy
import random
from typing import Dict, List, Optional

import numpy as np

from mapper.terrain_mapper import TerrainMapper
from planner.action_spaces import DiscreteActionSpace
from planner.common import compute_flight_time, Planner
from utils import utils


class State:
    def __init__(self, remaining_budget: float, pose: np.array):
        self.pose = pose
        self.remaining_budget = remaining_budget
        self.state_id = self.create_state_id()

    def create_state_id(self) -> str:
        x, y, z = round(self.pose[0], 2), round(self.pose[1], 2), round(self.pose[2], 2)
        budget = round(self.remaining_budget, 2)
        return f"{x}_{y}_{z}_{budget}"


class Node:
    def __init__(self, state: State, node_id: int, depth: int, parent=None):
        self.node_id = node_id
        self.depth = depth
        self.parent = parent
        self.state = state
        self.value = 0
        self.visits = 0
        self.children = {}

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
        self.nodes = {}
        self.max_node_id = 0

    def compute_uct(self, node: Node, min_val: float, max_val: float):
        if node.visits == 0:
            return np.inf

        exploration = self.exploration_constant * np.sqrt(np.log(node.parent.visits) / node.visits)

        if max_val == 0:
            return node.value + exploration

        if max_val == min_val:
            normalized_value = (node.value - min_val) / max_val
        else:
            normalized_value = (node.value - min_val) / (max_val - min_val)

        return normalized_value + exploration

    def select_utc_child(self, node: Node):
        max_children = []
        max_action_ids = []
        max_uct = -np.inf

        children_vals = [child.value for child in node.children.values()]
        min_child_val, max_child_val = min(children_vals), max(children_vals)

        for action_id in node.children.keys():
            uct_child = self.compute_uct(node.children[action_id], min_child_val, max_child_val)
            if max_uct == uct_child:
                max_children.append(node.children[action_id])
                max_action_ids.append(action_id)
            if uct_child > max_uct:
                max_uct = uct_child
                max_children = [node.children[action_id]]
                max_action_ids = [action_id]

        sampled_node_id = np.random.choice(len(max_children))
        return max_children[sampled_node_id], max_action_ids[sampled_node_id]

    def compute_reward(self, state: State, next_state: State, train_data_count_map: np.array):
        (uncertainty_submap, representation_submap, hit_submap, _, fov_indices) = self.mapper.get_map_state(
            next_state.pose
        )
        lu_x, lu_y, rd_x, rd_y = fov_indices
        train_data_count_submap = train_data_count_map[lu_y:rd_y, lu_x:rd_x]

        unknown_space_mask = (train_data_count_submap == 0) & (hit_submap == 0)
        uncertainty_submap[unknown_space_mask] = self.mapper.uncertainty_prior_const
        representation_submap[unknown_space_mask] = self.mapper.representation_prior_const

        score_submap = self.get_schematic_image(uncertainty_submap, representation_submap, self.mission_id)
        return np.sum(score_submap) / (np.sum(train_data_count_submap) + 1)

    def forward_simulate_train_data_map(self, train_data_count_map: np.array, next_pose: np.array) -> np.array:
        fov_corners, _ = utils.get_fov(next_pose, self.sensor_angle, self.mapper.gsd, self.mapper.world_range)
        lu, _, rd, _ = fov_corners
        lu_x, lu_y = self.mapper.find_map_index(lu)
        rd_x, rd_y = self.mapper.find_map_index(rd)
        next_train_data_count_map = train_data_count_map
        next_train_data_count_map[lu_y:rd_y, lu_x:rd_x] += 1
        return next_train_data_count_map

    def predict_next_state(self, state: State, action: np.array) -> State:
        next_pose = state.pose + action
        remaining_budget = state.remaining_budget - compute_flight_time(state.pose, next_pose, self.uav_specifications)
        return State(remaining_budget, next_pose)

    def rollout(self, state: State, depth: int, train_data_count_map: np.array) -> float:
        raise NotImplementedError(f"MCTS planner '{self.planner_name}' does not implement the 'rollout' function!")

    def simulate(self, node: Node, depth: int, train_data_count_map: np.array) -> float:
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
        root = Node(State(budget, previous_pose), self.max_node_id, self.horizon_length, parent=None)

        for i in range(self.num_simulations):
            self.simulate(root, self.horizon_length, copy.deepcopy(self.mapper.train_data_map.count_map))

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

    def greedy_action_id(self, state: State, train_data_count_map: np.array) -> int:
        greedy_action_id = None
        max_reward = -np.inf

        for action_id in self.action_space.get_valid_actions(state.pose, state.remaining_budget):
            next_state = self.predict_next_state(state, self.action_space.actions[action_id])
            reward = self.compute_reward(state, next_state, train_data_count_map)
            if reward > max_reward:
                greedy_action_id = action_id
                max_reward = reward

        return greedy_action_id

    def eps_greedy_policy(self, state: State, train_data_count_map: np.array) -> int:
        if np.random.uniform(0, 1) < self.eps_greedy_prob:
            return self.greedy_action_id(state, train_data_count_map)

        return self.action_space.sample(state.pose, state.remaining_budget)

    def rollout(self, state: State, depth: int, train_data_count_map: np.array) -> float:
        if depth == 0 or state.remaining_budget < min(self.step_sizes):
            return 0

        sampled_action_id = self.eps_greedy_policy(state, train_data_count_map)
        next_state = self.predict_next_state(state, self.action_space.actions[sampled_action_id])
        reward = self.compute_reward(state, next_state, train_data_count_map)
        next_train_data_count_map = self.forward_simulate_train_data_map(train_data_count_map, next_state.pose)

        return reward + self.gamma * self.rollout(next_state, depth - 1, next_train_data_count_map)

    def simulate(self, node: Node, depth: int, train_data_count_map: np.array) -> float:
        if depth == 0 or node.state.remaining_budget < min(self.step_sizes):
            return 0

        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node.node_id

            for action_id in self.action_space.get_valid_actions(node.state.pose, node.state.remaining_budget):
                child_state = self.predict_next_state(node.state, self.action_space.actions[action_id])
                next_node = Node(child_state, self.max_node_id + 1, depth - 1, parent=node)
                self.max_node_id += 1
                node.children[action_id] = next_node

            return self.rollout(node.state, depth, train_data_count_map)

        next_node, next_action_id = self.select_utc_child(node)
        reward = self.compute_reward(node.state, next_node.state, train_data_count_map)
        next_train_data_count_map = self.forward_simulate_train_data_map(train_data_count_map, next_node.state.pose)
        value = reward + self.gamma * self.simulate(next_node, depth - 1, next_train_data_count_map)

        node.visits += 1
        node.value += (value - node.value) / node.visits

        return value
