from typing import Dict

from mapper.terrain_mapper import TerrainMapper
from planner.baselines import CoveragePlanner, GlobalRandomPlanner, LocalRandomPlanner
from planner.cmaes import CMAESPlanner
from planner.common import Planner
from planner.frontier import FrontierPlanner
from planner.local_planners import ImagePlanner
from planner.mcts import DiscreteMCTSPlanner


def get_planner(
    cfg: Dict,
    mapper: TerrainMapper,
    **kwargs,
) -> Planner:
    simulator_name = cfg["simulator"]["name"]
    planner_type = cfg["planner"]["type"]
    planner_params = cfg["planner"][planner_type]

    if planner_type == "local_random":
        planner = LocalRandomPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            planner_params["step_size"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup()
        return planner
    elif planner_type == "global_random":
        planner = GlobalRandomPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["objective_fn"],
            planner_params["min_radius"],
            planner_params["max_radius"],
        )
        planner.setup()
        return planner
    elif planner_type == "coverage":
        planner = CoveragePlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            planner_params["step_sizes"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup(mission_id=kwargs["mission_id"])
        return planner
    elif planner_type == "cmaes":
        planner = CMAESPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["budget"],
            planner_params["max_iter"],
            planner_params["population_size"],
            planner_params["sigma0"],
            planner_params["horizon_length"],
            planner_params["lattice_step_size"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup()
        return planner
    elif planner_type == "frontier":
        planner = FrontierPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            planner_params["step_size"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup()
        return planner
    elif planner_type == "local_image":
        planner = ImagePlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            planner_params["step_size"],
            planner_params["edge_width"],
            cfg["planner"]["objective_fn"],
        )
        planner.setup()
        return planner
    elif planner_type == "discrete_mcts":
        planner = DiscreteMCTSPlanner(
            mapper,
            cfg["planner"]["altitude"],
            cfg["simulator"][simulator_name]["sensor"],
            cfg["planner"]["uav_specifications"],
            cfg["planner"]["budget"],
            cfg["planner"]["objective_fn"],
            planner_params["num_simulations"],
            planner_params["discount_factor"],
            planner_params["horizon_length"],
            planner_params["exploration_constant"],
            planner_params["eps_greedy_prob"],
            planner_params["step_sizes"],
        )
        planner.setup()
        return planner
    else:
        raise ValueError(f"Planner type '{planner_type}' unknown!")
