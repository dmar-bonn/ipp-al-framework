from typing import Dict

from mapper.terrain_mapper import TerrainMapper


def get_mapper(cfg: Dict, model_cfg: Dict) -> TerrainMapper:
    simulator_name = cfg["simulator"]["name"]

    if simulator_name not in cfg["simulator"].keys():
        raise KeyError(f"No simulation with name '{simulator_name}' specified in config file!")

    return TerrainMapper(cfg, model_cfg, simulator_name)
