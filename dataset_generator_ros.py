import os
from typing import Dict, Tuple

import click
import numpy as np
import rospy
import yaml

from mapper import get_mapper, TerrainMapper
from simulator.load_simulators import get_simulator
from utils.logger import Logger


def read_config_files(config_file_path: str) -> Tuple[Dict, Dict]:
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Cannot find config file '{config_file_path}'!")

    if not config_file_path.endswith((".yaml", ".yml")):
        raise ValueError(f"Config file is not a yaml-file! Only '.yaml' or '.yaml' file endings allowed!")

    with open(config_file_path, "r") as file:
        cfg = yaml.safe_load(file)

    with open(cfg["network"]["path_to_config"], "r") as config_file:
        model_cfg = yaml.safe_load(config_file)

    return cfg, model_cfg


def sample_random_pose(altitude: float, mapper: TerrainMapper) -> np.array:
    boundary_space = altitude * np.tan(np.deg2rad(mapper.sensor_angle))
    max_y = mapper.map_boundary[1] * mapper.ground_resolution[1] - boundary_space[1]
    max_x = mapper.map_boundary[0] * mapper.ground_resolution[0] - boundary_space[0]

    sampled_y = np.random.uniform(low=boundary_space[1], high=max_y)
    sampled_x = np.random.uniform(low=boundary_space[0], high=max_x)
    return np.array([sampled_x, sampled_y, altitude], dtype=np.float32)


@click.command()
@click.option(
    "--config_file",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"),
)
@click.option(
    "--dataset_folder",
    "-d",
    type=str,
    help="dataset folder name, either 'training_set', 'validation_set' or 'test_set'",
    default="training_set",
)
@click.option(
    "--num_data_samples",
    "-n",
    type=int,
    help="number of to-be-generated image-annotation data points",
    default=1000,
)
def main(config_file: str, dataset_folder: str, num_data_samples: int):
    cfg, model_cfg = read_config_files(config_file)

    experiment_name = f"{cfg['simulator']['name']}_{cfg['planner']['type']}"
    logger = Logger(experiment_name, cfg, model_cfg)

    simulator = get_simulator(cfg)
    mapper = get_mapper(cfg, model_cfg)

    simulator.start_mission(np.array([0, 0, cfg["planner"]["altitude"]], dtype=np.float32))
    for _ in range(num_data_samples):
        random_pose = sample_random_pose(cfg["planner"]["altitude"], mapper)
        simulator.move_to_next_waypoint(random_pose)
        measurement = simulator.get_measurement(random_pose, True, 0)

        logger.save_train_data_to_disk(
            measurement["image"],
            measurement["anno"],
            model_cfg["data"]["path_to_dataset"],
            dataset_folder=dataset_folder,
        )


if __name__ == "__main__":
    rospy.init_node("dataset_generator")
    main()
    rospy.spin()
