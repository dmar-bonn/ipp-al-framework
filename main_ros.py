import os
from typing import Dict, Tuple

import click
import numpy as np
import rospy
import yaml

from active_learning import get_learner
from active_learning.missions import Mission
from mapper import get_mapper
from planner import get_planner
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


def get_starting_pose(simulator_name: str, altitude: float, starting_position: str) -> np.array:
    if simulator_name == "potsdam":
        if starting_position == "top_left":
            x_pos, y_pos = 30, 30
        elif starting_position == "top_right":
            x_pos, y_pos = 870, 30
        elif starting_position == "bottom_left":
            x_pos, y_pos = 30, 870
        elif starting_position == "bottom_right":
            x_pos, y_pos = 870, 870
        else:
            raise ValueError(f"Starting position '{starting_position}' not found!")
    elif simulator_name == "rit18":
        if starting_position == "top_left":
            x_pos, y_pos = 16, 16
        elif starting_position == "top_right":
            x_pos, y_pos = 245, 16
        elif starting_position == "bottom_left":
            x_pos, y_pos = 16, 550
        elif starting_position == "bottom_right":
            x_pos, y_pos = 245, 550
        else:
            raise ValueError(f"Starting position '{starting_position}' not found!")
    elif simulator_name == "flightmare":
        if starting_position == "top_left":
            x_pos, y_pos = 20, 15
        elif starting_position == "top_right":
            x_pos, y_pos = 130, 15
        elif starting_position == "bottom_left":
            x_pos, y_pos = 20, 115
        elif starting_position == "bottom_right":
            x_pos, y_pos = 130, 115
        else:
            raise ValueError(f"Starting position '{starting_position}' not found!")
    else:
        raise ValueError(f"Simulator '{simulator_name}' not found!")

    return np.array([x_pos, y_pos, altitude], dtype=np.float32)


@click.command()
@click.option(
    "--config_file",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml"),
)
def main(config_file: str):
    cfg, model_cfg = read_config_files(config_file)

    experiment_name = f"{cfg['simulator']['name']}_{cfg['planner']['type']}"
    logger = Logger(experiment_name, cfg, model_cfg)
    logger.save_config_files_to_disk(cfg, model_cfg)

    simulator = get_simulator(cfg)
    learner = get_learner(model_cfg, cfg["network"]["path_to_checkpoint"], logger.logger_name, model_id="0")
    trained_model = learner.setup_model()

    for notifier in logger.notifiers:
        notifier.start_experiment()

    try:
        init_pose = get_starting_pose(
            cfg["simulator"]["name"], cfg["planner"]["altitude"], cfg["planner"]["starting_position"]
        )

        for mission_id in range(cfg["planner"]["num_missions"]):
            mapper = get_mapper(cfg, model_cfg)
            planner = get_planner(cfg, mapper, mission_id=mission_id)
            mission = Mission(planner, mapper, simulator, trained_model, init_pose, cfg, model_cfg, logger)
            simulator.start_mission(init_pose)

            mission.execute(mission_id)
            trained_model = learner.train(mission_id)
            learner.evaluate()

            logger.finished_mission(mission_id, learner.test_statistics)
            mission.update_train_data_representations()

        for notifier in logger.notifiers:
            notifier.finish_experiment(additional_info=learner.test_statistics)

    except Exception as e:
        for notifier in logger.notifiers:
            notifier.failed_experiment(e)

        raise Exception(e)


if __name__ == "__main__":
    rospy.init_node("al_pipeline")
    main()
    rospy.spin()
