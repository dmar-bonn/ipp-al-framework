import os
from typing import Dict, Tuple

import click
import numpy as np
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

    if cfg["simulator"]["name"] == "potsdam":
        init_pose = np.array([30, 30, cfg["planner"]["altitude"]], dtype=np.float32)
    elif cfg["simulator"]["name"] == "rit18":
        init_pose = np.array([16, 16, cfg["planner"]["altitude"]], dtype=np.float32)
    elif cfg["simulator"]["name"] == "flightmare":
        init_pose = np.array([20, 15, cfg["planner"]["altitude"]], dtype=np.float32)
    else:
        raise ValueError(f"Simulator '{cfg['simulator']['name']}' not found!")

    simulator = get_simulator(cfg)
    learner = get_learner(model_cfg, cfg["network"]["path_to_checkpoint"], logger.logger_name, model_id="0")
    trained_model = learner.setup_model()

    for notifier in logger.notifiers:
        notifier.start_experiment()

    try:
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
    main()
