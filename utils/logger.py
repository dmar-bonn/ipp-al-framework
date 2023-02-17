import os
from typing import Dict, List

import cv2
import matplotlib
import numpy as np
import seaborn as sns
import torch
import yaml
from agri_semantics.datasets import get_data_module
from agri_semantics.utils.utils import toOneHot

from utils.notifiers import Notifier
from utils.notifiers.slack import SlackNotifier
from utils.notifiers.telegram import TelegramNotifier
from utils.utils import load_from_env

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Logger:
    def __init__(self, experiment_name: str, cfg: Dict, model_cfg: Dict):
        self.logger_name = experiment_name

        self.setup_log_dir()

        self.mission_waypoints = np.empty((0, 3), np.float32)
        self.all_waypoints = []

        self.mission_train_poses = np.empty((0, 3), np.float32)
        self.all_train_poses = []

        self.cfg = cfg
        self.model_cfg = model_cfg

        self.simulator_name = cfg["simulator"]["name"]
        self.train_data_representations = []

        self.notifiers = self.setup_notifiers()

    def setup_notifiers(self) -> List[Notifier]:
        notifiers = []
        if self.cfg["notifications"]["telegram"]["is_used"]:
            notifiers.append(
                TelegramNotifier(
                    self.logger_name,
                    load_from_env("TELEGRAM_TOKEN", str, "my_telegram_token"),
                    load_from_env("TELEGRAM_CHAT_ID", str, "my_telegram_chat_id"),
                    cfg=self.cfg,
                    model_cfg=self.model_cfg,
                    verbose=self.cfg["notifications"]["telegram"]["verbose"],
                )
            )

        if self.cfg["notifications"]["slack"]["is_used"]:
            notifiers.append(
                SlackNotifier(
                    self.logger_name,
                    load_from_env("SLACK_WEBHOOK", str, "my_slack_webhook"),
                    load_from_env("SLACK_BOTNAME", str, "my_slack_botname"),
                    icon=self.cfg["notifications"]["slack"]["icon"],
                    cfg=self.cfg,
                    model_cfg=self.model_cfg,
                    verbose=self.cfg["notifications"]["slack"]["verbose"],
                )
            )

        return notifiers

    def finished_mission(self, mission_id: int, test_statistics: Dict):
        self.save_evaluation_metrics_to_disk(test_statistics)
        self.reset_mission_train_poses()
        self.reset_mission_waypoint_poses()
        self.reset_train_data_representations()

        for notifier in self.notifiers:
            notifier.finished_iteration(mission_id, additional_info=test_statistics)

    def setup_log_dir(self):
        if os.path.exists(self.logger_name):
            raise ValueError(f"{self.logger_name} log directory already exists!")

        os.makedirs(self.logger_name)

    def reset_mission_train_poses(self):
        self.mission_train_poses = np.empty((0, 3), np.float32)

    def reset_mission_waypoint_poses(self):
        self.mission_waypoints = np.empty((0, 3), np.float32)

    def add_waypoint(self, waypoint: np.array, measurement: Dict):
        self.mission_waypoints = np.append(self.mission_waypoints, [waypoint], axis=0)
        self.all_waypoints.append(measurement)

    def add_train_pose(self, pose: np.array, measurement: Dict):
        self.mission_train_poses = np.append(self.mission_train_poses, [pose], axis=0)
        self.all_train_poses.append(measurement)

    @staticmethod
    def save_train_data_to_disk(
        image: np.array, anno: np.array, dataset_path: str, dataset_folder: str = "training_set"
    ):
        anno_dir = os.path.join(dataset_path, dataset_folder, "anno")
        image_dir = os.path.join(dataset_path, dataset_folder, "image")

        if not os.path.exists(anno_dir):
            os.makedirs(anno_dir)

        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        train_data_id = len([name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))])
        image_filepath = os.path.join(image_dir, f"rgb_{str(train_data_id).zfill(5)}.png")
        anno_filepath = os.path.join(anno_dir, f"gt_{str(train_data_id).zfill(5)}.png")

        cv2.imwrite(image_filepath, image)
        cv2.imwrite(anno_filepath, anno)

    def save_qualitative_results(
        self, measurement: Dict, map_data: Dict, mission_id: int, timestep: int, map_name: str
    ):
        if not os.path.exists(os.path.join(self.logger_name, "qualitative_results")):
            os.makedirs(os.path.join(self.logger_name, "qualitative_results"))

        folder_path = os.path.join(self.logger_name, "qualitative_results", f"mission_{mission_id}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        cv2.imwrite(os.path.join(folder_path, f"image_{timestep}.png"), measurement["image"])
        cv2.imwrite(os.path.join(folder_path, f"anno_{timestep}.png"), measurement["anno"])

        prediction = toOneHot(torch.from_numpy(map_data["logits"]).unsqueeze(0), map_name)
        cv2.imwrite(os.path.join(folder_path, f"pred_{timestep}.png"), cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))

        plt.imsave(
            os.path.join(folder_path, f"representation_score_{timestep}.png"),
            np.squeeze(map_data["representation_score"]),
            cmap="plasma",
        )
        plt.imsave(
            os.path.join(folder_path, f"uncertainty_{timestep}.png"),
            np.squeeze(map_data["uncertainty"]),
            cmap="plasma",
        )

    def add_train_data_representation(self, hidden_representation: np.array):
        _, x_dim, y_dim = hidden_representation.shape
        for x in range(x_dim):
            for y in range(y_dim):
                self.train_data_representations.append(hidden_representation[:, x, y])

    def reset_train_data_representations(self):
        self.train_data_representations = []

    def save_maps_to_disk(
        self,
        semantic_map: np.array,
        epistemic_map: np.array,
        representation_score_map: np.array,
        file_id: str,
        map_name: str,
    ):
        if semantic_map.shape[0] > 1:
            semantic_map_name = "semantics"
            plt.imsave(
                os.path.join(self.logger_name, f"{semantic_map_name}_{file_id}.png"),
                toOneHot(torch.from_numpy(semantic_map).unsqueeze(0), map_name),
            )
        else:
            semantic_map_name = "elevation"
            plt.imsave(
                os.path.join(self.logger_name, f"{semantic_map_name}_{file_id}.png"),
                np.squeeze(semantic_map),
                cmap="gray",
            )

        plt.imsave(
            os.path.join(self.logger_name, f"uncertainty_{file_id}.png"), np.squeeze(epistemic_map), cmap="plasma"
        )
        plt.imsave(
            os.path.join(self.logger_name, f"representation_{file_id}.png"),
            np.squeeze(representation_score_map),
            cmap="plasma",
        )

        with open(os.path.join(self.logger_name, f"semantic_map_{file_id}.npy"), "wb") as file:
            np.save(file, semantic_map)

        with open(os.path.join(self.logger_name, f"epistemic_map_{file_id}.npy"), "wb") as file:
            np.save(file, epistemic_map)

        with open(os.path.join(self.logger_name, f"representation_score_map_{file_id}.npy"), "wb") as file:
            np.save(file, representation_score_map)

        plt.clf()
        plt.cla()

    def save_path_to_disk(self, file_id: str):
        plt.plot(self.mission_train_poses[:, 0], self.mission_train_poses[:, 1], "-ok")
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.logger_name, f"path_{file_id}.png"))

        with open(os.path.join(self.logger_name, f"path_poses_{file_id}.npy"), "wb") as file:
            np.save(file, self.mission_train_poses)

        plt.clf()
        plt.cla()

    def save_config_files_to_disk(self, cfg: Dict, model_cfg: Dict):
        with open(os.path.join(self.logger_name, "config.yaml"), "w") as file:
            yaml.dump(cfg, file)

        with open(os.path.join(self.logger_name, "model_config.yaml"), "w") as file:
            yaml.dump(model_cfg, file)

    def save_evaluation_metrics_to_disk(self, test_statistics: Dict):
        with open(os.path.join(self.logger_name, "evaluation_metrics.yaml"), "w") as file:
            yaml.dump(test_statistics, file)

    def save_train_data_stats(self, model_cfg: Dict, file_id: str):
        dataloader = get_data_module(model_cfg)
        dataloader.setup(stage=None)

        train_data_stats = torch.zeros(model_cfg["model"]["num_classes"])
        for batch in dataloader.train_dataloader():
            class_ids, class_counts = torch.unique(batch["anno"], return_counts=True)
            train_data_stats[class_ids] += class_counts

        total_num_pixels = torch.sum(train_data_stats).item()
        train_data_stats = {
            class_id: class_count.item() / total_num_pixels * 100
            for class_id, class_count in enumerate(train_data_stats)
        }

        ax = sns.barplot(x=list(train_data_stats.keys()), y=list(train_data_stats.values()))
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Class Frequency [%]")
        plt.savefig(os.path.join(self.logger_name, f"train_data_stats_{file_id}.png"), dpi=300)

        plt.clf()
        plt.cla()
