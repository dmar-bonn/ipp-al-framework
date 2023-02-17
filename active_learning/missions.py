from typing import Dict

import cv2
import numpy as np
import torch
from agri_semantics.utils.utils import infer_anno_and_epistemic_uncertainty_from_image
from pytorch_lightning import LightningModule

from mapper.terrain_mapper import TerrainMapper
from planner.common import Planner
from planner.common import compute_flight_time, compute_representation_score
from simulator import Simulator
from utils.logger import Logger


class Mission:
    def __init__(
        self,
        planner: Planner,
        mapper: TerrainMapper,
        simulator: Simulator,
        model: LightningModule,
        init_pose: np.array,
        cfg: Dict,
        model_cfg: Dict,
        logger: Logger,
    ):
        self.logger = logger
        self.planner = planner
        self.mapper = mapper
        self.simulator = simulator
        self.model = model
        self.init_pose = init_pose
        self.altitude = cfg["planner"]["altitude"]
        self.budget = cfg["planner"]["budget"]
        self.uav_specifications = cfg["planner"]["uav_specifications"]
        self.use_informed_map_prior = cfg["planner"]["informed_map_prior"]
        self.map_continuous_sensor_stream = cfg["mapper"]["map_continuous_sensor_stream"]
        self.simulator_name = cfg["simulator"]["name"]
        self.sensor_resolution = cfg["simulator"][self.simulator_name]["sensor"]["resolution"]
        self.cfg = cfg
        self.model_cfg = model_cfg

    def update_train_data_representations(self):
        for measurement in self.logger.all_train_poses:
            map_data = self.infer_map_data(measurement)
            self.logger.add_train_data_representation(map_data["hidden_representation"])

    def compute_informed_map_prior(self, mission_id: int):
        for measurement in self.logger.all_waypoints:
            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)

        for measurement in self.logger.all_train_poses:
            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)

        file_id = f"{self.cfg['simulator']['name']}_{self.cfg['planner']['type']}_{mission_id}_prior"
        self.logger.save_maps_to_disk(
            self.mapper.terrain_map.semantic_map,
            self.mapper.epistemic_map.mean_map,
            self.mapper.representation_score_map.mean_map,
            file_id,
            self.cfg["mapper"]["map_name"],
        )

    def infer_map_data(self, measurement: Dict) -> Dict:
        image = cv2.cvtColor(measurement["image"], cv2.COLOR_BGR2RGB)
        image = image.transpose(1, 0, 2)
        probs, uncertainty, hidden_representation = infer_anno_and_epistemic_uncertainty_from_image(
            self.model,
            image,
            num_mc_epistemic=self.model_cfg["train"]["num_mc_epistemic"],
            resize_image=False,
            aleatoric_model=self.model_cfg["model"]["aleatoric_model"],
            num_mc_aleatoric=self.model_cfg["train"]["num_mc_aleatoric"],
            ensemble_model=self.model_cfg["model"]["ensemble_model"],
            task=self.model_cfg["model"]["task"],
        )

        _, preds = torch.max(torch.from_numpy(probs), dim=0)
        image, preds, uncertainty, hidden_representation, probs = (
            image.transpose(1, 0, 2),
            preds.transpose(1, 0),
            uncertainty.transpose(1, 0),
            hidden_representation.transpose(0, 2, 1),
            probs.transpose(0, 2, 1),
        )
        representation_score = np.zeros((self.sensor_resolution[1], self.sensor_resolution[0]))
        if self.planner.objective_fn_name == "representation":
            representation_score = compute_representation_score(
                self.logger,
                hidden_representation,
                self.sensor_resolution,
                score_fn_name=self.cfg["planner"]["score_fn"]["name"],
                score_fn_mode=self.cfg["planner"]["score_fn"]["mode"],
                score_fn_params=self.cfg["planner"]["score_fn"]["params"],
            )

        return {
            "logits": probs,
            "uncertainty": uncertainty,
            "hidden_representation": hidden_representation,
            "representation_score": representation_score,
            "fov": measurement["fov"],
            "gsd": measurement["gsd"],
            "is_train_data": measurement["is_train_data"],
        }

    def execute(self, mission_id: int):
        if self.use_informed_map_prior:
            self.compute_informed_map_prior(mission_id)

        self.simulator.start_mission(self.init_pose)

        previous_pose = self.init_pose
        timestep = 0
        while self.budget > 0:
            measurement = self.simulator.get_measurement(previous_pose, True, mission_id)

            self.logger.add_train_pose(previous_pose, measurement)
            self.logger.save_train_data_to_disk(
                measurement["image"], measurement["anno"], self.model_cfg["data"]["path_to_dataset"]
            )

            map_data = self.infer_map_data(measurement)
            self.logger.add_train_data_representation(map_data["hidden_representation"])
            self.logger.save_qualitative_results(measurement, map_data, mission_id, timestep, self.mapper.map_name)
            self.mapper.update_map(map_data)

            pose = self.planner.replan(
                self.budget,
                previous_pose,
                uncertainty_image=map_data["uncertainty"],
                representation_image=map_data["representation_score"],
                mission_id=mission_id,
            )
            if pose is None:
                print(f"FINISHED '{self.planner.planner_name}' PLANNING MISSION")
                print(f"CHOSEN PATH: {self.logger.mission_train_poses}")
                break

            if self.map_continuous_sensor_stream:
                self.reach_next_pose(previous_pose, pose, mission_id)

            self.simulator.move_to_next_waypoint(pose)
            self.budget -= compute_flight_time(pose, previous_pose, uav_specifications=self.uav_specifications)
            print(f"REACHED NEXT POSE: {pose}, REMAINING BUDGET: {self.budget}")

            previous_pose = pose
            timestep += 1

        file_id = f"{self.cfg['simulator']['name']}_{self.cfg['planner']['type']}_{mission_id}"
        self.logger.save_maps_to_disk(
            self.mapper.terrain_map.semantic_map,
            self.mapper.epistemic_map.mean_map,
            self.mapper.representation_score_map.mean_map,
            file_id,
            self.cfg["mapper"]["map_name"],
        )
        self.logger.save_path_to_disk(file_id)
        self.logger.save_train_data_stats(self.model_cfg, file_id)

    def reach_next_pose(self, current_pose: np.array, next_pose: np.array, mission_id: int):
        distance = np.linalg.norm(next_pose - current_pose, ord=2)
        sensor_frequency = self.cfg["simulator"][self.simulator_name]["sensor"]["frequency"]
        num_waypoints = int(distance * sensor_frequency / self.uav_specifications["max_v"])

        waypoints_x = np.linspace(current_pose[0], next_pose[0], num=num_waypoints, endpoint=False)
        waypoints_y = np.linspace(current_pose[1], next_pose[1], num=num_waypoints, endpoint=False)
        waypoints_z = self.altitude * np.ones(num_waypoints)
        waypoints = np.array([waypoints_x, waypoints_y, waypoints_z]).T[1:]

        for waypoint in waypoints:
            self.simulator.move_to_next_waypoint(waypoint)
            measurement = self.simulator.get_measurement(waypoint, False, mission_id)
            map_data = self.infer_map_data(measurement)
            self.mapper.update_map(map_data)
            self.logger.add_waypoint(waypoint, measurement)
