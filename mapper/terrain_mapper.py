from typing import Dict, List, Tuple, Union

import numpy as np

from utils import utils


class CountMap:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.count_map = self.init_map()

    def init_map(self) -> np.array:
        return np.zeros((self.map_boundary[1], self.map_boundary[0]))

    def update(self, map_indices: np.array):
        self.count_map[map_indices[:, 0], map_indices[:, 1]] += 1


class DiscreteVariableMap:
    def __init__(self, cfg: Dict, model_cfg: Dict, num_classes: int):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.num_classes = num_classes

        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.log_odds_map = self.init_map()

    def init_map(self) -> np.array:
        return self.log_odds_prior_const * np.ones((self.num_classes, self.map_boundary[1], self.map_boundary[0]))

    @property
    def log_odds_prior_const(self) -> float:
        prob_prior = 1 / self.num_classes
        return np.log(prob_prior / (1 - prob_prior))

    @property
    def prob_map(self) -> np.array:
        return 1 - (1 / (1 + np.exp(self.log_odds_map)))

    @property
    def semantic_map(self) -> np.array:
        return self.prob_map

    @property
    def uncertainty_map(self) -> np.array:
        return -np.sum(np.log(self.prob_map) * self.prob_map, axis=0)

    def update(self, map_indices: np.array, probs_measured: np.array, **kwargs):
        probs_measured = np.clip(probs_measured, a_min=10 ** (-6), a_max=1 - 10 ** (-6))
        probs_measured /= np.sum(probs_measured, axis=0)
        log_odds_measured = np.log(probs_measured / (1 - probs_measured))
        self.log_odds_map[:, map_indices[:, 0], map_indices[:, 1]] += log_odds_measured - self.log_odds_prior_const


class ContinuousVariableMap:
    def __init__(self, cfg: Dict, model_cfg: Dict, hit_map: CountMap, update_type: str, num_dimensions: int):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.update_type = update_type
        self.num_dimensions = num_dimensions

        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.mean_map, self.variance_map, self.hit_map = self.init_map(hit_map)

    def init_map(self, hit_map: CountMap) -> Tuple[np.array, np.array, CountMap]:
        mean_map = self.mean_prior_const * np.ones((self.num_dimensions, self.map_boundary[1], self.map_boundary[0]))
        variance_map = self.variance_prior_const * np.ones((self.map_boundary[1], self.map_boundary[0]))

        return mean_map, variance_map, hit_map

    @property
    def mean_prior_const(self) -> float:
        if self.update_type == "map":
            return self.model_cfg["model"]["value_range"]["max_value"] / 2
        else:
            return 0

    @property
    def variance_prior_const(self) -> float:
        if self.update_type == "map":
            return self.model_cfg["model"]["value_range"]["max_value"] / 6
        else:
            return 0

    @property
    def semantic_map(self) -> np.array:
        return self.mean_map

    @property
    def uncertainty_map(self) -> np.array:
        return self.variance_map

    def update(self, map_indices: np.array, mean_measured: np.array, variance_measured: np.array = None):
        if self.update_type == "mle":
            self.mean_map[:, map_indices[:, 0], map_indices[:, 1]] = self.maximum_likelihood_update(
                mean_measured,
                self.mean_map[:, map_indices[:, 0], map_indices[:, 1]],
                self.hit_map.count_map[map_indices[:, 0], map_indices[:, 1]],
            )
        elif self.update_type == "map":
            if variance_measured is None:
                raise ValueError(f"Bayesian continuous variable map update requires measurement variance!")
            (
                self.mean_map[:, map_indices[:, 0], map_indices[:, 1]],
                self.variance_map[map_indices[:, 0], map_indices[:, 1]],
            ) = self.kalman_update(
                mean_measured,
                self.mean_map[:, map_indices[:, 0], map_indices[:, 1]],
                variance_measured,
                self.variance_map[map_indices[:, 0], map_indices[:, 1]],
            )
        else:
            raise NotImplementedError(f"Continuous variable map update type '{self.update_type}' not implemented!")

    @staticmethod
    def maximum_likelihood_update(mean_measured: np.array, mean_prior: np.array, hit_map: np.array) -> np.array:
        return mean_prior + (mean_measured - mean_prior) / hit_map

    @staticmethod
    def kalman_update(
        mean_measured: np.array, mean_prior: np.array, variance_measured: np.array, variance_prior: np.array
    ) -> Tuple[np.array, np.array]:
        kalman_gain = variance_prior / (variance_prior + variance_measured + 10 ** (-8))
        mean_post = mean_prior + kalman_gain * (mean_measured - mean_prior)
        variance_post = (1 - kalman_gain) * variance_prior

        return mean_post, variance_post


class TerrainMapper:
    def __init__(self, cfg: Dict, model_cfg: Dict, simulator_name: str):
        self.cfg = cfg
        self.model_cfg = model_cfg

        self.map_name = cfg["mapper"]["map_name"]
        self.map_boundary = cfg["mapper"]["map_boundary"]
        self.ground_resolution = cfg["mapper"]["ground_resolution"]

        self.world_range = cfg["simulator"][simulator_name]["world_range"]
        self.sensor_angle = cfg["simulator"][simulator_name]["sensor"]["angle"]
        self.gsd = cfg["simulator"][simulator_name]["gsd"]

        (
            self.terrain_map,
            self.epistemic_map,
            self.representation_score_map,
            self.hit_map,
            self.train_data_map,
        ) = self.init_map()

    @property
    def class_num(self) -> int:
        task = self.cfg["simulator"]["task"]
        if task == "classification":
            return self.cfg["mapper"]["class_number"]
        else:
            raise NotImplementedError(f"Mapping for {task} task is not implemented!")

    def init_map(
        self,
    ) -> Tuple[
        Union[ContinuousVariableMap, DiscreteVariableMap],
        ContinuousVariableMap,
        ContinuousVariableMap,
        CountMap,
        CountMap,
    ]:
        hit_map = CountMap(self.cfg)
        epistemic_map = ContinuousVariableMap(self.cfg, self.model_cfg, hit_map, "mle", 1)
        representation_score_map = ContinuousVariableMap(self.cfg, self.model_cfg, hit_map, "mle", 1)

        task = self.cfg["simulator"]["task"]
        if task == "classification":
            terrain_map = DiscreteVariableMap(self.cfg, self.model_cfg, self.class_num)
        else:
            raise NotImplementedError(f"Semantic map update for '{task}' task not implemented!")

        return terrain_map, epistemic_map, representation_score_map, hit_map, CountMap(self.cfg)

    @property
    def representation_prior_const(self) -> float:
        return 0.7

    @property
    def uncertainty_prior_const(self) -> float:
        task = self.cfg["simulator"]["task"]
        if task == "classification":
            use_entropy_criterion = (
                self.model_cfg["train"]["num_mc_epistemic"] <= 1 and not self.model_cfg["model"]["ensemble_model"]
            )
            if use_entropy_criterion:
                return np.log(self.class_num)

            return 0.1
        else:
            raise NotImplementedError(f"Mapping for {task} task is not implemented!")

    def find_map_index(self, data_point) -> Tuple[float, float]:
        x_index = np.floor(data_point[0] / self.ground_resolution[0]).astype(int)
        y_index = np.floor(data_point[1] / self.ground_resolution[1]).astype(int)

        return x_index, y_index

    def update_map(self, data_source: Dict):
        semantics = data_source["logits"]
        uncertainty = data_source["uncertainty"]
        representation_score = data_source["representation_score"]
        fov = data_source["fov"]
        gsd = data_source["gsd"]
        is_train_data = data_source["is_train_data"]
        _, m_y_dim, m_x_dim = semantics.shape

        measurement_indices = np.array(np.meshgrid(np.arange(m_y_dim), np.arange(m_x_dim))).T.reshape(-1, 2).astype(int)
        x_ground = fov[0][0] + (0.5 + np.arange(m_x_dim)) * gsd[0]
        y_ground = fov[0][1] + (0.5 + np.arange(m_y_dim)) * gsd[1]
        ground_coords = np.array(np.meshgrid(y_ground, x_ground)).T.reshape(-1, 2)
        map_indices = np.floor(ground_coords / np.array(self.ground_resolution)).astype(int)

        semantics_proj = semantics[:, measurement_indices[:, 0], measurement_indices[:, 1]]
        uncertainty_proj = uncertainty[measurement_indices[:, 0], measurement_indices[:, 1]]
        rep_score_proj = representation_score[measurement_indices[:, 0], measurement_indices[:, 1]]

        self.hit_map.update(map_indices)
        if is_train_data:
            self.train_data_map.update(map_indices)

        self.epistemic_map.update(map_indices, uncertainty_proj, variance_measured=None)
        self.representation_score_map.update(map_indices, rep_score_proj, variance_measured=None)
        self.terrain_map.update(map_indices, semantics_proj, variance_measured=uncertainty_proj)

    def get_map_state(self, pose: np.array) -> Tuple[np.array, np.array, np.array, np.array, List]:
        fov_corners, _ = utils.get_fov(pose, self.sensor_angle, self.gsd, self.world_range)
        lu, _, rd, _ = fov_corners
        lu_x, lu_y = self.find_map_index(lu)
        rd_x, rd_y = self.find_map_index(rd)

        return (
            self.epistemic_map.mean_map[0, lu_y:rd_y, lu_x:rd_x].copy(),
            self.representation_score_map.mean_map[0, lu_y:rd_y, lu_x:rd_x].copy(),
            self.hit_map.count_map[lu_y:rd_y, lu_x:rd_x].copy(),
            self.train_data_map.count_map[lu_y:rd_y, lu_x:rd_x].copy(),
            [lu_x, lu_y, rd_x, rd_y],
        )
