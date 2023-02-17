from typing import Dict, List, Optional

import numpy as np

from mapper.terrain_mapper import TerrainMapper
from utils.logger import Logger


def compute_flight_time(action: np.array, previous_action: np.array, uav_specifications: Dict = None) -> float:
    dist_total = np.linalg.norm(action - previous_action, ord=2)
    dist_acc = min(dist_total * 0.5, np.square(uav_specifications["max_v"]) / (2 * uav_specifications["max_a"]))
    dist_const = dist_total - 2 * dist_acc

    time_acc = np.sqrt(2 * dist_acc / uav_specifications["max_a"])
    time_const = dist_const / uav_specifications["max_v"]
    time_total = time_const + 2 * time_acc

    return time_total


def patchwise_cosine_similarity(representation_3d: np.array, train_representations: np.array) -> np.array:
    return np.einsum("ijk,li->ljk", representation_3d, train_representations) / (
        np.linalg.norm(representation_3d, ord=2, axis=0)[np.newaxis, :, :]
        * np.linalg.norm(train_representations, axis=1, ord=2)[:, np.newaxis, np.newaxis]
    )


def patchwise_cosine_distance(representation_3d: np.array, representation_1d: np.array) -> np.array:
    return 1 - patchwise_cosine_similarity(representation_3d, representation_1d)


def patchwise_euclidean_distance(representation_3d: np.array, train_representations: np.array) -> np.array:
    n_dim, x_dim, y_dim = train_representations.shape[0], representation_3d.shape[1], representation_3d.shape[2]
    representation_scores = np.zeros((n_dim, x_dim, y_dim))
    for i, train_representation in enumerate(train_representations):
        representation_scores[i, :] = np.linalg.norm(
            representation_3d - train_representation[:, np.newaxis, np.newaxis], ord=2, axis=0
        )

    return representation_scores


def upsample_representation_image(sensor_resolution: List, representation_image: np.array) -> np.array:
    x_dim, y_dim = representation_image.shape
    upsample_resolution_y = int(sensor_resolution[0] / y_dim)
    upsample_resolution_x = int(sensor_resolution[1] / x_dim)
    return representation_image.repeat(upsample_resolution_x, axis=0).repeat(upsample_resolution_y, axis=1)


def compute_representation_score(
    logger: Logger,
    hidden_representation: np.array,
    sensor_resolution: List,
    score_fn_name: str,
    score_fn_mode: str,
    score_fn_params: Dict,
) -> np.array:
    if score_fn_name == "cosine_distance":
        score_fn = patchwise_cosine_distance
    elif score_fn_name == "cosine_similarity":
        score_fn = patchwise_cosine_similarity
    elif score_fn_name == "euclidean_distance":
        score_fn = patchwise_euclidean_distance
    else:
        raise ValueError(f"Score function name '{score_fn_name}' does not exist!")

    if score_fn_mode == "mean":
        representation_measure_fn = compute_mean_representation_score
    elif score_fn_mode == "maximum":
        representation_measure_fn = compute_maximum_representation_score
    elif score_fn_mode == "density":
        representation_measure_fn = compute_density_representation_score
    else:
        raise ValueError(f"Score function mode '{score_fn_mode}' does not exist!")

    return representation_measure_fn(logger, hidden_representation, sensor_resolution, score_fn, **score_fn_params)


def compute_density_representation_score(
    logger: Logger,
    hidden_representation: np.array,
    sensor_resolution: List,
    score_fn: callable,
    knn_number: int,
    distance_based: bool,
    **kwargs,
) -> np.array:
    if len(logger.train_data_representations) == 0:
        return np.zeros((sensor_resolution[1], sensor_resolution[0]))

    n_dim, c_dim = len(logger.train_data_representations), logger.train_data_representations[0].shape
    _, x_dim, y_dim = hidden_representation.shape

    representation_scores = score_fn(hidden_representation, np.array(logger.train_data_representations))

    representation_score = np.zeros((x_dim, y_dim))
    for x in range(x_dim):
        for y in range(y_dim):
            split_idx = np.minimum(knn_number, n_dim - 1)
            if distance_based:
                knn_patch_scores = np.partition(representation_scores[:, x, y], split_idx)[:split_idx]
            else:
                knn_patch_scores = np.partition(representation_scores[:, x, y], -split_idx)[-split_idx:]

            representation_score[x, y] = np.mean(knn_patch_scores)

    return upsample_representation_image(sensor_resolution, representation_score)


def compute_maximum_representation_score(
    logger: Logger,
    hidden_representation: np.array,
    sensor_resolution: List,
    score_fn: callable,
    **kwargs,
) -> np.array:
    _, x_dim, y_dim = hidden_representation.shape
    if len(logger.train_data_representations) == 0:
        return np.zeros((sensor_resolution[1], sensor_resolution[0]))

    representation_scores = score_fn(hidden_representation, np.array(logger.train_data_representations))
    representation_score = np.max(representation_scores, axis=0)

    return upsample_representation_image(sensor_resolution, representation_score)


def compute_mean_representation_score(
    logger: Logger,
    hidden_representation: np.array,
    sensor_resolution: List,
    score_fn: callable,
    **kwargs,
) -> np.array:
    if len(logger.train_data_representations) == 0:
        return np.zeros((sensor_resolution[1], sensor_resolution[0]))

    _, x_dim, y_dim = hidden_representation.shape

    representation_scores = score_fn(hidden_representation, np.array(logger.train_data_representations))
    representation_score = np.mean(representation_scores, axis=0)

    return upsample_representation_image(sensor_resolution, representation_score)


class Planner:
    def __init__(
        self,
        mapper: TerrainMapper,
        altitude: float,
        sensor_info: Dict,
        uav_specifications: Dict,
        objective_fn_name: str,
    ):
        self.planner_name = "planner"
        self.mapper = mapper
        self.altitude = altitude
        self.sensor_angle = sensor_info["angle"]
        self.sensor_resolution = sensor_info["resolution"]
        self.uav_specifications = uav_specifications
        self.objective_fn_name = objective_fn_name

    def get_schematic_image(self, uncertainty_image: np.array, representation_image: np.array) -> np.array:
        if self.objective_fn_name == "uncertainty":
            return uncertainty_image
        elif self.objective_fn_name == "representation":
            return representation_image
        else:
            raise NotImplementedError(
                f"Planning objective function '{self.objective_fn_name}' not implemented for {self.planner_name} planner"
            )

    def setup(self, **kwargs):
        pass

    def replan(self, budget: float, previous_pose: np.array, **kwargs) -> Optional[np.array]:
        raise NotImplementedError("Replan function not implemented!")
