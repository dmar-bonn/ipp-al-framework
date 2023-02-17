import os
from typing import List

import numpy as np


def get_fov(pose: np.array, sensor_angle: List, gsd: float, world_range: List):
    half_fov_size = pose[2] * np.tan(np.deg2rad(sensor_angle))

    # fov in world coordinate frame
    lu = [pose[0] - half_fov_size[0], pose[1] - half_fov_size[1]]
    ru = [pose[0] + half_fov_size[0], pose[1] - half_fov_size[1]]
    rd = [pose[0] + half_fov_size[0], pose[1] + half_fov_size[1]]
    ld = [pose[0] - half_fov_size[0], pose[1] + half_fov_size[1]]
    corner_list = np.array([lu, ru, rd, ld])

    # fov index in orthomosaic space
    lu_index = [np.floor(lu[0] / gsd).astype(int), np.floor(lu[1] / gsd).astype(int)]
    ru_index = [np.ceil(ru[0] / gsd).astype(int), np.floor(ru[1] / gsd).astype(int)]
    rd_index = [np.ceil(rd[0] / gsd).astype(int), np.ceil(rd[1] / gsd).astype(int)]
    ld_index = [np.floor(ld[0] / gsd).astype(int), np.ceil(ld[1] / gsd).astype(int)]

    index_list = np.array([lu_index, ru_index, rd_index, ld_index])
    min_x = np.min(index_list[:, 0])
    max_x = np.max(index_list[:, 0])
    min_y = np.min(index_list[:, 1])
    max_y = np.max(index_list[:, 1])

    if np.any(np.array([min_x, min_y]) < np.array([0, 0])) or np.any(np.array([max_x, max_y]) > np.array(world_range)):
        raise ValueError(f"Invalid measurement! Measurement out of environment bounds.")

    return corner_list, [min_x, max_x, min_y, max_y]


def load_from_env(env_var_name: str, data_type: callable, default=None):
    if env_var_name in os.environ and os.environ[env_var_name] != "":
        value = os.environ[env_var_name]
        if data_type == bool:
            if value.lower() == "true":
                value = True
            else:
                value = False
        else:
            value = data_type(value)
        return value
    elif env_var_name not in os.environ and default is None:
        raise ValueError(
            f"Could not find environment variable '{env_var_name}'. "
            f"Please check .env file or provide a default value when calling load_from_env()."
        )
    return default
