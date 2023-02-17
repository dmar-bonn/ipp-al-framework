from typing import Dict

from active_learning.learners import EnsembleLearner, ModelLearner, Learner


def get_learner(model_cfg: Dict, weights_path: str, logger_name: str, model_id: str = "0") -> Learner:
    if model_cfg["model"]["ensemble_model"]:
        return EnsembleLearner(model_cfg, weights_path, logger_name)
    else:
        return ModelLearner(model_cfg, weights_path, logger_name, model_id=model_id)
