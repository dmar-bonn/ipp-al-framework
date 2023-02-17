import copy
from typing import Dict

from agri_semantics.datasets import get_data_module
from agri_semantics.models import get_model
from agri_semantics.models.models import EnsembleNet
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule


class Learner:
    def __init__(self, cfg: Dict, weights_path: str, logger_name: str, model_id: str = ""):
        self.cfg = cfg
        self.weights_path = weights_path
        self.logger_name = logger_name
        self.model_id = model_id
        self.patience = cfg["train"]["patience"]
        self.task = cfg["model"]["task"]

        self.model = None
        self.trainer = None
        self.data_module = None
        self.test_statistics = {}

    @property
    def monitoring_metric(self) -> str:
        if self.task == "classification":
            return "mIoU"
        else:
            raise NotImplementedError(f"No early stopping metric implemented for {self.task} task!")

    @property
    def monitoring_mode(self) -> str:
        if self.task == "classification":
            return "max"
        else:
            raise NotImplementedError(f"No monitoring mode implemented for {self.task} task!")

    def setup_trainer(self, iter_count: int) -> Trainer:
        early_stopping = EarlyStopping(
            monitor=f"Validation/{self.monitoring_metric}",
            min_delta=0.00,
            patience=self.patience,
            verbose=False,
            mode=self.monitoring_mode,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_saver = ModelCheckpoint(
            monitor=f"Validation/{self.monitoring_metric}",
            filename=f"{self.cfg['experiment']['id']}_model_{self.model_id}_iter{iter_count}_best",
            mode=self.monitoring_mode,
            save_last=True,
        )
        tb_logger = pl_loggers.TensorBoardLogger(
            f"experiments/{self.cfg['experiment']['id']}",
            name=f"{self.cfg['model']['name']}_{self.model_id}",
            version=iter_count,
            default_hp_metric=False,
        )

        trainer = Trainer(
            gpus=self.cfg["train"]["n_gpus"],
            logger=tb_logger,
            max_epochs=self.cfg["train"]["max_epoch"],
            callbacks=[lr_monitor, checkpoint_saver, early_stopping],
            log_every_n_steps=1,
        )

        return trainer

    def setup_data_module(self, stage: str = None) -> LightningDataModule:
        data_module = get_data_module(self.cfg)
        data_module.setup(stage)

        return data_module

    def setup_model(self, iter_count: int = 0, num_train_data: int = 1) -> LightningModule:
        raise NotImplementedError("Learner does not implement 'setup_model()' function!")

    def train(self, iter_count: int) -> LightningModule:
        raise NotImplementedError("Learner does not implement 'train()' function!")

    def evaluate(self):
        self.data_module = self.setup_data_module(stage=None)
        self.trainer.test(self.model, self.data_module)
        self.track_test_statistics()

    def track_classification_metrics(self, num_train_data: int):
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/Acc", self.model.test_evaluation_metrics["Test/Acc"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/mIoU", self.model.test_evaluation_metrics["Test/mIoU"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/ECE", self.model.test_evaluation_metrics["Test/ECE"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/Precision", self.model.test_evaluation_metrics["Test/Precision"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/Recall", self.model.test_evaluation_metrics["Test/Recall"], num_train_data
        )
        self.model.logger.experiment.add_scalar(
            "ActiveLearning/F1-Score", self.model.test_evaluation_metrics["Test/F1-Score"], num_train_data
        )

    def track_test_statistics(self):
        num_train_data = len(self.data_module.train_dataloader().dataset)
        self.test_statistics[num_train_data] = self.model.test_evaluation_metrics.copy()

        if self.task == "classification":
            self.track_classification_metrics(num_train_data)


class ModelLearner(Learner):
    def __init__(self, cfg: Dict, weights_path: str, logger_name: str, model_id: str = ""):
        super(ModelLearner, self).__init__(cfg, weights_path, logger_name, model_id=model_id)

        self.model = self.setup_model(iter_count=0)
        self.trainer = self.setup_trainer(0)

    def setup_model(self, iter_count: int = 0, num_train_data: int = 1) -> LightningModule:
        model = get_model(
            self.cfg, al_logger_name=self.logger_name, al_iteration=iter_count, num_train_data=num_train_data
        )
        if self.weights_path:
            model = model.load_from_checkpoint(
                self.weights_path,
                cfg=self.cfg,
                al_logger_name=self.logger_name,
                al_iteration=iter_count,
                num_train_data=num_train_data,
            )
            if self.cfg["model"]["num_classes_pretrained"] != self.cfg["model"]["num_classes"]:
                model.replace_output_layer()

        return model

    def retrain_model(self, iter_count: int, num_train_data: int):
        self.model = self.setup_model(iter_count=iter_count, num_train_data=num_train_data)
        self.trainer.fit(self.model, self.data_module)
        self.model = self.model.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path,
            cfg=self.cfg_fine_tuned,
            al_logger_name=self.logger_name,
            al_iteration=iter_count,
            num_train_data=num_train_data,
        )

    @property
    def cfg_fine_tuned(self) -> Dict:
        cft_fine_tuned = copy.deepcopy(self.cfg)
        cft_fine_tuned["model"]["num_classes_pretrained"] = cft_fine_tuned["model"]["num_classes"]
        return cft_fine_tuned

    def train(self, iter_count: int) -> LightningModule:
        print(f"START {self.cfg['model']['name']}_{self.model_id} TRAINING")

        self.trainer = self.setup_trainer(iter_count)
        self.data_module = self.setup_data_module(stage=None)
        self.retrain_model(iter_count, len(self.data_module.train_dataloader().dataset))

        return self.model


class EnsembleLearner(Learner):
    def __init__(self, cfg: Dict, weights_path: str, logger_name: str):
        super(EnsembleLearner, self).__init__(cfg, weights_path, logger_name, model_id="ensemble")

        self.num_models = cfg["model"]["num_models"]
        self.model_learners = [
            ModelLearner(self.cfg, self.weights_path, logger_name, model_id=str(i)) for i in range(self.num_models)
        ]

        self.model = self.setup_model(iter_count=0)
        self.trainer = self.setup_trainer(0)

    def setup_model(self, iter_count: int = 0, num_train_data: int = 1) -> LightningModule:
        return EnsembleNet(
            self.cfg,
            [copy.deepcopy(ml.model) for ml in self.model_learners],
            al_logger_name=self.logger_name,
            al_iteration=iter_count,
        )

    def train(self, iter_count: int) -> LightningModule:
        print(f"START {self.cfg['model']['name']}_ensemble_{self.num_models} TRAINING")

        self.trainer = self.setup_trainer(iter_count)
        for model_learner in self.model_learners:
            model_learner.train(iter_count)

        self.model = self.setup_model(iter_count=iter_count)
        return self.model
