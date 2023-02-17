from typing import Dict


class Notifier:
    def __init__(
        self,
        experiment_name: str,
        cfg: Dict = None,
        model_cfg: Dict = None,
        verbose: bool = True,
    ):
        self.notifier_name = "Base"
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.cfg = cfg
        self.model_cfg = model_cfg

    def start_experiment(self):
        """Record the start of the experiment and send message via notifier."""
        start_text = f"Start '{self.experiment_name}' experiment!"
        self.send_message(start_text)

        if self.verbose:
            start_text = self.format_info_dict("Experiment config", self.cfg)
            self.send_message(start_text)

            start_text = self.format_info_dict("Model config", self.model_cfg)
            self.send_message(start_text)

    def finished_iteration(self, iteration_id: int, additional_info: Dict = None):
        """Record the completion of an iteration and send message via notifier."""
        iteration_text = f"Completed iteration {iteration_id} of '{self.experiment_name}' experiment!"
        self.send_message(iteration_text)

        if self.verbose:
            iteration_text = self.format_info_dict("Additional information", additional_info)
            self.send_message(iteration_text)

    def finish_experiment(self, additional_info: Dict = None):
        """Record the termination of the experiment. Summarize results and send message via notifier."""
        finish_text = f"Finished '{self.experiment_name}' experiment!"
        self.send_message(finish_text)

        if self.verbose:
            finish_text = self.format_info_dict("Additional information", additional_info)
            self.send_message(finish_text)

    def failed_experiment(self, e: Exception):
        failed_text = f"Experiment '{self.experiment_name}' failed with the following exception:\n{e}"
        self.send_message(failed_text)

    def send_message(self, message: str):
        raise NotImplementedError(f"{self.notifier_name} does not implement 'send_message' function!")

    def format_info_dict(self, info_name: str, additional_info: Dict) -> str:
        raise NotImplementedError(f"{self.notifier_name} does not implement 'format_info_dict' function!")
