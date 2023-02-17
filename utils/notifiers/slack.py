import json
import logging
from typing import Dict

import requests

from utils.notifiers import Notifier

logger = logging.getLogger(__name__)


class SlackNotifier(Notifier):
    def __init__(
        self,
        experiment_name: str,
        webhook: str,
        bot_name: str,
        icon: str = ":robot_face:",
        cfg: Dict = None,
        model_cfg: Dict = None,
        verbose: bool = True,
    ):
        super(SlackNotifier, self).__init__(experiment_name, cfg, model_cfg, verbose)

        self.notifier_name = "Slack"
        self.webhook = webhook
        self.icon = icon
        self.bot_name = bot_name

    def format_info_dict(self, info_name: str, additional_info: Dict) -> str:
        return f"{info_name}:\n{json.dumps(additional_info, indent=2, default=str)}".strip()

    def send_message(self, message: str):
        """Prepare and send the POST request to Slack."""
        data = {
            "text": message,
            "icon_emoji": self.icon,
            "username": self.bot_name,
        }

        try:
            requests.post(self.webhook, headers={"Content-type": "application/json"}, json=data)
        except Exception as e:
            print(f"While sending a {self.notifier_name} notification, the following error occurred:\n{e}")
