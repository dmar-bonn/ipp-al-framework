import json
import logging
from typing import Dict

from telegram import Bot

from utils.notifiers import Notifier

logger = logging.getLogger(__name__)


class TelegramNotifier(Notifier):
    def __init__(
        self,
        experiment_name: str,
        token: str,
        chat_id: str,
        cfg: Dict = None,
        model_cfg: Dict = None,
        verbose: bool = True,
    ):
        super(TelegramNotifier, self).__init__(experiment_name, cfg, model_cfg, verbose)

        self.notifier_name = "Telegram"
        self.chat_id = chat_id
        self.bot = Bot(token)

    def format_info_dict(self, info_name: str, additional_info: Dict) -> str:
        return f"{info_name}:\n{json.dumps(additional_info, indent=2, default=str)}".strip()[-4096:]

    def send_message(self, message: str):
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            print(f"While sending a {self.notifier_name} notification, the following error occurred:\n{e}")
