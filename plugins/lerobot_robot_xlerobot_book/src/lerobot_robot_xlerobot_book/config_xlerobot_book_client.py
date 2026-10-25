"""--robot.type=xlerobot_book_client の設定。項目は公式の XLerobotClientConfig と同じ。"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig
from lerobot_robot_xlerobot.config_xlerobot import XLerobotClientConfig


@RobotConfig.register_subclass("xlerobot_book_client")
@dataclass
class XLerobotBookClientConfig(XLerobotClientConfig):
    pass
