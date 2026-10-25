"""公式 XLeRobot の client を、lerobot-record と lerobot-rollout で使えるようにする plugin。

公式の lerobot_robot_xlerobot（XLeRobot リポジトリの software/plugins）は、
xlerobot_client の型を登録するが、client を package から export しておらず、カメラの一覧も持たない。
そのため --robot.type=xlerobot_client は lerobot の CLI で構築できない。
この package は公式の client を継いで足りない分を足し、xlerobot_book_client の名前で登録する。
"""

from __future__ import annotations

# 公式の plugin。import した時点で xlerobot と xlerobot_client が --robot.type に登録される
from lerobot_robot_xlerobot import XLerobot

from . import observation_bus

# lerobot は config クラス名から Config を除いた名前をこの package から探すので、どれもここで export する
from .config_xlerobot_book_client import XLerobotBookClientConfig
from .xlerobot_book_client import XLerobotBookClient

# 最後の観測を Teleoperator へ差し出す（理由は observation_bus の冒頭）
observation_bus.install(XLerobot)
observation_bus.install(XLerobotBookClient)

__all__ = [
    "XLerobotBookClient",
    "XLerobotBookClientConfig",
    "observation_bus",
]
