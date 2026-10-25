"""公式の XLerobotClient に、lerobot-record が数えるカメラの一覧と、lerobot-replay の値の変換を足した client。"""

from __future__ import annotations

from typing import Any

from lerobot_robot_xlerobot.xlerobot_client import XLerobotClient

from .config_xlerobot_book_client import XLerobotBookClientConfig


class XLerobotBookClient(XLerobotClient):
    config_class = XLerobotBookClientConfig
    # name は公式と同じ xlerobot_client のまま変えない。
    # データセットの robot_type になり、merge と --resume は記録済みのデータセットとの一致を求める

    def __init__(self, config: XLerobotBookClientConfig) -> None:
        super().__init__(config)
        # 公式の client は robot_type を config.type（xlerobot_book_client）にする。
        # --resume は記録済みの robot_type（name）とこれを比べるので、name にそろえる
        self.robot_type = self.name

    # lerobot-record（v0.6.1）はデータセットを作るときに len(robot.cameras) を数える。
    # 映像はホストが撮って送るので、クライアントはカメラを開かず、設定だけを返す
    @property
    def cameras(self) -> dict:
        return self.config.cameras

    # lerobot-replay はデータセットの action を Tensor のまま渡す。公式の client は値を直さずに json.dumps するので止まる。
    # lerobot v0.6.1 の LeKiwiClient と同じく float に直してから送る。公式に同じ修正が入れば、この上書きは消せる
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return super().send_action({key: float(value) for key, value in action.items()})
