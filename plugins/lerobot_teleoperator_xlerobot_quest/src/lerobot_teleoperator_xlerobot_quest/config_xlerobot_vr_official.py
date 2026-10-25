"""xlerobot_vr_official Teleoperator の設定。

公式の XLerobotVRTeleopConfig の項目（kp、xlevr_path など）をそのまま継ぎ、
公式の Teleoperator の動かし方を標準のコマンドで試すときの安全のための項目を足す。
--teleop.<field>=<value> で CLI から変えられる。
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig

# 公式の VR の plugin（XLeRobot リポジトリの software/plugins/lerobot_teleoperator_xlerobot_vr）
from lerobot_teleoperator_xlerobot_vr.configuration_xlerobot_vr import XLerobotVRTeleopConfig


@TeleoperatorConfig.register_subclass("xlerobot_vr_official")
@dataclass
class XLeRobotVROfficialConfig(XLerobotVRTeleopConfig):
    """公式の XLeVR と動かし方のまま、本書の host と client で動かす。"""

    # 公式リポジトリの XLeVR ディレクトリ。install.sh の clone 先を既定にする。
    # 公式の既定値（/your_local_DIR/...）は書き換える前提の仮の値
    xlevr_path: str = "~/xlerobot-book-env/XLeRobot/XLeVR"

    # 起動直後に腕が今の姿勢からゼロ姿勢へ移る間だけ、1周あたりの変化をこの値で丸める。
    # 単位はロボットの action と同じ（公式 host は -100〜100 の正規化値）。0 で丸めない。
    # 公式の動かし方は腕をゼロ姿勢から始める前提で、最初の周の腕の指令が全関節0になる（head は今の位置を保つ）。
    # 公式はこれを1周で動かすが、速度の制限が無く、腕が一気に振れる
    start_max_step: float = 2.0

    # 右コントローラの入力がこの秒数届かなければ、ベースの速度を0にする。0 で止めない。
    # 公式の VR モニタは接続が切れても最後の入力を持ち続け、スティックを倒したまま切れると
    # 同じ速度を送り続ける。指令は届き続けるので、host の watchdog では止まらない
    stale_timeout_s: float = 0.5
