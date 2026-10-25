"""Quest の VR 入力で XLeRobot の 17 軸を動かす LeRobot Teleoperator plugin。

    lerobot-teleoperate --robot.type=xlerobot_book_client --robot.remote_ip=<ホスト> \
        --teleop.type=xlerobot_quest --teleop.calibration_dir=<公式 17 台 JSON の置き場> --teleop.id=<robot id>

lerobot v0.6.1 は `lerobot_teleoperator_` で始まる distribution を起動時に import する。
import 時に読むのは config だけにしてある。
teleop_xr / jax は VR サーバを実際に立てる connect() まで読まない。
CLI の --help や record の引数検査を、IK 依存の無い環境でも通すため。

--teleop.type=xlerobot_vr_official で、公式の XLerobotVRTeleop（公式の VR の plugin）も選べる。
公式の Teleoperator の動かし方を、本書の host と client のまま試すためのもの（xlerobot_vr_official.py）。
"""

from __future__ import annotations

from .config_xlerobot_vr import XLeRobotVRConfig
from .config_xlerobot_vr_official import XLeRobotVROfficialConfig

__all__ = ["XLeRobotVR", "XLeRobotVRConfig", "XLeRobotVROfficial", "XLeRobotVROfficialConfig"]


def __getattr__(name: str):
    # lerobot の make_device_from_device_class は、config クラスのモジュールの
    # 親パッケージ（ここ）から "XLeRobotVR" を getattr で探す。
    # 遅延 import にしておくと、その瞬間まで teleop_xr を読まずに済む。
    if name == "XLeRobotVR":
        from .xlerobot_vr import XLeRobotVR

        return XLeRobotVR
    if name == "XLeRobotVROfficial":
        from .xlerobot_vr_official import XLeRobotVROfficial

        return XLeRobotVROfficial
    raise AttributeError(name)
