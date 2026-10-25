"""xlerobot_vr Teleoperator の設定。

--teleop.<field>=<value> で CLI から変えられる。
自分の機体で計り直す前提の値には、その根拠を書いた。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from lerobot.teleoperators.config import TeleoperatorConfig


class IKBackend(str, Enum):
    """--teleop.ik_backend= の選択肢。draccus 0.10 は Literal を読めないので Enum にする。"""

    PYROKI = "pyroki"
    CLOSED_FORM = "closed_form"


@TeleoperatorConfig.register_subclass("xlerobot_quest")
@dataclass
class XLeRobotVRConfig(TeleoperatorConfig):
    """Quest → teleop_xr → IK → 17 キーの action。IK の既定は閉形式（closed_form）。"""

    # --- VR サーバ（teleop_xr） ---
    host: str = "0.0.0.0"
    port: int = 4443

    # --- IK ---
    # closed_form: 同梱の so101_chain（閉形式）で、腕ごとに解く
    # pyroki: teleop_xr の PyrokiSolver で、両腕を 1 つの最適化として解く
    # 既定は closed_form。pyroki は起動のたびに JIT コンパイルで数十秒待つ
    ik_backend: IKBackend = IKBackend.CLOSED_FORM

    # --- gripper ---
    # 開き切った状態と閉じ切った状態の値（0..100）。
    # どちらが開きかはキャリブレーションの取り方で変わるので、実機で確認する
    gripper_open: float = 100.0
    gripper_closed: float = 0.0

    # --- 操作 ---
    # head を HMD の向きに追従させる
    follow_head: bool = True
    # 左スティックで動かすベースの速さ（倒し切ったとき）とデッドゾーン
    base_xy_speed_mps: float = 0.2
    base_theta_speed_dps: float = 60.0
    stick_deadzone: float = 0.15

    # --- 記録 ---
    # lerobot-record でエピソードを終えたら、腕と head をこの姿勢へ戻す（host の --rest.path と同じ JSON）。
    # None なら戻さない
    rest_path: str | None = None

    # --- 安全 ---
    # 1 秒あたりの関節角の変化上限（度/秒）。初回は 120 まで下げる
    max_joint_speed_dps: float = 180.0
    # VR 入力がこの時間途絶えたら追従を止めて現在値を保持する。
    # Quest の入力は 0.30〜0.33 秒途切れることがある（50 エピソードの記録で 21 回）。それでは切らない
    stale_timeout_s: float = 0.5
    # 制御周期。lerobot-teleoperate の --fps と揃える
    fps: int = 30

    def __post_init__(self) -> None:
        # 文字列で作ったときも Enum にそろえる。知らない名前はここで ValueError になる
        self.ik_backend = IKBackend(self.ik_backend)
