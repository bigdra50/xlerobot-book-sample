"""この plugin の内側と外側の境界で使う型と定数。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- action のキー -----------------------------------------------------------
# 公式 XLerobot._state_ft と同じ名前・同じ順。
# dataset の action.names はこの順になり、学習済みポリシーはこの順でしか読めない。

ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# IK が解く関節。gripper はトリガーで直接決める
IK_JOINTS: tuple[str, ...] = ARM_JOINTS[:-1]

SIDES: tuple[str, ...] = ("left", "right")

LEFT_ARM_KEYS: tuple[str, ...] = tuple(f"left_arm_{j}.pos" for j in ARM_JOINTS)
RIGHT_ARM_KEYS: tuple[str, ...] = tuple(f"right_arm_{j}.pos" for j in ARM_JOINTS)
HEAD_KEYS: tuple[str, ...] = ("head_motor_1.pos", "head_motor_2.pos")  # pan, tilt
BASE_KEYS: tuple[str, ...] = ("x.vel", "y.vel", "theta.vel")
ACTION_KEYS: tuple[str, ...] = LEFT_ARM_KEYS + RIGHT_ARM_KEYS + HEAD_KEYS + BASE_KEYS
assert len(ACTION_KEYS) == 17

# キャリブレーション JSON（公式 17 台 1 ファイル形式）のキー。".pos" を外した名前
MOTOR_NAMES: tuple[str, ...] = tuple(
    k.removesuffix(".pos") for k in LEFT_ARM_KEYS + RIGHT_ARM_KEYS + HEAD_KEYS
)


def arm_motor(side: str, joint: str) -> str:
    """キャリブレーション JSON のモーター名。action のキーは、これに ".pos" を付けたもの。"""
    return f"{side}_arm_{joint}"


def ik_joint_name(side: str, joint: str) -> str:
    """両腕 URDF の関節名。単腕 URDF の名前に side を前置する。"""
    return f"{side}_{joint}"


# --- 制限 --------------------------------------------------------------


@dataclass(frozen=True)
class JointLimits:
    """モーターの度で表した上下限。lower <= upper"""

    lower_deg: float
    upper_deg: float

    def clamp(self, deg: float) -> float:
        return min(self.upper_deg, max(self.lower_deg, deg))


# --- VR 入力のスナップショット ------------------------------------------------
# teleop_xr の XRState から、この plugin が使う分だけを取り出した形。
# ボタン添字は WebXR xr-standard: 0 trigger / 1 squeeze / 2 touchpad / 3 thumbstick / 4 A,X / 5 B,Y


@dataclass(frozen=True)
class Controller:
    trigger: float = 0.0  # 0..1
    squeeze: bool = False  # grip
    primary: bool = False  # A（右）/ X（左）
    secondary: bool = False  # B（右）/ Y（左）
    stick_x: float = 0.0  # -1..1
    stick_y: float = 0.0  # -1..1
    has_pose: bool = False


@dataclass(frozen=True)
class HeadPose:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    has_pose: bool = False


@dataclass(frozen=True)
class VRSnapshot:
    left: Controller
    right: Controller
    head: HeadPose


# --- 状態 --------------------------------------------------------------------


class EngagePhase(str, Enum):
    IDLE = "idle"  # 追従していない。現在値を保持
    ENGAGED = "engaged"  # コントローラに追従中
    PAUSED = "paused"  # grip 中。離したら基準を取り直して ENGAGED へ
