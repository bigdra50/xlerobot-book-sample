"""host の関節の値を、公式の XLeRobot の URDF（Vector-Wangel/XLeRobot の ManiSkill 用）の関節の角度 [rad] にする。

host の値は、キャリブレーションの範囲の中央を 0 とする度に直してから（navgate.arm_client.norm_to_deg）、
URDF の角度 = 符号 × 度 + 基準の角度 で換算する。
腕の基準の角度（肩の持ち上げ 84.7°、肘 −81.1°、手首の曲げ −19.0°）は、筆者が公式 URDF を VR テレオペで
使ったときに合わせた値を出発点にしている。自分の機体で合っているかは、head カメラの画像との重ね合わせで確かめる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from navgate.arm_client import norm_to_deg


@dataclass(frozen=True)
class JointMap:
    urdf: str
    sign: float = 1.0
    offset_deg: float = 0.0


_ARM = {
    "shoulder_pan": ("Rotation", 0.0),
    "shoulder_lift": ("Pitch", 84.7),
    "elbow_flex": ("Elbow", -81.1),
    "wrist_flex": ("Wrist_Pitch", -19.0),
    "wrist_roll": ("Wrist_Roll", 0.0),
    "gripper": ("Jaw", 0.0),
}

HOST_TO_URDF: dict[str, JointMap] = {
    **{f"right_arm_{j}": JointMap(urdf, offset_deg=off) for j, (urdf, off) in _ARM.items()},
    **{f"left_arm_{j}": JointMap(f"{urdf}_2", offset_deg=off) for j, (urdf, off) in _ARM.items()},
    # head_motor_1 は正が右回り（head カメラの画像で確かめた）。URDF の head_pan_joint は z 軸まわりで左回りが正
    "head_motor_1": JointMap("head_pan_joint", sign=-1.0),
    # head_motor_2 は正が下向き。URDF の head_tilt_joint は y 軸まわりで、正が下向き。
    # 基準の角度は、head カメラに写る上段トレイの前縁と、URDF の台車のメッシュの前縁を重ねて求めた（−20°）
    "head_motor_2": JointMap("head_tilt_joint", offset_deg=-20.0),
}


def to_urdf(host: dict[str, float], cal: dict, table: dict[str, JointMap] = HOST_TO_URDF) -> dict[str, float]:
    """host の値（腕と head は −100〜100、gripper は 0〜100）の辞書から、URDF の関節名と角度 [rad] の辞書を作る。"""
    out: dict[str, float] = {}
    for name, value in host.items():
        m = table.get(name)
        if m is None or name not in cal:
            continue
        out[m.urdf] = math.radians(m.sign * norm_to_deg(name, value, cal) + m.offset_deg)
    return out
