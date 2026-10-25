"""コントローラの姿勢を、SO-101 が追従できる pitch と roll に分ける。

腕の yaw は手先の位置が決める。
手を垂直軸まわりに回しても、その分は姿勢としては渡さない。
pitch は手の前方軸が水平から何度上がったか。
roll はその前方軸まわりのひねり。

分け方は NektoDron/lekiwi-vr-teleop の operator.HandOrientation と同じ。
quaternion は、teleop_xr が ROS 規約（X 前、Y 左、Z 上）へ変換したあとの grip 姿勢で、並びは xyzw。
この変換では、コントローラの前方が +X に載る。
"""

from __future__ import annotations

import math

import numpy as np
from lerobot.utils.rotation import Rotation

_FORWARD = np.array([1.0, 0.0, 0.0])


def elevation_deg(axis: np.ndarray) -> float:
    """方向が水平から何度上を向いているか。真上でも真下でも定義できる。"""
    return math.degrees(math.atan2(float(axis[2]), math.hypot(float(axis[0]), float(axis[1]))))


def twist_deg(delta: Rotation, axis: np.ndarray) -> float:
    """delta のうち、単位ベクトル axis まわりの回転だけを度で返す。

    傾きとひねりが混ざった回転から、ひねりだけを残す。
    axis と直交する軸まわりに 180 度回っているときはひねりが定まらないので 0 にする。
    """
    x, y, z, w = delta.as_quat()
    along = float(np.dot([x, y, z], axis))
    if math.hypot(along, w) < 1e-6:
        return 0.0
    angle = math.degrees(2.0 * math.atan2(along, w))
    return (angle + 180.0) % 360.0 - 180.0


def orientation_offsets(origin_xyzw: np.ndarray, current_xyzw: np.ndarray) -> tuple[float, float]:
    """engage のときの姿勢から見た (pitch, roll) の変化（度）。

    絶対姿勢を手首にコピーせず、差だけを足す。そのため engage した瞬間は腕が動かない。
    """
    origin = Rotation.from_quat(origin_xyzw)
    current = Rotation.from_quat(current_xyzw)
    axis = current.apply(_FORWARD)
    pitch = elevation_deg(axis) - elevation_deg(origin.apply(_FORWARD))
    return pitch, twist_deg(current * origin.inv(), axis)
